import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from architecture.network import DimReduction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TransformerPatchEncoder_old(nn.Module):
    """
    Memory-efficient, checkpoint-compatible Transformer patch encoder.

    IMPORTANT:
    - Same trainable architecture as the original.
    - Same state_dict parameter names.
    - Existing pretrained checkpoints can load with strict=True.
    - Global attention is preserved:
        each query patch still attends to ALL patches.
    - Only the execution is chunked to reduce peak CUDA memory.

    Input:
        x: [B, N, input_dim]

    Output:
        z: [B, N, model_dim]
    """

    def __init__(
        self,
        input_dim,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4,
        dropout=0.1,
        attention_chunk_size=4,
        ffn_chunk_size=4,
        use_gradient_checkpointing=False,
    ):
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim={model_dim} must be divisible by "
                f"num_heads={num_heads}."
            )

        # ============================================================
        # EXACT SAME TRAINABLE MODULES AS ORIGINAL
        # ============================================================

        self.input_projection = (
            nn.Identity()
            if input_dim == model_dim
            else nn.Linear(input_dim, model_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(model_dim)

        # Runtime-only settings.
        # These do NOT appear in state_dict.
        self.attention_chunk_size = attention_chunk_size
        self.ffn_chunk_size = ffn_chunk_size
        self.use_gradient_checkpointing = use_gradient_checkpointing


    def _chunked_self_attention(
        self,
        layer,
        x,
        padding_mask=None,
    ):
        """
        Global self-attention with chunked queries.

        K and V are projected ONCE for the full sequence.

        Each query chunk still attends to all N patches.

        Uses the EXISTING MultiheadAttention parameters:
            in_proj_weight
            in_proj_bias
            out_proj.weight
            out_proj.bias
        """

        B, N, D = x.shape

        self_attn = layer.self_attn

        num_heads = self_attn.num_heads
        head_dim = D // num_heads

        chunk_size = self.attention_chunk_size

        if (
            chunk_size is None
            or chunk_size <= 0
        ):
            chunk_size = N

        # ============================================================
        # Existing pretrained Q/K/V projection parameters
        # ============================================================

        W = self_attn.in_proj_weight
        b = self_attn.in_proj_bias

        W_q = W[:D]
        W_k = W[D:2 * D]
        W_v = W[2 * D:]

        if b is not None:
            b_q = b[:D]
            b_k = b[D:2 * D]
            b_v = b[2 * D:]
        else:
            b_q = None
            b_k = None
            b_v = None

        # ============================================================
        # Project FULL K and V only once
        # ============================================================

        k = F.linear(
            x,
            W_k,
            b_k,
        )

        v = F.linear(
            x,
            W_v,
            b_v,
        )

        # [B, N, D] -> [B, H, N, Dh]

        k = (
            k.view(
                B,
                N,
                num_heads,
                head_dim,
            )
            .transpose(1, 2)
        )

        v = (
            v.view(
                B,
                N,
                num_heads,
                head_dim,
            )
            .transpose(1, 2)
        )

        # ============================================================
        # Convert padding mask for scaled_dot_product_attention
        # ============================================================

        attn_mask = None

        if padding_mask is not None:
            # MultiheadAttention convention:
            #     True = ignore
            #
            # SDPA Boolean mask convention:
            #     True = allowed
            #
            # [B, N] -> [B, 1, 1, N]

            attn_mask = (
                ~padding_mask
            )[:, None, None, :]

        # ============================================================
        # Process queries in chunks
        # ============================================================

        outputs = []

        dropout_p = (
            self_attn.dropout
            if self.training
            else 0.0
        )

        for start in range(
            0,
            N,
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                N,
            )

            x_query = x[
                :,
                start:end,
                :
            ]

            # --------------------------------------------------------
            # Project only current Q chunk
            # --------------------------------------------------------

            q = F.linear(
                x_query,
                W_q,
                b_q,
            )

            C = end - start

            q = (
                q.view(
                    B,
                    C,
                    num_heads,
                    head_dim,
                )
                .transpose(1, 2)
            )

            # --------------------------------------------------------
            # Global attention:
            #
            # Q: [B, H, C, Dh]
            # K: [B, H, N, Dh]
            # V: [B, H, N, Dh]
            # --------------------------------------------------------

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

            # [B, H, C, Dh] -> [B, C, D]

            out = (
                out.transpose(1, 2)
                .contiguous()
                .view(
                    B,
                    C,
                    D,
                )
            )

            # --------------------------------------------------------
            # SAME original output projection
            # --------------------------------------------------------

            out = F.linear(
                out,
                self_attn.out_proj.weight,
                self_attn.out_proj.bias,
            )

            outputs.append(
                out
            )

        attn_output = torch.cat(
            outputs,
            dim=1,
        )

        # Same dropout used by original TransformerEncoderLayer.
        return layer.dropout1(
            attn_output
        )


    def _chunked_ffn(
        self,
        layer,
        x,
    ):
        """
        Apply the EXISTING Transformer FFN in chunks over patches.

        This is mathematically safe because the FFN is token-wise:

            FFN(x_i) = W2 GELU(W1 x_i + b1) + b2

        No token interacts with another token inside the FFN.

        Uses the SAME:
            linear1
            activation
            dropout
            linear2
            dropout2
        """

        B, N, D = x.shape

        chunk_size = self.ffn_chunk_size

        if (
            chunk_size is None
            or chunk_size <= 0
        ):
            chunk_size = N

        outputs = []

        for start in range(
            0,
            N,
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                N,
            )

            x_chunk = x[
                :,
                start:end,
                :
            ]

            ff_chunk = layer.linear1(
                x_chunk
            )

            ff_chunk = layer.activation(
                ff_chunk
            )

            ff_chunk = layer.dropout(
                ff_chunk
            )

            ff_chunk = layer.linear2(
                ff_chunk
            )

            ff_chunk = layer.dropout2(
                ff_chunk
            )

            outputs.append(
                ff_chunk
            )

        return torch.cat(
            outputs,
            dim=1,
        )


    def _forward_layer(
        self,
        layer,
        x,
        padding_mask=None,
    ):
        """
        Equivalent to nn.TransformerEncoderLayer.forward()
        for norm_first=True:

            x = x + SA(norm1(x))
            x = x + FFN(norm2(x))

        using the same original parameters.
        """

        if not layer.norm_first:
            raise RuntimeError(
                "This implementation expects norm_first=True, "
                "matching the original encoder."
            )

        # ============================================================
        # Self-attention block
        # ============================================================

        norm_x = layer.norm1(
            x
        )

        attn_output = self._chunked_self_attention(
            layer=layer,
            x=norm_x,
            padding_mask=padding_mask,
        )

        x = (
            x
            + attn_output
        )

        # ============================================================
        # Feed-forward block
        # ============================================================

        norm_x = layer.norm2(
            x
        )

        ff = self._chunked_ffn(
            layer=layer,
            x=norm_x,
        )

        x = (
            x
            + ff
        )

        return x


    def _checkpointed_forward_layer(
        self,
        layer,
        x,
        padding_mask,
    ):
        """
        Gradient-checkpoint one encoder layer.

        This saves training activation memory.
        No parameters or model behavior are added.
        """

        def layer_forward(inp):
            return self._forward_layer(
                layer=layer,
                x=inp,
                padding_mask=padding_mask,
            )

        return checkpoint(
            layer_forward,
            x,
            use_reentrant=False,
        )


    def forward(
        self,
        x,
        padding_mask=None,
    ):
        """
        padding_mask:
            Optional Boolean tensor [B, N].

            True means that patch is padding and should be ignored.
        """

        # ============================================================
        # Original input projection
        # ============================================================

        x = self.input_projection(
            x
        )

        # ============================================================
        # Same original Transformer layers
        # ============================================================

        for layer in self.encoder.layers:

            if (
                self.training
                and self.use_gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                x = self._checkpointed_forward_layer(
                    layer=layer,
                    x=x,
                    padding_mask=padding_mask,
                )

            else:
                x = self._forward_layer(
                    layer=layer,
                    x=x,
                    padding_mask=padding_mask,
                )

        # Your original TransformerEncoder normally has no
        # encoder-level final norm, but preserve it if present.
        if self.encoder.norm is not None:
            x = self.encoder.norm(
                x
            )

        # ============================================================
        # Same final normalization
        # ============================================================

        return self.final_norm(
            x
        )
class TransformerPatchEncoder(nn.Module):
    """
    Contextualizes all patch embeddings using self-attention.

    Input:
        x: [B, N, input_dim]

    Output:
        z: [B, N, model_dim]
    """

    def __init__(
        self,
        input_dim,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim={model_dim} must be divisible by "
                f"num_heads={num_heads}."
            )

        self.input_projection = (
            nn.Identity()
            if input_dim == model_dim
            else nn.Linear(input_dim, model_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x, padding_mask=None):
        """
        padding_mask:
            Optional Boolean tensor [B, N].
            True means the patch is padding and should be ignored.
        """
        x = self.input_projection(x)

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        return self.final_norm(x)

class TransformerActor(nn.Module):
    def __init__(
        self,
        conf,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        model_dim = conf.D_inner

        self.encoder = TransformerPatchEncoder(
            input_dim=conf.D_feat,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.action_head = nn.Linear(model_dim, 1)

    def forward(self, x, padding_mask=None):
        """
        Input:
            x: [B, N, D_feat]

        Output:
            logits: [B, 1, N]

        The output shape matches the original actor.
        """
        contextualized = self.encoder(
            x,
            padding_mask=padding_mask,
        )  # [B, N, D_inner]

        logits = self.action_head(
            contextualized
        ).squeeze(-1)  # [B, N]

        return logits.unsqueeze(1)  # [B, 1, N]

class TransformerCritic(nn.Module):
    def __init__(
        self,
        conf,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        model_dim = conf.D_inner

        self.input_projection = nn.Linear(
            conf.D_feat,
            model_dim,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, model_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(model_dim)

        self.value_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x, padding_mask=None):
        """
        Input:
            x: [B, N, D_feat]

        Output:
            value: [B, 1]
        """
        batch_size = x.shape[0]

        x = self.input_projection(x)  # [B, N, D_inner]

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )  # [B, 1, D_inner]

        x = torch.cat(
            [cls, x],
            dim=1,
        )  # [B, N+1, D_inner]

        if padding_mask is not None:
            cls_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=x.device,
            )

            padding_mask = torch.cat(
                [cls_mask, padding_mask],
                dim=1,
            )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        cls_representation = self.final_norm(
            x[:, 0]
        )  # [B, D_inner]

        value = self.value_head(
            cls_representation
        )  # [B, 1]

        return value

class Value_fc(nn.Module):
    def __init__(self, n_channels, droprate=0.0):
        super(Value_fc, self).__init__()
        self.fc = nn.Linear(n_channels, 1)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N
    

class Critic(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(Critic, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        if conf.dim_reduction:
            self.attention = Attention_Gated(conf.D_inner, D, 1)
            self.final_layer = Value_fc(conf.D_inner, droprate) 
        else:
            self.attention = Attention_Gated(conf.D_feat, D, 1)
            self.final_layer = Value_fc(conf.D_feat, droprate)
        self.use_dim_reduction = conf.dim_reduction


    def forward(self, x):
        state = x
        x = self.dimreduction(x) if self.use_dim_reduction else x
        A = self.attention(x).transpose(0,2).transpose(0,1)

        A = F.softmax(A, dim=-1) 
        h = torch.bmm(A, state)
        value = self.final_layer(h)
        return value
    

class Actor(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(Actor, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        if conf.dim_reduction:
            self.attention = Attention_Gated(conf.D_inner, D, 1)
        else:
            self.attention = Attention_Gated(conf.D_feat, D, 1)
        self.use_dim_reduction = conf.dim_reduction

    
    def forward(self, x):
        state = x
        x = self.dimreduction(x) if self.use_dim_reduction else x
        A = self.attention(x).transpose(0,2).transpose(0,1)

        # A = F.softmax(A, dim=-1) 
        return A
    

'''
Agnet: 
1. Actor - Input (N * d) state representation; Outputs a probability distribution over N
2. Critic - Input (N * d) state representation; Outputs the expected return from being in that state
'''

class Agent(nn.Module):
    def __init__(self, actor, critic, conf):
        super(Agent, self).__init__()
        self.conf = conf
        self.critic = critic
        self.actor = actor


    def get_value(self, x):
        value = self.critic(x)
        return value

    def get_action(self, x, visited_patch_ids, is_eval=False, is_top_k = False, is_top_p = False):
        A_raw = self.actor(x).view(1, -1)
        A_raw[0][visited_patch_ids] = -torch.inf
        dist = Categorical(logits=A_raw)


        # This code is added for random policy (Uncomment if need to execute this)
        # Sample Randomly from the patches --->
        # N = A_raw.shape[1]
        # all_indices = list(range(N))
        # remaining_indices = list(set(all_indices) - set(visited_patch_ids))
        # random_patch = random.choice(remaining_indices)
        # return torch.tensor(random_patch), None , None

        # Get action based on different sampling strategy ----
        # By default we will follow pick the max from probability distribution for SASHA inference

        if is_eval :
            # Now check which flag is up
            if is_top_k and is_top_p :
                raise ValueError("Only one of is_top_k or is_top_p can be True")

            elif is_top_k and not is_top_p :

                k = 3  # or 3 / 5 ----> determine top_k to be unmasked
                A_raw = A_raw.squeeze()
                topk_vals, topk_idxs = torch.topk(A_raw, k)
                probs = torch.zeros_like(A_raw)
                probs[topk_idxs] = torch.nn.functional.softmax(topk_vals, dim=0)
                dist = Categorical(probs=probs)
                action = dist.sample()

            elif not is_top_k and is_top_p:

                p = 0.3
                A_raw = A_raw.squeeze()
                sorted_logits, sorted_indices = torch.sort(A_raw, descending=True)
                sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=0)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                # Mask out all logits beyond the cumulative probability threshold
                cutoff_idx = torch.where(cumulative_probs > p)[0]
                if len(cutoff_idx) > 0:
                    last_idx = cutoff_idx[0].item() + 1
                else:
                    last_idx = len(A_raw)

                selected_indices = sorted_indices[:last_idx]
                probs = torch.zeros_like(A_raw)
                probs[selected_indices] = torch.nn.functional.softmax(A_raw[selected_indices], dim=0)
                dist = Categorical(probs=probs)
                action = dist.sample()

            else :
                action = torch.argmax(dist.probs)  # This is the by default choice for the inference mode

        else :
            action = dist.sample() # This is performed during training

        # Continue ....
        log_prob = dist.log_prob(action)

        return action, log_prob.detach(), dist.entropy()
    

    def get_action_and_value(self, x, visited_patch_ids):
        action = self.get_action(x, visited_patch_ids)
        value = self.get_value(x)
        return action, value


    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)

        Return:
            V - the predicted values of batch_obs
            log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		
        """

		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()
        A_raw = self.actor(batch_obs)
        dist = Categorical(logits=A_raw.squeeze(1))
        log_prob = dist.log_prob(batch_acts)
		
        return V, log_prob, dist.entropy()

