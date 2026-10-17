import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models_features_extraction import get_encoder
from step1_create_patches import create_time_df


def compute_w_loader(loader, model, verbose=0, extract_high_res_features=True, device="cpu"):
    if verbose > 0:
        print(f"processing a total of {len(loader)} batches")

    hr_features_list, lr_features_list = [], []
    hr_coords_list, lr_coords_list = [], []
    hr_total_time_list, lr_total_time_list = [], []

    features_list, coords_list, total_time_list = [], [], []

    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            if extract_high_res_features:
                hr_img = data["hr_img"]
                hr_coords = data["hr_coords"]
                hr_time = data["hr_time"]

                lr_img = data["lr_img"]
                lr_coords = data["lr_coords"]
                lr_time = data["lr_time"]

                start_time_hr = time.time()
                mapping_factor = hr_img.shape[1]

                hr_batch = torch.reshape(
                    hr_img,
                    (-1, hr_img.shape[-3], hr_img.shape[-2], hr_img.shape[-1]),
                ).to(device, non_blocking=True)

                hr_features = model(hr_batch)
                hr_features = torch.reshape(
                    hr_features,
                    (-1, mapping_factor, hr_features.shape[-1]),
                ).cpu()

                hr_features_list.append(hr_features)
                hr_coords_list.append(hr_coords)

                total_time_hr = hr_time.sum().item() + time.time() - start_time_hr
                hr_total_time_list.append(total_time_hr)

                start_time_lr = time.time()

                lr_batch = torch.reshape(
                    lr_img,
                    (-1, lr_img.shape[-3], lr_img.shape[-2], lr_img.shape[-1]),
                ).to(device, non_blocking=True)

                lr_features = model(lr_batch).cpu()

                lr_features_list.append(lr_features)
                lr_coords_list.append(lr_coords)

                total_time_lr = lr_time.sum().item() + time.time() - start_time_lr
                lr_total_time_list.append(total_time_lr)

            else:
                start_time_feat = time.time()

                batch = data["img"]
                coords = data["coord"].numpy().astype(np.int32)
                time_1 = data["time"]

                batch = torch.reshape(
                    batch,
                    (-1, batch.shape[-3], batch.shape[-2], batch.shape[-1]),
                ).to(device, non_blocking=True)

                coords = np.reshape(coords, (-1, coords.shape[-1]))

                features = model(batch).cpu()

                features_list.append(features)
                coords_list.append(coords)

                total_time_feat = time_1.sum().item() + time.time() - start_time_feat
                total_time_list.append(total_time_feat)

    if extract_high_res_features:
        hr_features = torch.cat(hr_features_list, dim=0).numpy()
        lr_features = torch.cat(lr_features_list, dim=0).numpy()
        hr_coords = np.concatenate(hr_coords_list, axis=0)
        lr_coords = np.concatenate(lr_coords_list, axis=0)

        return (
            hr_features,
            hr_coords,
            sum(hr_total_time_list),
            lr_features,
            lr_coords,
            sum(lr_total_time_list),
        )

    features = torch.cat(features_list, dim=0).numpy()
    coords = np.concatenate(coords_list, axis=0)

    return features, coords, sum(total_time_list)


def update_arguments(args):
    if args.time_csv is None:
        args.time_csv = os.path.join("outputs/time", "time_feat.csv")

    if args.lr_time_col_name is None:
        args.lr_time_col_name = f"fe_b_{args.batch_size}_lr_level_{args.patch_level_low_res}"

    if args.hr_time_col_name is None:
        args.hr_time_col_name = f"fe_b_{args.batch_size}_hr_level_{args.patch_level_high_res}"

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_arguments():
    parser = argparse.ArgumentParser(description="Feature Extraction")

    parser.add_argument("--dataset_name", type=str, default=None, choices=["camelyon16", "tcga"])
    parser.add_argument("--data_h5_dir", type=str, default=None)
    parser.add_argument("--data_slide_dir", type=str, default=None)
    parser.add_argument("--slide_ext", type=str, default=".svs")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--feat_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--extract_high_res_features", type=str2bool, default=False)
    parser.add_argument("--patch_level_low_res", type=int, default=None)
    parser.add_argument("--patch_level_high_res", type=int, default=None)
    parser.add_argument("--index0", type=int, default=0)
    parser.add_argument("--indexn", type=int, default=1000)
    parser.add_argument("--time_csv", type=str, default=None)
    parser.add_argument("--lr_time_col_name", type=str, default=None)
    parser.add_argument("--hr_time_col_name", type=str, default=None)

    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-S/16",
        choices=["resnet50_trunc", "uni_v1", "conch_v1", "ViT-S/16", "Resnet50"],
    )
    parser.add_argument("--pretrain", type=str, default="medical_ssl", choices=["medical_ssl"])
    parser.add_argument("--no_auto_skip", default=True, action="store_true")
    parser.add_argument("--target_patch_size", type=int, default=224)

    return parser.parse_args()


def save_slide_h5_atomic(out_path, slide_id, feat, coords, label):
    tmp_path = out_path + ".tmp"

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with h5py.File(tmp_path, "w") as f:
        g = f.create_group(slide_id)
        g.create_dataset("feat", data=feat.astype(np.float16))
        g.create_dataset("coords", data=coords)
        g.attrs["label"] = label

    os.replace(tmp_path, out_path)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = get_arguments()
    args.device = device
    args = update_arguments(args)

    print("Initializing dataset")

    if args.csv_path is None:
        raise NotImplementedError("csv_path is required")

    bags_dataset = Dataset_All_Bags(args.csv_path, args.index0, args.indexn)
    df = bags_dataset.df.set_index("slide_id")
    print(f"Total slides: {len(bags_dataset)}")

    os.makedirs(args.feat_dir, exist_ok=True)

    lr_pt_dir = os.path.join(args.feat_dir, "lr", "pt_files")
    lr_h5_dir = os.path.join(args.feat_dir, "lr", "h5_files_per_slide")
    os.makedirs(lr_pt_dir, exist_ok=True)
    os.makedirs(lr_h5_dir, exist_ok=True)

    hr_pt_dir = os.path.join(args.feat_dir, "hr", "pt_files")
    hr_h5_dir = os.path.join(args.feat_dir, "hr", "h5_files_per_slide")
    os.makedirs(hr_pt_dir, exist_ok=True)
    os.makedirs(hr_h5_dir, exist_ok=True)

    model, img_transforms = get_encoder(args.model_name, pretrain=args.pretrain)
    model.eval()
    model = model.to(device)

    loader_kwargs = {"num_workers": 4, "pin_memory": True} if device.type == "cuda" else {}

    time_df = create_time_df(
        csv_file_path=args.time_csv,
        column_name_ls=["slide_name", args.lr_time_col_name, args.hr_time_col_name],
    )

    for bag_candidate_idx in tqdm(range(len(bags_dataset))):
        slide_id_with_ext = bags_dataset[bag_candidate_idx]
        slide_id = slide_id_with_ext.split(args.slide_ext)[0]

        bag_name = slide_id + ".h5"
        h5_file_path = os.path.join(args.data_h5_dir, "patches", bag_name)

        if not os.path.exists(h5_file_path):
            print(f"Skipping {slide_id}: patch h5 not found")
            continue

        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

        if not os.path.exists(slide_file_path):
            print(f"Skipping {slide_id}: slide file not found")
            continue

        lr_h5_path = os.path.join(lr_h5_dir, slide_id + ".h5")
        hr_h5_path = os.path.join(hr_h5_dir, slide_id + ".h5")

        lr_pt_path = os.path.join(lr_pt_dir, slide_id + ".pt")
        hr_pt_path = os.path.join(hr_pt_dir, slide_id + ".pt")

        if args.extract_high_res_features:
            if os.path.exists(lr_h5_path) and os.path.exists(hr_h5_path):
                print(f"Skipping {slide_id}: LR and HR per-slide h5 already exist")
                continue
        else:
            if os.path.exists(lr_h5_path):
                print(f"Skipping {slide_id}: LR per-slide h5 already exists")
                continue

        print(f"\nprogress: {bag_candidate_idx}/{len(bags_dataset)}")
        print(f"Slide name: {slide_id}")

        try:
            time_start = time.time()
            wsi = openslide.open_slide(slide_file_path)

            dataset = Whole_Slide_Bag_FP(
                file_path=h5_file_path,
                wsi=wsi,
                img_transforms=img_transforms,
                extract_high_res_features=args.extract_high_res_features,
                patch_level_low_res=args.patch_level_low_res,
                patch_level_high_res=args.patch_level_high_res,
                dataset_name=args.dataset_name,
            )

            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

            label = df.loc[slide_id_with_ext]["label"]

            if args.extract_high_res_features:
                (
                    hr_features,
                    hr_coords,
                    hr_total_time,
                    lr_features,
                    lr_coords,
                    lr_total_time,
                ) = compute_w_loader(
                    loader=loader,
                    model=model,
                    verbose=1,
                    device=device,
                    extract_high_res_features=True,
                )

                print(f"\ncomputing features for {slide_id} took {time.time() - time_start} s")

                if slide_id in time_df["slide_name"].values:
                    time_df.loc[time_df["slide_name"] == slide_id, args.lr_time_col_name] = lr_total_time
                    time_df.loc[time_df["slide_name"] == slide_id, args.hr_time_col_name] = hr_total_time
                else:
                    new_row = {col: "" for col in time_df.columns}
                    new_row["slide_name"] = slide_id
                    new_row[args.lr_time_col_name] = lr_total_time
                    new_row[args.hr_time_col_name] = hr_total_time
                    time_df = pd.concat([time_df, pd.DataFrame([new_row])], ignore_index=True)

                time_df.to_csv(args.time_csv, index=False)

                save_slide_h5_atomic(hr_h5_path, slide_id, hr_features, hr_coords, label)
                save_slide_h5_atomic(lr_h5_path, slide_id, lr_features, lr_coords, label)

                torch.save(torch.from_numpy(hr_features), hr_pt_path)
                torch.save(torch.from_numpy(lr_features), lr_pt_path)

            else:
                features, coords, total_time = compute_w_loader(
                    loader=loader,
                    model=model,
                    verbose=1,
                    device=device,
                    extract_high_res_features=False,
                )

                if slide_id in time_df["slide_name"].values:
                    time_df.loc[time_df["slide_name"] == slide_id, args.lr_time_col_name] = total_time
                else:
                    new_row = {col: "" for col in time_df.columns}
                    new_row["slide_name"] = slide_id
                    new_row[args.lr_time_col_name] = total_time
                    time_df = pd.concat([time_df, pd.DataFrame([new_row])], ignore_index=True)

                time_df.to_csv(args.time_csv, index=False)

                save_slide_h5_atomic(lr_h5_path, slide_id, features, coords, label)
                torch.save(torch.from_numpy(features), lr_pt_path)

            wsi.close()

        except Exception as e:
            print(f"[ERROR] failed on {slide_id}: {repr(e)}")

            for tmp_path in [lr_h5_path + ".tmp", hr_h5_path + ".tmp"]:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            try:
                wsi.close()
            except Exception:
                pass

            continue