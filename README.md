## Requirements

To install requirements:

```setup
conda create --name <env> --file conda-packages.txt
```

## Training

Data preparation:

change the ./dataset_cs/camelyon16.csv and ./dataset_cs/camelyon16/camelyon16.csv, then use: 
```
python split.py --csv_path ./camelyon16/camelyon16.csv    --out_dir ./camelyon16/splits/     --n_splits 5     --val_frac 0.2     --seed 42 
```
to create splits.


