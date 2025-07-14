# TaxonomicalLoss
Source code and data for the paper: Taxonomical loss for weed seedlings image classification

Setup the environment using the *setup_env.sh* script.

Then download some data from the *data* folder.

![dataset overview](https://github.com/etiennelord/TaxonomicalLoss/blob/main/images/WPD.png)

## Usages

1. Create some dataset using the *setup_k_dataset.py* which split a dataset into the corresponding train/val/test split, but using *k* images for each class in the training set. Example for the Weed Phenological Dataset (WPD) for 100 images per class, and a maximum of 250 images in the validation and test sets:

```
python setup_k.py --data WPD_mixed_BBCH/ --output WPD_mixed_BBCH_100 --seed=42 --k=100 --max_test=250 --complete
```

2. Train the generic models for the baseline classification using the *generic_resnet50.py* or *generic_mobilenet3.py*:

```
python generic_mobilenet3.py --epoch=50 --data_dir WPD_mixed_BBCH_100 --output output_WPD_mobilenet --batch_size=128 --imgsz=512 --seed=42
```

3. Train the using the taxonomical loss with the mobilenet model:

```
python train.py --epoch=50 --data WPD_mixed_BBCH_100 --tree WPD_mixed_BBCH/taxonomy.txt --output output_WPD_mobilenet_taxonimic --batch_size=64 --imgsz=512 --lr=0.001 --model "mobilenet" --seed=42 --embeddings_size=64 --loss "taxonomic" 
```

Different options are available:

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--pretrained` | str | None | No | Path to a pretrained model - if wanted. |
| `--output_model` | str | "output.pth" | No | Specific filename of the output trained model. |
| `--output` | str | "./output" | No | Default output directory. |
| `--data` | str | "" | Yes | Path to the images data directory. |
| `--testdata` | str | "" | No | Path to the test images data directory if needed. |
| `--tree` | str | "" | Yes | Path to taxonomic tree in Nexus format. |
| `--model` | str | "resnet50" | No | Base network model used (can be resnet50, mobilenet, vit). |
| `--loss` | str | "taxonomic" | No | Loss criterion: tripletloss, taxonomic or HTL. |
| `--epochs` | int | 200 | No | Number of fine tuning training epochs. |
| `--lr` | float | 0.001 | No | Fine tuning learning rate. |
| `--batch_size` | int | 32 | No | Batch size parameter. |
| `--imgsz` | int | 224 | No | Image resize parameter. |
| `--embeddings_size` | int | 128 | No | Size of final vector embeddings. |
| `--margin` | float | 2.0 | No | Triplet margin parameters - default 2.0 |
| `--train_val_test_split` | float (list) | [0.70, 0.15, 0.15] | No | Dataset train, val, test split in percent e.g. 0.70 0.15 0.15 or 0.80 0.20* |
| `--seed` | int | 42 | No | Random seed. |
| `--beta` | float | 0.1 | No | Beta parameter for hierarchical triplet loss. |
| `--tree_depth` | int | 16 | No | Depth of the hierarchical tree. |
| `--tree_update_freq` | int | 5 | No | Frequency to update the hierarchical tree (epochs). |

*Use the *setup_k_dataset.py* is preferred.

## Simulations

The study simulation scripts are provided: 
- *run_PlantSeedlings.sh* for the PlantSeedlings dataset simulations.
-  *run_DeepWeeds.sh* for the DeepWeed dataset simulations.
-  *run_WPD.sh* for the Weed Phenological dataset simulations.

## Sample results using the EuroSat dataset

Some results using a subset of the EuroSat dataset [Helber, 2019]
![dataset overview](https://github.com/etiennelord/TaxonomicalLoss/blob/main/images/EuroSat1.png)

The taxonomy used
![dataset overview](https://github.com/etiennelord/TaxonomicalLoss/blob/main/images/EuroSat2.png)

Command-line used for training the model: 
```
python train.py --epoch=50 --data EuroSatRGBmini --tree taxonomy_Eurosat.txt --batch_size=128 --imgsz=64 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=32 --loss "taxonomic" --train_val_test_split 0.70 0.15 0.15
```

## References

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217-2226.
