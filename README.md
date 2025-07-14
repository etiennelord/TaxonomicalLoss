# TaxonomicalLoss
Source code and data for the paper: Taxonomical loss for weed seedlings image classification

Setup the environment using the *setup_env.sh* script.

Then download some data from the *data* folder.

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
