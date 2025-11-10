**Dataset used in the study**

Each dataset contains the taxonomical tree used during the training steps

| Filename                    | Content                                                           | Download links (google drive )                                                     |
| --------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| WPD_mixed_BBCH.tar.gz       | The unnanotated Weed Phenological Dataset images used in simulations (4205 images)                  | https://drive.google.com/file/d/1uRUTQEh6vdSj1S3lJZWw__AMHtZgjG71/view?usp=sharing |
| WPD_separated_BBCH.tar.gz   | A selection of Weed Phenolofical Dataset images used in simulations (BBCH10-12, 2467 images)| https://drive.google.com/file/d/1_RvyQRIpwVOA1RpsQpU32iNrPAR2dBRM/view?usp=sharing |
| WPDv2.tar.gz | Fully annotated dataset (BBCH9-BBCH21+) | https://drive.google.com/file/d/1U4tUJaVFQSwQdsBilshsSvjxMkM59K8g/view?usp=sharing |
| DeepWeeds_256x256.tar.gz    | A selection of images from the DeepWeeds dataset [Olsen, 2019]                 | https://drive.google.com/file/d/1uI-q69fN3FkjiLZLP5H1gMrJd3Hs07QF/view?usp=sharing | 
| PlantSeedlings_64x64.tar.gz | A selection of images from the PlantSeedlings dataset [Giselsson, 2017]           | https://drive.google.com/file/d/1gpE9id6pr90wpu9cXqGAtugwNMri4Sal/view?usp=sharing |

Note: Images have been resized from their original size to either 512x512px (WPD), 256 x 256 px (DeepWeeds) and 64 x 64 px (PlantSeedlings).


```
python train.py --epoch=50 --data EuroSatRGBmini --tree taxonomy_Eurosat.txt --batch_size=128 --imgsz=64 --lr=0.001 --model "mobilenet" --seed=$iseed --embeddings_size=32 --loss "taxonomic" --train_val_test_split 0.70 0.15 0.15
```

**References**

Olsen, Alex, Dmitry A. Konovalov, Bronson Philippa, Peter Ridd, Jake C. Wood, Jamie Johns, Wesley Banks et al. "DeepWeeds: A multiclass weed species image dataset for deep learning." Scientific reports 9, no. 1 (2019): 2058.

Giselsson, T. M., Jørgensen, R. N., Jensen, P. K., Dyrmann, M., & Midtiby, H. S. (2017). A public image database for benchmark of plant seedling classification algorithms. arXiv preprint arXiv:1711.05458.

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217-2226.
