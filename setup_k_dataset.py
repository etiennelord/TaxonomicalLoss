########################################################################################################################
### Divide Image Folder Utility
### Authors: Hans-Olivier Fontaine (UdeS/AAFC), Etienne Lord (AAFC, etienne.lord@agr.gc.ca)
### Copyright Agriculture and Agri-Food Canada
########################################################################################################################
import os
import glob
import shutil
import random
from pathlib import Path
from random import choice, shuffle
import argparse

def count_files(directory):
    return sum(1 for _ in Path(directory).rglob('*') if _.is_file())

def trainonly(source_dir,destination_dir,k):
    train_dir = destination_dir / 'train'

    # Create the necessary directories
    for dir_path in [train_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Process each class folder
    print(f'Processing {source_dir}\n')
    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            images = [file for file in Path(class_dir).rglob('*.*')
                                    if file.name.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
            random.shuffle(images)  # Shuffle the images
            print(f'{class_dir} {len(images)} images')
            # Create class subfolders in train, val, and test directories
            (train_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

            # Copy k images to the train folder
            for img in images[:k]:
                shutil.copy(img, train_dir / class_dir.name / img.name)

    print('Dataset split completed.')

def process(source_dir, destination_dir, k, max_test):
    train_dir = destination_dir / 'train'
    val_dir = destination_dir / 'val'
    test_dir = destination_dir / 'test'
    # Create the necessary directories
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    total_class=0
    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            total_class=total_class+1
        
    n_per_class=int(max_test/total_class)
    print(f"{total_class} {n_per_class}")
    # Process each class folder
    print(f'Processing {source_dir}\n')
    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            images = [file for file in Path(class_dir).rglob('*.*')
                                    if file.name.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
            random.shuffle(images)  # Shuffle the images
            print(f'{class_dir} {len(images)} images')
            # Create class subfolders in train, val, and test directories
            (train_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
            (val_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
            (test_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

            # Copy k images to the train folder
            for img in images[:k]:
                shutil.copy(img, train_dir / class_dir.name / img.name)

            # Split the remaining images between val and test folders
            remaining_images = images[k:]
            total_val_test = min(len(remaining_images), n_per_class * 2)  # Ensure we don't exceed available images
           
            split_index = total_val_test // 2
            print(f"{total_val_test} {split_index}")
            val_images = remaining_images[:split_index]
            test_images = remaining_images[split_index:total_val_test]

            for img in val_images:
                shutil.copy(img, val_dir / class_dir.name / img.name)

            for img in test_images:
                shutil.copy(img, test_dir / class_dir.name / img.name)

    for dir_name in ['train', 'val', 'test']:
        file_count = count_files(f"{destination_dir}/{dir_name}")
        print(f"Total number of files in {dir_name} directory: {file_count}")
    
    print('Dataset split completed.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating dataset')
    parser.add_argument('--output', type=str, required=True, default="",help='Path to the output directory')
    parser.add_argument('--data', type=str, required=True, default="", help='Path to the images data directory')
    parser.add_argument('--k', type=int, required=True, default="", help='k')
    parser.add_argument('--seed', type=int, default=42, help='k')
    parser.add_argument('--complete', default=True, action=argparse.BooleanOptionalAction, type=bool, help='Divise in order to have train,val,test directory')
    parser.add_argument('--max_test', type=int, default=1000, help='Maximum number of images in val and test per class sets combined')
    
    args = parser.parse_args()
    random.seed(args.seed)
    k=args.k
    source_dir=args.data
    destination_dir=args.output
    complete=args.complete
    max_test = args.max_test
    # Define paths
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    if complete:
        process(source_dir, destination_dir, k, max_test)
    else:
        trainonly(source_dir, destination_dir, k)

    
    
