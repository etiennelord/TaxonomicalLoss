########################################################################################################################
### Taxonomic Loss using Siamese Networks - 2024-2025
### Authors: Hans-Olivier Fontaine (UdeS/AAFC), Etienne Lord (AAFC, etienne.lord@agr.gc.ca)
### Copyright Agriculture and Agri-Food Canada
### Version 1.0
########################################################################################################################
import copy
import os
import warnings
import time
from datetime import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, \
    confusion_matrix, silhouette_score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from random import choice, shuffle
import argparse
from PIL import Image
from vit_pytorch import ViT
from Bio import Phylo
import networkx
import torch
from torch import Tensor, tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms import transforms
import torchvision.models as models
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

def get_tree_leaves(tree) -> dict:
    """
    Return the leaves of the taxonomic tree
    @param tree: a taxonomic tree from biopython
    @return: a dict of class names, clade
    """
    leaves = {}
    for clade in tree.find_clades(order="preorder"):
        if clade.is_terminal() and clade.name:
            leaves[clade.name]=clade
    return leaves


def create_matrix(tree) -> (np.ndarray, dict[str, dict[str, int]]):
    """
    Create the distance matrix used in the training step
    @param tree: a taxonimic tree from biopython
    @param metric: the distance metric used
    @return: a NxN matrix as a dictionary
    """
    leaves = get_tree_leaves(tree)
    leaves_names=list(leaves.keys())
    leaves_names.sort()
    total_leaves=len(leaves_names)	
    print(f'Total leaves:{total_leaves}\n')
    matrix = {key: {} for key in leaves_names}
    matrix_array = np.array([[0 for _ in leaves_names] for _ in leaves_names],dtype=float)    
    net = Phylo.to_networkx(tree)
    path=dict(networkx.all_pairs_dijkstra_path_length(net, weight='branch_length'))
    #exit(0)
    for i in range(0,total_leaves):
        for j in range(i+1,total_leaves):
            leave_a=leaves_names[i]
            leave_b=leaves_names[j]
            leaves_a_clade=leaves[leave_a]
            leaves_b_clade=leaves[leave_b]
            dist=path[leaves_a_clade][leaves_b_clade]
            matrix_array[i][j] = dist
            matrix_array[j][i] = dist
    max_value = np.max(matrix_array)
    matrix_array=matrix_array/max_value
    for i, leave_a in enumerate(leaves_names):
        for j, leave_b in enumerate(leaves_names):
            matrix[leave_a][leave_b]=matrix_array[i][j]
    matrix_array=pd.DataFrame(matrix_array, index=leaves_names)
    print(matrix_array)
    return matrix, matrix_array

def batch_mean_and_sd(loader: DataLoader):
    """
    Calculate from a batch of images, some statistics for image normalization
    @param loader: Dataloader for the image
    @return: two tensor representing the mean and std of the images (e.g. R,G,B)
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    m_tq = tqdm(loader, desc="Processing mean and std study", postfix={"mean": 0, "std": 0})

    for images in m_tq:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        m_tq.set_postfix({"mean": fst_moment, "std": f"{torch.sqrt(snd_moment - fst_moment ** 2)}"})

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean, std

def parse_directory(root_dir):
    data = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'filename': os.path.join(root_dir, class_dir, filename),
                        'label': class_name
                    })
    return pd.DataFrame(data)


class LabeledPathDataset(Dataset):
    def __init__(self, filename_or_directory="",datapath=None, m_transform=None, m_taxonomy=None):
        self.transforms = m_transform
        self.taxonomy = m_taxonomy
        #Parse data
        self.cls2idx = {}
        self.df = []
        self.img_cls_not_found = {}
        self.image_files = [file for file in self.data_root.rglob('*.*')
                            if file.name.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
        for i, image_path in enumerate(list(self.image_files)):
            img_cls = image_path.parent.name
            # Validate inputs

            if img_cls in self.taxonomy.keys():
                self.df.append((image_path, img_cls))
                if img_cls not in self.cls2idx.keys():
                    self.cls2idx[img_cls] = [i]

    def __len__(self):
        return len(self.df)

    def __open_and_transform(self, filename: str):
        try:
            image = Image.open(filename)
            image = self.transforms(image)
            return image
        except OSError as error:
            print(f"Error loading image at index {filename}: {error}")
            return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor_filename = row['filename']
        anchor_cls = row['label']
        anchor_cls_onehot = tensor(self.classes2onehot[anchor_cls])
        anchor_image = self.__open_and_transform(anchor_filename)
        if CUDA:
            return anchor_image.cuda(), anchor_cls_onehot.cuda(), anchor_filename, anchor_cls
        else:
            return anchor_image, anchor_cls_onehot, anchor_filename, anchor_cls


class FlatDataset(Dataset):
    """
    Class representing a dataset of images from a folder
    @param data_root Path : path to the images
    @param m_transform: transform function for the images (e.g. transforms.Compose)
    """
    def __init__(self, data_root: Path, m_transform=None):
        self.folder_path = data_root
        self.transform = m_transform
        self.image_files = [file for file in data_root.rglob('*.*')
                            if file.name.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
        #SAMPLE
        indice=np.random.choice(len(self.image_files),100, replace=False)
        self.image_files=np.array(self.image_files)[indice]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image


def image_folder_stats(data_path: Path, img_size=32) -> tuple[Tensor, Tensor]:
    """
    Function to calculate some statistics for image normalization
    @param data_path Path : path to the images
    @param img_size: parameter to resize the image for faster processing (e.g. 64)
    """
    batch_size = 32
    transform_img = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = FlatDataset(data_path, transform_img)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8)

    mean, std = batch_mean_and_sd(loader)

    print(f"ImageFolder {data_path}\n\tMEAN: {mean}\n\tSTD: {std}")
    return mean, std


class ConvBlock(nn.Module):
    """
    Custom convolutional block for the siamese network
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionBlock(nn.Module):
    """
    Custom Attention Block for the siamese network
    """
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class SiameseNetwork(nn.Module):
    """
    Main siamese network class and architecture
    """
    def __init__(self, model: str = "resnet50", embeddings_size: int = 128, image_size: int=224):
        """

        @type image_size: object
        """
        super(SiameseNetwork, self).__init__()
        self.model_version = model
        self.image_size=image_size
        if model == "resnet50":
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            num_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_features, embeddings_size)
            torch.nn.init.xavier_uniform_(self.model.fc.weight)
            torch.nn.init.zeros_(self.model.fc.bias)
        elif model == "mobilenet":
            self.model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            num_features = self.model.classifier[-1].in_features
            custom_classifier = torch.nn.Linear(num_features, embeddings_size)
            self.model.classifier[-1] = custom_classifier
            torch.nn.init.xavier_uniform_(custom_classifier.weight)
            torch.nn.init.zeros_(custom_classifier.bias)
        elif model == "vit":
            self.model = ViT(
                image_size=image_size,
                patch_size=16,
                num_classes=1000,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072,
                dropout=0.1,
                emb_dropout=0.1
            )
            custom_classifier = torch.nn.Linear(768, embeddings_size)
            self.model.head = custom_classifier
        else:
            raise NotImplementedError(f"Model not implemented: {model}")

    def forward_one(self, x):
        if self.model_version == "base":
            x = self.model(x)
            attention_weights = self.attention(x)
            x = x * attention_weights
            x = torch.mean(x, dim=(2, 3))  # Global average pooling
            x = self.fc(x)
            return x
        return self.model(x)

    def forward(self, m_anchor, m_positive, m_negative):
        m_anchor_output = self.forward_one(m_anchor)
        m_positive_output = self.forward_one(m_positive)
        m_negative_output = self.forward_one(m_negative)
        return m_anchor_output, m_positive_output, m_negative_output


def create_one_hot_encoding_mapping(dictionary):
    """
    Create the one_hot_encoding of class
    @param dictionary: dict of key (classname): value
    @return: a dict of key -> one hot vector
    """
    keys = list(dictionary.keys())
    keys.sort()
    num_classes = len(keys)
    mapping = {}
    for i, key in enumerate(keys):
        encoding = np.zeros(num_classes)
        encoding[i] = 1
        mapping[key] = encoding
    return mapping

########################################################################################################################
### Triplet or Taxonomical Image Data Loader
########################################################################################################################
class TripletImageDataset(Dataset):
    """
    Main Triplet dataset class
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_imgs = {cls: [] for cls in self.classes}

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):  # Only add files, not subdirectories
                    self.class_to_imgs[class_name].append(file_path)

        self.triplets = self._generate_triplets()

    def _generate_triplets(self):
        triplets = []
        for anchor_class in self.classes:
            for anchor_img in self.class_to_imgs[anchor_class]:
                positive_img = random.choice([img for img in self.class_to_imgs[anchor_class] if img != anchor_img])
                negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])
                negative_img = random.choice(self.class_to_imgs[negative_class])
                triplets.append((anchor_img, positive_img, negative_img))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

class TaxonomicTripletImageDataset(Dataset):
    """
    Main Taxonomic dataset class
    """
    def __init__(self, image_path: Path, tree_path= "", data_transforms=None, test_transforms=None, train_val_test_split: tuple = [0.70,0.15, 0.15]):
        # dataset
        self.emb_flag=False
        self.train_df, self.val_df, self.test_df = None, None, None
        # image buffer
        self.image_buffer={}
        if tree_path.endswith("newick"):
            try :
                print("Trying reading tree data in newick format.")
                self.tree = Phylo.read(tree_path, "newick")
                for clade in self.tree.find_clades():
                    if clade.branch_length is None:
                        clade.branch_length = 1.0
                    if clade.branch_length == 0.0:
                        clade.branch_length = 1.0
                print(self.tree)
                newick = True
            except:
                newick=False
        else:
            try :
                print("Trying reading tree data in nexus format.")
                self.tree = Phylo.read(tree_path, "nexus")
                for clade in self.tree.find_clades():
                    if clade.branch_length is None:
                        clade.branch_length = 1.0
                    if clade.branch_length == 0.0:
                        clade.branch_length = 1.0
                print(self.tree)
            except:
                print(f'Unable to parse the data in the tree file {tree_path}.\n* Ensure the taxonomic data is in nexus format. *\n')
                exit(-1)
        self.taxonomy, self.taxonomy_matrix = create_matrix(self.tree)
        self.data_root = Path(image_path)
        self.data_transforms = data_transforms
        self.test_transforms = test_transforms
        self.transforms=self.data_transforms
        self.train_val_test_split = train_val_test_split
        self.test_dataset_flag=False
        self.cls2idx = {}
        self.df = []
        self.img_cls_not_found = {}
        self.image_files = [file for file in self.data_root.rglob('*.*')
                            if file.name.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
        for i, image_path in enumerate(list(self.image_files)):
            img_cls = image_path.parent.name
            # Validate inputs

            if img_cls in self.taxonomy.keys():
                self.df.append((image_path, img_cls))
                if img_cls not in self.cls2idx.keys():
                    self.cls2idx[img_cls] = [i]
                else:
                    self.cls2idx[img_cls].append(i)
            else:
                print(image_path)
                self.image_files.remove(image_path)
                self.img_cls_not_found[img_cls]=1
        self.classes2onehot = create_one_hot_encoding_mapping(self.cls2idx)
        self.classes = list(self.cls2idx.keys())

        for nf in self.img_cls_not_found.keys():
            print(f'Image class not found in taxonomy {nf}.')

        # Randomly split the dataset into the train, val, test
        self.__split()

        # Precompute positive and negative dict index
        self.positive_class={cls: [] for cls in self.classes}
        self.negative_class = {}
        self.neighbor_class={cls: [] for cls in self.classes}
        for anchor_cls in self.classes:
            #max_dist = max(self.taxonomy[anchor_cls].values())
            positive=list(self.taxonomy[anchor_cls].values())
            nearest_distance = min ((x for x in positive if x > 0))            
            #nearest_distance=np.min(positive)
            max_dist=0.0001
            
            for k_pos,v_pos in self.taxonomy[anchor_cls].items():
                if v_pos == nearest_distance:
                    self.neighbor_class[anchor_cls].append(k_pos)
                    for k_neg, v_neg in self.taxonomy[anchor_cls].items():
                     if (anchor_cls,k_pos) not in self.negative_class.keys():
                                self.negative_class[(anchor_cls,k_pos)]=[k_neg]
                if v_pos < max_dist and k_pos in self.cls2idx:
                    self.positive_class[anchor_cls].append(k_pos)
                    # Negative
                    for k_neg, v_neg in self.taxonomy[anchor_cls].items():
                        if v_pos<v_neg and k_neg in self.cls2idx:                            
                            if (anchor_cls,k_pos) not in self.negative_class.keys():
                                self.negative_class[(anchor_cls,k_pos)]=[k_neg]
                            else:
                                self.negative_class[(anchor_cls,k_pos)].append(k_neg)
        #print(self.positive_class)
        #print(self.negative_class)
        #exit(0)

    def __len__(self):
        return len(self.df)

    def __split(self):
        self.train_df = []
        self.val_df = []
        self.test_df = []
        self.train_cls2idx = {cls: [] for cls in self.classes}
        self.val_cls2idx = {cls: [] for cls in self.classes}
        self.test_cls2idx = {cls: [] for cls in self.classes}
        # If found in self.data_root some csv
        file_found=False
        if os.path.exists(os.path.join(self.data_root,"train")):
            file_found=True
            train_files = [file for file in Path(os.path.join(self.data_root,"train")).rglob('*.*')
                                if file.name.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]

            for i, image_path in enumerate(list(train_files)):
                img_cls = image_path.parent.name
                self.train_df.append((image_path, img_cls))
                self.train_cls2idx[img_cls].append(len(self.train_df) - 1)
        if os.path.exists(os.path.join(self.data_root,"val")):
            file_found=True
            val_files = [file for file in Path(os.path.join(self.data_root,"val")).rglob('*.*')
                                if file.name.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
            for i, image_path in enumerate(list(val_files)):
                img_cls = image_path.parent.name
                self.val_df.append((image_path, img_cls))
                self.val_cls2idx[img_cls].append(len(self.val_df) - 1)
        if os.path.exists(os.path.join(self.data_root,"test")):
            file_found=True
            test_files = [file for file in Path(os.path.join(self.data_root,"test")).rglob('*.*')
                                if file.name.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp'))]
            for i, image_path in enumerate(list(test_files)):
                img_cls = image_path.parent.name
                self.test_df.append((image_path, img_cls))
                self.test_cls2idx[img_cls].append(len(self.test_df) - 1)

        if not file_found:
            for cls in self.classes:
                indexes = self.cls2idx[cls]
                shuffle(indexes)
                n_idx = len(indexes)
                for i, index in enumerate(indexes):
                    img_file = self.df[index]
                    if i / n_idx < self.train_val_test_split[0]:
                        self.train_df.append(img_file)
                        self.train_cls2idx[cls].append(len(self.train_df) - 1)
                    elif i / n_idx < self.train_val_test_split[0] + self.train_val_test_split[1]:
                        self.val_df.append(img_file)
                        self.val_cls2idx[cls].append(len(self.val_df) - 1)
                    else:
                        self.test_df.append(img_file)
                        self.test_cls2idx[cls].append(len(self.test_df) - 1)
        print(f'Total images for Training: {len(self.train_df)} Validation: {len(self.val_df)} Test: {len(self.test_df)}')

    def train(self):
        assert self.train_df is not None
        tmp = copy.deepcopy(self)
        tmp.df = tmp.train_df
        tmp.cls2idx = tmp.train_cls2idx
        return tmp

    def emb(self):
        tmp = copy.deepcopy(self)
        tmp.df = tmp.train_df
        tmp.cls2idx = tmp.train_cls2idx
        tmp.emb_flag=True
        tmp.transforms = self.test_transforms
        return tmp

    def val(self):
        assert self.val_df is not None
        tmp = copy.deepcopy(self)
        tmp.df = tmp.val_df
        tmp.cls2idx = tmp.val_cls2idx
        return tmp

    def test(self):
        tmp = copy.deepcopy(self)
        tmp.df = self.test_df
        tmp.cls2idx = self.test_cls2idx
        # replace some functions
        tmp.__getitem__=self.getitem_test
        tmp.transforms=self.test_transforms
        tmp.emb_flag=True
        return tmp

    def __open_and_transform(self, file_path: Path):
        try:
            if file_path.__str__() in self.image_buffer:
                image = self.image_buffer[file_path.__str__()]
            else:
                image = Image.open(file_path.__str__())
                image = self.transforms(image)
                self.image_buffer[file_path.__str__()] = image
            #image = self.transforms(image)
            return image
        except OSError as error:
            print(f"Error loading image at index {file_path.__str__()}: {error}")
            return None

    def getitem_test(self, idx):
        anchor_filename, anchor_cls = self.df[idx]
        #anchor_cls_onehot = tensor(self.classes2onehot[anchor_cls])
        anchor_image = self.__open_and_transform(anchor_filename)        
        return anchor_image, anchor_cls

    def __getitem__(self, idx):
        if self.emb_flag:
            return self.getitem_test(idx)
        anchor_filename, anchor_cls = self.df[idx]
        positive_idx, positive_dist, pos_cls = self._get_random_positive_index(idx)
        positive_filename, positive_cls = self.df[positive_idx]
        negative_idx, negative_dist, neg_cls = self._get_random_negative_index(anchor_cls, pos_cls)
        anchor_cls = tensor(self.classes2onehot[anchor_cls])
        anchor_image = self.__open_and_transform(anchor_filename)
        positive_cls = tensor(self.classes2onehot[positive_cls])
        positive_image = self.__open_and_transform(positive_filename)
        negative_filename, negative_cls = self.df[negative_idx]
        negative_cls = tensor(self.classes2onehot[negative_cls])
        negative_image = self.__open_and_transform(negative_filename)
        return anchor_image, positive_image, negative_image, anchor_cls, positive_cls, negative_cls, positive_dist, negative_dist

    def _get_random_negative_index(self, anchor_cls, positive_cls):
        available_class = self.negative_class[(anchor_cls,positive_cls)]
        neg_cls = choice(available_class)
        neg_idx = choice(self.cls2idx[neg_cls])
        return neg_idx, self.taxonomy[anchor_cls][neg_cls], neg_cls

    def _get_random_positive_index(self, anchor_idx):
        anchor_cls = self.df[anchor_idx][1]
        available_class = self.positive_class[anchor_cls]
        prob=random.random()
        if prob<0.05:
            available_class=self.neighbor_class[anchor_cls]
        i=0
        while i<100:  #Just for edge cases - unlikely
            i += 1
            pos_cls = choice(available_class)
            pos_idx = choice(self.cls2idx[pos_cls])
            if pos_idx != anchor_idx:
                return pos_idx, self.taxonomy[anchor_cls][pos_cls], pos_cls
        pos_cls = choice(self.classes)
        pos_idx = choice(self.cls2idx[pos_cls])
        return pos_idx, self.taxonomy[anchor_cls][pos_cls]

########################################################################################################################
### Loss Functions
########################################################################################################################

class HierarchicalTripletLoss(nn.Module):
    """
    Custom Implementation of Hierarchical Triplet Loss with dynamic margin calculation based on class taxonomy.
    """
    def __init__(self, beta=0.1, tree=None, dataloader=None, num_classes=None, num_levels=16):
        super(HierarchicalTripletLoss, self).__init__()
        self.beta = beta
        self.hierarchical_tree = None
        self.class_distances = None
        self.class_centers = {}
        self.distance_matrix = None
        self.tree = tree
        
        # Initialize tree if provided
        if tree is not None and dataloader is not None and num_classes is not None:
            self.build_hierarchical_tree(tree, dataloader, num_classes, num_levels)
    
    def build_hierarchical_tree(self, model, dataloader, num_classes, num_levels=16):
        """
        Build hierarchical tree based on class distances in the embedding space
        """
        print("Building hierarchical tree...")
        # Set model to evaluation mode
        model.eval()
        
        # Compute class centers
        class_features = {}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing class features"):
                images, labels = batch[0], batch[3] #batch 3 = anchor_cls
                images = images.to(DEVICE)
                features = model.forward_one(images)
                
                for i, label in enumerate(labels):                    
                    label_str = str(label)
                    if label_str not in class_features:
                        class_features[label_str] = []
                    class_features[label_str].append(features[i].cpu())
        
        # Calculate class centers
        for label in class_features:
            self.class_centers[label] = torch.stack(class_features[label]).mean(0)
        
        # Compute distance matrix between class centers
        distance_matrix = torch.zeros((num_classes, num_classes))
        
        class_list=[str(key) for key in self.class_centers.keys()]
        class_list = sorted(class_list)
        for i, class1 in enumerate(class_list):
            for j, class2 in enumerate(class_list):
                if i != j:
                    distance_matrix[i, j] = torch.norm(self.class_centers[class1] - self.class_centers[class2])
        
        # Calculate average inner-class distance
        inner_distances = []
        for label, features in class_features.items():
            if len(features) > 1:
                pairs = [(i, j) for i in range(len(features)) for j in range(i+1, len(features))]
                for i, j in pairs:
                    inner_distances.append(torch.norm(features[i] - features[j]).item())
        
        d0 = np.mean(inner_distances) if inner_distances else 0.1
        
        # Build hierarchical tree
        self.class_distances = distance_matrix
        self.hierarchical_tree = {}
        
        # Create levels of hierarchy
        for l in range(num_levels):
            dl = l * (4 - d0) / num_levels + d0
            self.hierarchical_tree[l] = self._merge_classes(distance_matrix, dl, class_list)
        
        print(f"Hierarchical tree built with {num_levels} levels")
        model.train() #Ensure training mode after rebuilt        
        return self.hierarchical_tree
    
    def _merge_classes(self, distance_matrix, threshold, class_list):
        """Merge classes that are closer than threshold"""
        num_classes = distance_matrix.shape[0]
        merged_classes = {}
        
        # Initialize each class as its own cluster
        for i in range(num_classes):
            merged_classes[i] = [class_list[i]]
        
        # Merge classes
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                if distance_matrix[i, j] < threshold:
                    # Find the clusters containing i and j
                    cluster_i = None
                    cluster_j = None
                    for cluster_id, cluster in merged_classes.items():
                        if class_list[i] in cluster:
                            cluster_i = cluster_id
                        if class_list[j] in cluster:
                            cluster_j = cluster_id
                    
                    # Merge the clusters if they're different
                    if cluster_i is not None and cluster_j is not None and cluster_i != cluster_j:
                        merged_classes[cluster_i].extend(merged_classes[cluster_j])
                        del merged_classes[cluster_j]
        
        return merged_classes
    
    def compute_dynamic_margin(self, anchor_class, neg_class, avg_class_distance):
        """Compute dynamic margin based on hierarchical tree"""
        # Find the level where anchor_class and neg_class are merged
        merge_level = None
        for level in sorted(self.hierarchical_tree.keys()):
            for cluster in self.hierarchical_tree[level].values():
                if anchor_class in cluster and neg_class in cluster:
                    merge_level = level
                    break
            if merge_level is not None:
                break
        
        if merge_level is None:
            merge_level = max(self.hierarchical_tree.keys())
        
        # Compute the threshold at this level
        d0 = avg_class_distance
        dH = merge_level * (4 - d0) / len(self.hierarchical_tree) + d0
        
        # Compute the margin
        margin = self.beta + dH - avg_class_distance
        return margin
    
    def forward(self, anchors, positives, negatives, anchor_labels=None, neg_labels=None):
        """Compute the hierarchical triplet loss"""
        # Normalize embeddings to unit length
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)
        
        # Compute distances
        pos_distances = (anchors - positives).pow(2).sum(dim=1)
        neg_distances = (anchors - negatives).pow(2).sum(dim=1)
        
        # Use dynamic margin if tree is available, otherwise use fixed margin
        if self.hierarchical_tree is not None and anchor_labels is not None and neg_labels is not None:
            batch_size = anchors.size(0)
            margins = torch.zeros(batch_size).to(anchors.device)
            
            # For each sample, compute dynamic margin
            class_distances = {}
            for i in range(batch_size):
                anchor_label = anchor_labels[i]
                neg_label = neg_labels[i]
                
                # Compute inner-class distance as a proxy for s_ya
                s_ya = 0.1  # Default value
                if anchor_label in self.class_centers:
                    s_ya = class_distances.get(anchor_label, s_ya)
                
                margins[i] = self.compute_dynamic_margin(anchor_label, neg_label, s_ya)
        else:
            # Default to a fixed margin
            margins = torch.ones_like(pos_distances) * 0.2
        
        # Compute losses with dynamic margins
        losses = F.relu(pos_distances - neg_distances + margins)
        return losses.mean()


class TaxonomicLoss(torch.nn.Module):
    """
    Main Taxonomic loss definition
    """
    def __init__(self):
        super(TaxonomicLoss, self).__init__()

    def forward(self, anchors, positives, negatives, positive_dist, negative_dist):
        distances_positive = (anchors - positives).pow(2).sum(-1)
        distances_negative = (anchors - negatives).pow(2).sum(-1)
        distances_total=(positive_dist - negative_dist).pow(2).sum(-1)        
        max_distance = torch.max(distances_total)
        min_distance = torch.min(distances_total)
        epsilon = 1e-6  # Small value to prevent division by zero
        distances_total_bounded = ((distances_total - min_distance) / (max_distance - min_distance + epsilon))
        
        losses =  nn.functional.relu(distances_positive-distances_negative+distances_total_bounded)
        return losses.mean()

########################################################################################################################
### EVALUATION AND OUTPUT FUNCTIONS
########################################################################################################################

def evaluate(train_dataset_embedding, train_dataset_class, test_dataset_embedding,test_dataset_class):
    """
    Evaluate the test embedding to the train data embedding
    @param train_dataset_embedding: the trained embedding for the network
    @param train_dataset_class: the associated labels
    @param test_dataset_embedding: the test data embedding
    @param test_dataset_class: the test label
    @return: accuracy, precision, recall score, classification report, confusion matrix
    """
    classes=train_dataset_class+test_dataset_class
    cls2idx={}
    for cls in (set(sorted(classes))):
        cls2idx[cls]=len(cls2idx.keys())

    train_class = [cls2idx.get(x, x) for x in train_dataset_class]
    test_class = [cls2idx.get(x, x) for x in test_dataset_class]

    results = {}
    y_true = test_class
    y_pred = []

    for i,sample in enumerate(test_dataset_embedding):
        dist=torch.norm(torch.from_numpy(train_dataset_embedding)-torch.from_numpy(sample), dim=1)
        nearest=torch.argmin(dist)
        y_pred.append(train_class[nearest])

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred,  average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    results= {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'F1':f1}
    print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f},')
    # Classification report
    classification_rep = classification_report(y_true, y_pred)
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    return results, cls2idx, classification_rep, conf_matrix

def label_to_color_dict(labels):
    """
    Map some specific color to the labels for the reproductibility
    @param labels: Sorted, Unique labels
    @return: a dict linking labels -> colors
    """
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, len(labels)))
    color_dict = dict(zip(labels, colors))
    return color_dict

def visualize_embeddings(embeddings, labels, title, filename):
    """
    Create a t-SNE visualization of an embedding, saving to filename
    Will also output a silhouette score, based on the t-SNE
    @param embeddings: datasets embedding
    @param labels: associated label to each embedding
    @param title: figure title
    @param filename: output filename
    @return: silhouette score
    """
    # Reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=2, random_state=421)
    embeddings_2d = tsne.fit_transform(embeddings)
    # Cast
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    unique_labels=np.sort(unique_labels)
    # Create figure
    plt.figure(figsize=(12, 10))

    # Create scatter plot
    random.seed(421)
    colors = label_to_color_dict(unique_labels)

    for label in sorted(unique_labels):
        label_indices = (labels == label)
        plt.scatter(embeddings_2d[label_indices, 0],
                    embeddings_2d[label_indices, 1],
                    color=colors.get(label),
                    label=label,
                    alpha=0.7)

    # Customize plot
    plt.title(title, fontsize=20)
    # Add legend
    if len(unique_labels)<25:
        plt.legend(title="Classes", title_fontsize='13', fontsize='13', loc='center left',
               bbox_to_anchor=(1, 0.5), ncol=1)
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{filename}", dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {filename}")

    # Calculate silhouette score
    silhouette_avg = silhouette_score(embeddings_2d, labels)
    return silhouette_avg

########################################################################################################################
### MAIN FUNCTION
########################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for pretrained learning triplet experiment.')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to a pretrained model - if wanted.' )
    parser.add_argument('--output_model', type=str, default="output.pth", help='Specific filename of the output trained model.')
    parser.add_argument('--output', type=str, default="./output",help='Default output directory.')
    parser.add_argument('--data', type=str, required=True, default="", help='Path to the images data directory.')
    parser.add_argument('--testdata', type=str, required=False, default="", help='Path to the test images data directory if needed.')
    parser.add_argument('--tree', type=str, required=True, default="", help='Path to taxonomic tree in Nexus format.')
    parser.add_argument('--model', type=str, default="resnet50", help='Base network model used (can be resnet50, mobilenet, vit).')
    parser.add_argument('--loss', type=str, default="taxonomic",help='Loss criterion: tripletloss, taxonomic.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of fine tuning training epochs.')
    parser.add_argument('--lr', default=0.001, type=float, help='Fine tuning learning rate.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size parameter.')
    parser.add_argument('--imgsz', default=224, type=int, help='Image resize parameter.')
    parser.add_argument('--embeddings_size', default=128, type=int, help='Size of embeddings.')
    parser.add_argument('--margin', default=2.0, type=float, help='Triplet margin parameters - default 2.0')
    parser.add_argument('--train_val_test_split', default=[0.70,0.15,0.15], nargs="+", type=float, help='Dataset train, val, test split in percent e.g. 0.70 0.15 0.15 or 0.80 0.20')
    parser.add_argument('--seed', default=5, type=int, help='Random seed.')
    parser.add_argument('--beta', default=0.1, type=float, help='Beta parameter for hierarchical triplet loss.')
    parser.add_argument('--tree_depth', default=16, type=int, help='Depth of the hierarchical tree.')
    parser.add_argument('--tree_update_freq', default=5, type=int, help='Frequency to update the hierarchical tree (epochs).')
    
    args = parser.parse_args()
    # Set random seeds
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ## LOAD ARGUMENTS INTO VARIABLES ##
    DATA_DIR = Path(args.data)
    TREE_FILE=args.tree
    PRETRAINED = args.pretrained
    OUTPUT_MODEL = args.output_model
    OUTPUT_DIRECTORY=args.output
    MODEL = args.model
    LOSS = args.loss
    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size
    IMGSZ = args.imgsz
    EMBEDDINGS_SIZE = args.embeddings_size
    MARGIN = args.margin
    TRAIN_VAL_TEST_SPLIT = tuple(args.train_val_test_split)
    TESTDATA = args.testdata
    ## CREATE -OUTPUT- DIRECTORY
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    ## VALIDATE SOME INPUTS ##
    if not os.path.isdir(OUTPUT_DIRECTORY):
        print(f'The output directory {OUTPUT_DIRECTORY} is not a directory or could not be created!')
        exit(-1)
    if not os.path.isfile(TREE_FILE):
        print('The tree file {} does not exist!'.format(TREE_FILE))
        exit(-1)
    if not os.path.isdir(DATA_DIR):
        print('The image data directory {} does not exist!'.format(DATA_DIR))
        exit(-1)
    if PRETRAINED is not None and not os.path.isfile(PRETRAINED):
        print('A pretrained model was supplied, but the file {} does not exist!'.format(PRETRAINED))
        exit(-1)

    ## PREPROCESS SOME IMAGE TO ESTIMATE PARAMETERS ##
    print(f"Reading image data for normalization.")
    means, stds = [[0.485, 0.456, 0.406],[0.229,0.224,0.225]]
    #image_folder_stats(DATA_DIR, IMGSZ)
    #pd.DataFrame([means.numpy(), stds.numpy()], index=["means", "stds"]).to_csv("output/norm.csv")
    siamese_net = SiameseNetwork(MODEL, EMBEDDINGS_SIZE, IMGSZ).to(DEVICE)
    if PRETRAINED == None:
        print(f"Using default model {MODEL}.")
    else:
        print(f"Using pretrained weights from {PRETRAINED} for {MODEL}")
        if Path(PRETRAINED).exists():
            pretrained_weights = torch.load(PRETRAINED)
            siamese_net.load_state_dict(pretrained_weights)

    ########################################################################
    #   MAIN TRAINING LOOP
    ########################################################################
    #criterion = nn.TripletMarginLoss(margin=MARGIN)
    if LOSS == 'taxonomic':
        criterion = TaxonomicLoss()
    elif LOSS == 'triplet':
        criterion = nn.TripletMarginLoss(margin=MARGIN)
    elif LOSS == 'htl':
        criterion = HierarchicalTripletLoss(beta=args.beta)    
    else: 
        print(f"Loss model not found {LOSS}.")
        exit(0)


    optimizer = optim.Adam(siamese_net.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.1, patience=20, threshold=0.01)

    # Fine-tuning
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMGSZ, IMGSZ), antialias=False),
        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(mean=means, std=stds),
    ])
    pure_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMGSZ, IMGSZ), antialias=False),
        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(mean=means, std=stds),
    ])
    dataset = TaxonomicTripletImageDataset(image_path=DATA_DIR,tree_path=TREE_FILE,data_transforms=transform, test_transforms=pure_transform, train_val_test_split=TRAIN_VAL_TEST_SPLIT)
    NEIGHBORS = len(dataset.classes)
    
        
    train_dataloader = DataLoader(dataset.train(), batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset.val(), batch_size=BATCH_SIZE, shuffle=True)

    #Build HTL tree
    if LOSS == 'htl':
        criterion.build_hierarchical_tree(siamese_net, train_dataloader, len(dataset.classes), args.tree_depth)
    
    print(f'Starting training for {EPOCHS} epochs.\n')

    # Get the current time and epoch and setup tmp variables
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_time = int(time.time())    
    update_epoch=EPOCHS
    dataset.level=100
    history_train_loss=[]
    history_val_loss=[]
    history_train_acc=[]
    history_val_acc=[]
    history_LR=[]
    best_loss = 9999

    for epoch in range(EPOCHS):
        siamese_net.train()
        train_losses = []

        tq = tqdm(train_dataloader, desc=f"Train ({epoch + 1}/{EPOCHS})",
                  postfix={"loss": f"{round(0.0, 4)}±{round(0.0, 4)}"})
        for batch in tq:

            anchor, positive, negative, _, _, _, positive_dist, negative_dist = batch
            anchor, positive, negative, positive_dist, negative_dist = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE), positive_dist.to(DEVICE), negative_dist.to(DEVICE)

            optimizer.zero_grad()

            anchor_output, positive_output, negative_output = siamese_net(anchor, positive, negative)
            if LOSS == 'taxonomic':
                loss = criterion(anchor_output, positive_output, negative_output, positive_dist, negative_dist)
            elif LOSS == 'htl':
                 loss = criterion(anchor_output, positive_output, negative_output, batch[3], batch[5]) 
            else:
                loss = criterion(anchor_output, positive_output, negative_output)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            tq.set_postfix({"loss": f"{round(np.average(train_losses), 4)}±"
                                    f"{round(float(np.std(train_losses)), 4)}"})

            avg_loss = np.average(train_losses) / len(train_dataloader)
        scheduler.step(avg_loss)
        history_train_loss.append(avg_loss)
        print(f'\nTrain Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()}')
        history_LR.append(scheduler.get_last_lr()[0])
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_weights = copy.deepcopy(siamese_net.state_dict())  # Deep copy here
        
        # VALIDATION
        siamese_net.eval()
        with torch.no_grad():
            val_losses = []
            tqv = tqdm(val_dataloader, desc=f"Val ({epoch + 1}/{EPOCHS})",
                       postfix={"loss": f"{round(0.0, 4)}±{round(0.0, 4)}"})
            for batch in tqv:
                anchor, positive, negative, _, _, _, positive_dist, negative_dist = batch
                anchor, positive, negative, positive_dist, negative_dist = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE), positive_dist.to(DEVICE), negative_dist.to(DEVICE)
                anchor_output, positive_output, negative_output = siamese_net(anchor, positive, negative)
                if LOSS == 'taxonomic':
                    loss = criterion(anchor_output, positive_output, negative_output, positive_dist, negative_dist)
                else:
                    loss = criterion(anchor_output, positive_output, negative_output)

                val_losses.append(loss.item())
                tq.set_postfix({"loss": f"{round(np.average(val_losses), 4)}±"
                                        f"{round(float(np.std(val_losses)), 4)}"})

            avg_loss = np.average(val_losses) / len(val_dataloader)
            history_val_loss.append(avg_loss)
            print(f'\nVal Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}')
        
        # OUTPUT AND HTL UPDATE
        if LOSS == 'htl' and (epoch + 1) % args.tree_update_freq == 0:
            criterion.build_hierarchical_tree(siamese_net, train_dataloader, len(dataset.classes), args.tree_depth)     
        output_history_path=os.path.join(OUTPUT_DIRECTORY,f'history_{current_time}_{EPOCHS}_{SEED}.csv')
        pd.DataFrame(data={'train_loss':history_train_loss, 'val_loss':history_val_loss,'lr':history_LR}).to_csv(output_history_path)

    ## LOAD BEST MODEL
    siamese_net.eval()
    labels = []
    embeddings_train = []
    true_class_train = []
    siamese_net.load_state_dict(best_model_weights)
    print(f'Best model loss: {best_loss}')

    emb_dataloader = DataLoader(dataset.emb(), batch_size=100, shuffle=False)
    with torch.no_grad():
        tqe = tqdm(emb_dataloader, desc=f"Output final embedding samples")
        for batch in tqe:
            img, anchor_class = batch
            img=img.to(DEVICE)
            embedding = siamese_net.forward_one(img)
            if CUDA:
                embedding = embedding.cpu().detach()
                embedding = embedding.numpy()
            for e in embedding:
                embeddings_train.append(e)
            for a_cls in anchor_class:
                true_class_train.append(a_cls)

    embeddings_train = [e.flatten() for e in embeddings_train]
    embeddings_train = np.stack(embeddings_train)
    # print(test_dataset.classes2onehot)
    output_tsne_train_path = os.path.join(OUTPUT_DIRECTORY, f'output_tsne_train_{current_time}_{EPOCHS}_{SEED}.svg')
    silhouette_score_train=visualize_embeddings(embeddings_train, true_class_train, f't-SNE (train) Epochs={EPOCHS} N={len(true_class_train)}', output_tsne_train_path)
    output_embedding_path = os.path.join(OUTPUT_DIRECTORY, f'output_tsne_train_embedding_{current_time}_{EPOCHS}_{SEED}.npz')
    np.savez(output_embedding_path, embedding=np.array(embeddings_train), labels=np.array(true_class_train))

    # IF TEST DATA IS PRESENT
    if len(dataset.test_df) > 0:
        BATCH_SIZE=32
        test_dataloader = DataLoader(dataset.test(), batch_size=BATCH_SIZE, shuffle=True)
        embeddings = []
        true_class = []
        with torch.no_grad():
            tqt = tqdm(test_dataloader, desc=f"Output test embedding samples")
            for batch in tqt:
                img, anchor_class = batch
                img = img.to(DEVICE)
                embedding = siamese_net.forward_one(img)
                if CUDA:
                    embedding = embedding.cpu().detach()
                    embedding = embedding.numpy()
                for e in embedding:
                    embeddings.append(e)
                for a_cls in anchor_class:
                    true_class.append(a_cls)

        embeddings = [e.flatten() for e in embeddings]
        embeddings = np.stack(embeddings)
        output_tsne_test_path = os.path.join(OUTPUT_DIRECTORY, f'output_tsne_test_{current_time}_{EPOCHS}_{SEED}.svg')
        silhouette_score_test=visualize_embeddings(embeddings, true_class, f't-SNE (test) Epochs={EPOCHS} N={len(true_class)}', output_tsne_test_path)
        results, clsidx, confusion, class_report=evaluate(embeddings_train, true_class_train, embeddings, true_class)
        output_results_path = os.path.join(OUTPUT_DIRECTORY, f'model_{current_time}_{EPOCHS}_{SEED}.results.txt')
        print(f"Saving results to {output_results_path}")
        with open(output_results_path,'w') as f:
            f.write("\n")
            f.write(results.__str__())
            f.write("\n")
            f.write(args.__str__())
            f.write("\n")
            f.write(confusion.__str__())
            f.write("\n")
            f.write(clsidx.__str__())
            f.write("\n")
            f.write(class_report.__str__())
            f.write("\n")
            f.write(f'silhouette_score_train:\t{silhouette_score_train}\n')
            f.write(f'silhouette_score_test:\t{silhouette_score_test}\n')

    # FINAL OUTPUT OF THE MODEL
    print(f"Done. Saving final model to {OUTPUT_MODEL}.\n")
    #torch.save(siamese_net.state_dict(), OUTPUT_MODEL)
    output_model_path = os.path.join(OUTPUT_DIRECTORY, f'model_{current_time}_{EPOCHS}_{SEED}.pth')
    print(f'Also saving model to {output_model_path}\n')
    torch.save(siamese_net.state_dict(), output_model_path)

