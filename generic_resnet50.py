########################################################################################################################
### Taxonomic Loss using Siamese Networks - 2024/2025
### Authors: Hans-Olivier Fontaine (UdeS/AAFC), Etienne Lord (AAFC, etienne.lord@agr.gc.ca)
### Copyright Agriculture and Agri-Food Canada
########################################################################################################################
import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Define data transforms
def get_data_transforms(input_size):
    return {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# Load datasets
def load_datasets(data_dir, input_size):
    data_transforms = get_data_transforms(input_size)
    return {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
            for x in ['train', 'val', 'test']}

# Create data loaders
def create_dataloaders(image_datasets, batch_size, num_workers):
    return {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for x in ['train', 'val', 'test']}

# Load pre-trained ResNet50 model
def create_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_model(model, criterion, optimizer, dataloaders, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

#        for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

def evaluate(model, test_dataloader, device, output_results_path):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    results = {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'F1': f1}
    print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}')
    
    classification_rep = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Get class names
    class_names = test_dataloader.dataset.classes
    cls2idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    print(f"Saving results to {output_results_path}")
    with open(output_results_path, 'w') as f:
        f.write("Results:\n")
        f.write(str(results))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\nClass to Index Mapping:\n")
        f.write(str(cls2idx))
        f.write("\n\nClassification Report:\n")
        f.write(str(classification_rep))
    
    return results, cls2idx, classification_rep, conf_matrix

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # OUTPUT
    OUTPUT_DIRECTORY=args.output
    ## CREATE -OUTPUT- DIRECTORY
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    ## VALIDATE SOME INPUTS ##
    if not os.path.isdir(OUTPUT_DIRECTORY):
        print(f'The output directory {OUTPUT_DIRECTORY} is not a directory or could not be created!')
        exit(-1)
                    
    # Load datasets and create dataloaders
    image_datasets = load_datasets(args.data_dir, args.imgsz)
    dataloaders = create_dataloaders(image_datasets, args.batch_size, args.num_workers)
    
    # Create model
    num_classes = len(image_datasets['train'].classes)
    model = create_model(num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Train the model
    trained_model = train_model(model, criterion, optimizer, dataloaders, args.epoch, device)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_results_path = os.path.join(OUTPUT_DIRECTORY, f'model_{current_time}_{args.epoch}_{args.seed}.results.txt')
    
    # Evaluate the model
    results, cls2idx, classification_rep, conf_matrix = evaluate(
        trained_model, dataloaders['test'], device, output_results_path
    )
    
    print("Evaluation completed. Results saved to", output_results_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ResNet50 model on custom dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output", type=str, default="evaluation_results.txt", help="Path to save evaluation results")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument("--epoch", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)  
