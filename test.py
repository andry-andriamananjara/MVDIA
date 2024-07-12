# Import modules
import os
import sys
import cv2
import gc
import shutil
import numpy as np
import pandas as pd
import itertools
import logging
import argparse
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv
import timm
from glob import glob
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from decord import VideoReader,VideoLoader
from decord import cpu, gpu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, DistilBertModel, DistilBertConfig, DistilBertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

# Configurations
class CFG:
    debug = False
    image_path = "./training"
    captions_path = "./training"
    checkpoint_dir = "./kl2captioncheckpoints"
    batch_size = 32
    num_workers = 8
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 10
    factor = 0.8
    epochs = 250
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 0.05

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1



class CLIPDataset(torch.utils.data.Dataset):
    """
    Custom dataset for CLIP model validation.
    """
    def __init__(self, image_filenames, transforms):
        """
        Initializes the dataset.

        Args:
            image_filenames (array): Array of image filenames.
            transforms (albumentations.Compose): Image transformations.
        """

        self.image_filenames = image_filenames
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing image and label.
        """
        item = {}
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        item["image"] = image_tensor

        return item

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_filenames)
        
def get_transforms():
    """
    Returns image transformations.

    Args:

    Returns:
        albumentations.Compose: Image transformations.
    """
    return A.Compose(
        [
            A.Resize(CFG.size, CFG.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )
        
def build_loaders(dataframe, imagepaths_columnname):
    """
    Build data loaders for validation.

    Args:
        dataframe (pd.DataFrame): Dataframe containing image paths and labels.
        imagepaths_columnname (string): Column name of paths to the test images 

    Returns:
        torch.utils.data.DataLoader: Data loader instance.
    """
    transforms = get_transforms()
    dataset = CLIPDataset(
        dataframe[imagepaths_columnname].values,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= 1,
        num_workers=CFG.num_workers,
        shuffle=False,
    )
    return dataloader


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        """
        Initializes the image encoder.

        Args:
            model_name (str): Name of the image encoder model.
            pretrained (bool): Whether to use pre-trained weights.
            trainable (bool): Whether the encoder is trainable.
        """
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """
        Forward pass of the image encoder.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Encoded image features.
        """
        return self.model(x)


class TextEncoder(nn.Module):
    """
    Encodes text to a fixed-size vector.
    """
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        """
        Initializes the text encoder.

        Args:
            model_name (str): Name of the text encoder model.
            pretrained (bool): Whether to use pre-trained weights.
            trainable (bool): Whether the encoder is trainable.
        """
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # Using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the text encoder.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input.

        Returns:
            torch.Tensor: Encoded text features.
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    """
    Projection head module for projecting embeddings.
    """
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        """
        Initializes the projection head.

        Args:
            embedding_dim (int): Dimension of input embeddings.
            projection_dim (int): Dimension of projected embeddings.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        """
        Forward pass of the projection head.

        Args:
            x (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Projected embeddings.
        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    """
    CLIP (Contrastive Language-Image Pretraining) model.
    """
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        """
        Initializes the CLIP model.

        Args:
            temperature (float): Temperature parameter for contrastive loss.
            image_embedding (int): Dimension of image embeddings.
            text_embedding (int): Dimension of text embeddings.
        """
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        """
        Forward pass of the CLIP model.
 
        Args:
            batch (dict): Dictionary containing image and text inputs.

        Returns:
            tuple: Tuple containing image and text embeddings.
        """
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings
        
        
def test(model, test_loader):
    """
    Function to test the model's performance on the test dataset.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        list: A list containing predictions.
    """

    # Initialize tqdm progress bar
    tqdm_object = tqdm(test_loader, total=len(test_loader))

    # Initialize lists to store predictions and labels
    predictions = []

    # Load labels from CSV file
    temp_df = pd.read_csv('Clip_label.csv')
    labels = list(temp_df['name'].values)

    # Initialize tokenizer and encode captions
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_captions = tokenizer(labels, padding=True, truncation=True, max_length=CFG.max_length)
    item = {key: torch.tensor(values) for key, values in encoded_captions.items()}

    # Validation phase
    model.eval()
    # Disable gradients
    with torch.no_grad():
        for batch in tqdm_object:
            # Use model to encode image and text features
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_features = image_features.repeat(len(labels),1,1,1,1)
            text_features = model.text_encoder(input_ids=item["input_ids"].to(CFG.device), attention_mask=item["attention_mask"].to(CFG.device))

            # Get embeddings from model projections
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)

            # Calculate the dot product similarity between image and text embeddings
            dot_similarity = image_embeddings @ text_embeddings.T

            # Get the top-5 values and corresponding indices
            values, indices_pred = torch.topk(dot_similarity.squeeze(0), 5)
            indices_pred = indices_pred.detach().cpu().numpy()

            # Get the predicted label
            predicted_label = indices_pred[0][0][0][0][0] + 1
            
            # Append the prediction to list
            predictions.append(predicted_label)
        
    return predictions
    
def parse_option():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepathscsv', type=str, required = True, help='Path to the CSV file containing the test images file paths.')
    parser.add_argument('--columnname', type=str, required = True, help='Name of the column in the CSV file containing the paths to the test images.')
    args = parser.parse_args()
    return args

def main(imagepathscsv, imagepaths_columnname):
    # Load model
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load("kl2captioncheckpoints/best_0.81.pth"))
    
    # Load images
    test_df = pd.read_csv(imagepathscsv)
    test_loader = build_loaders(test_df, imagepaths_columnname)
    predictions = test(model, test_loader)
    
    return predictions

if __name__ == '__main__':
    args = parse_option()
    imagepathscsv = args.imagepathscsv
    imagepaths_columnname = args.columnname
    predictions = main(imagepathscsv, imagepaths_columnname)






