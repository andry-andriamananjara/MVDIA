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
from sklearn.utils import shuffle
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
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

# Initialize Tensorboard writer
writer = SummaryWriter()

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

class AvgMeter:
    """
    Computes and stores the average and current value
    of a given metric.
    """
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        """Resets the meter."""
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        """
        Updates the meter with new value.

        Args:
            val (float): New value to update.
            count (int, optional): Number of samples contributing to the value.
         """
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        """Representation of the metric."""
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    """Returns the current learning rate of the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]

        
class CLIPDataset(torch.utils.data.Dataset):
    """
    Custom dataset for CLIP model training/validation.
    """
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        Initializes the dataset.

        Args:
            image_filenames (array): Array of image filenames.
            captions (array): Array of corresponding captions.
            tokenizer (DistilBertTokenizer): Tokenizer for encoding captions.
            transforms (albumentations.Compose): Image transformations.
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing image, caption, and encoded caption.
        """
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.captions)



def get_transforms(mode="train"):
    """
    Returns image transformations based on the mode.
 
    Args:
        mode (str): Mode of transformation, 'train', 'valid' or 'test'.

    Returns:
        albumentations.Compose: Image transformations.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

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

class KLLoss(nn.Module):
    """
    KL Divergence loss function.
    """
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        """
        Initializes the KL Divergence loss function.

        Args:
            error_metric (nn.KLDivLoss): KL Divergence loss instance.
        """
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label):
        """
        Forward pass of the KL Divergence loss function.

        Args:
            prediction (torch.Tensor): Predicted logits.
            label (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy loss function.
    """
    def __init__(self, reduction='none'):
        """
        Initializes the Cross Entropy loss function.

        Args:
            reduction (str): Method of reducing the loss, 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Forward pass of the Cross Entropy loss function.

        Args:
            preds (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        log_softmax = F.log_softmax(preds, dim=-1)
        loss = (-targets * log_softmax).sum(1)
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

def make_train_valid_dfs():
    """
    Create training and validation dataframes.

    Returns:
        tuple: Tuple containing training and validation dataframes.
    """
    dataframe = pd.read_excel(f"{CFG.captions_path}/train.xlsx")
    max_id = dataframe.shape[0] if not CFG.debug else 100
    image_ids = np.arange(0, max_id)

    # Stratified sampling based on the "Class" column
    train_ids, valid_ids = train_test_split(image_ids, test_size=0.2, stratify=dataframe["Class"], random_state=42)

    train_dataframe = dataframe[dataframe.index.isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe.index.isin(valid_ids)].reset_index(drop=True)
    
    return train_dataframe, valid_dataframe

def save_ckp(state, epoch, checkpoint_dir):
    """
    Save model checkpoint.

    Args:
        state (dict): Model state dictionary.
        epoch (int): Current epoch.
        checkpoint_dir (str): Directory to save checkpoints.
    """
    save_path = os.path.join(checkpoint_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, save_path)


def build_loaders(dataframe, tokenizer, mode):
    """
    Build data loaders for training/validation.

    Args:
        dataframe (pd.DataFrame): Dataframe containing image paths and captions.
        tokenizer (DistilBertTokenizer): Tokenizer for encoding captions.
        mode (str): Mode of the data loader, 'train', 'valid', or 'test'.

    Returns:
        torch.utils.data.DataLoader: Data loader instance.
    """
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["Path"].values,
        dataframe["Caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= 1 if mode == "test" else CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, loss_fn):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model instance.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler instance.
        step (str): Whether the scheduler step is per batch or per epoch.
        loss_fn: Loss function instance.

    Returns:
        AvgMeter: Average loss meter for the epoch.
    """
    # Initialize the average loss meter and tqdm progress bar
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    scaler = GradScaler()
    for batch in tqdm_object:
        # Get batch caption
        caption = batch["caption"]
        # Backpropagation
        optimizer.zero_grad()
        with autocast():
            # Use model to encode image and text features
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            text_features = model.text_encoder(input_ids=batch["input_ids"].to(CFG.device), attention_mask=batch["attention_mask"].to(CFG.device))

            # Get embeddings from model projections
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)

            # Calculating the Loss (Jensen-Shannon (JS) Divergence)
            # Calculate dot product similarity between text embeddings and image embeddings
            texts_similarity = (text_embeddings @ image_embeddings.T) / CFG.temperature
            # Calculate dot product similarity between image embeddings and text embeddings
            images_similarity = (image_embeddings @ text_embeddings.T) / CFG.temperature
            # Generate target matrix
            target = np.array([[1.0 if caption[i] == c else 0.0 for c in caption] for i in range(len(caption))])
            target = torch.tensor(target, dtype=image_embeddings.dtype, device=CFG.device)
            # Calculate JS loss
            loss = (loss_fn(texts_similarity, target) + loss_fn(images_similarity, target))/2


            # # Calculating the Loss (Cross-Entropy (CE))
            # # Calculate logits
            # logits = (text_embeddings @ image_embeddings.T) / CFG.temperature
            # # Calculate dot product similarity between image embeddings and text embeddings
            # images_similarity = image_embeddings @ image_embeddings.T
            # # Calculate dot product similarity between text embeddings and image embeddings
            # texts_similarity = text_embeddings @ text_embeddings.T
            # # Generate target matrix
            # targets = F.softmax(
                # (images_similarity + texts_similarity) / 2 * CFG.temperature, dim=-1
            # )
            # # Calculate CE loss
            # texts_loss = cross_entropy(logits, targets, reduction='none')
            # images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            # loss =  (images_loss + texts_loss) / 2.0 
            # loss = loss.mean()
        
        # Backpropagate and update weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update LR scheduler
        if step == "batch":
            lr_scheduler.step()

        # Update loss meter 
        count = len(caption)
        loss_meter.update(loss.item(), count)
        # Display progress with average training loss and learning rate
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, valid_loader, test_loader, loss_fn):
    """
    Validate the model.

    Args:
        model (nn.Module): Model instance.
        valid_loader (torch.utils.data.DataLoader): Validation data loader
        test_loader (torch.utils.data.DataLoader): Validation data loader.
        loss_fn: Loss function instance.

    Returns:
        AvgMeter: Average loss meter for the validation.
    """

    # Initialize the average loss meter and tqdm progress bar
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))

    # Initialize accuracy metrics
    acc_1, acc_5 = 0, 0

    # Calculate the total number of items
    total_number = len(test_loader)

    # Load labels from CSV file
    temp_df = pd.read_csv('Clip_label.csv')
    labels = list(temp_df['name'].values)

    # Initialize tokenizer and encode captions
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_captions = tokenizer(labels, padding=True, truncation=True, max_length=CFG.max_length)
    item = {key: torch.tensor(values) for key, values in encoded_captions.items()}

    # Disable gradients
    with torch.no_grad():
        for batch in tqdm_object:
            # Get batch caption
            caption = batch["caption"]

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

            # Update accuracy metrics for top-1 and top-5 predictions
            acc_1 += (labels[indices_pred[0][0][0][0][0]] == caption[0])
            for a in indices_pred[0][0][0][0]:
                 acc_5 += (labels[a] == caption[0])

    # Initialize the tqdm progress bar
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    # Disable gradients
    with torch.no_grad():
        for batch in tqdm_object:
            # Get batch caption
            caption = batch["caption"]
            # Use model to encode image and text features
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            text_features = model.text_encoder(input_ids=batch["input_ids"].to(CFG.device), attention_mask=batch["attention_mask"].to(CFG.device))
            
            # Get embeddings from model projections
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)

            # Calculating the Loss (Jensen-Shannon (JS) Divergence)
            # Calculate dot product similarity between text embeddings and image embeddings
            texts_similarity = (text_embeddings @ image_embeddings.T) / CFG.temperature
            # Calculate dot product similarity between image embeddings and text embeddings
            images_similarity = (image_embeddings @ text_embeddings.T) / CFG.temperature
            # Generate target matrix
            target = np.array([[1.0 if caption[i] == c else 0.0 for c in caption] for i in range(len(caption))])
            target = torch.tensor(target, dtype=image_embeddings.dtype, device=CFG.device)
            # Calculate JS loss
            loss = (loss_fn(texts_similarity,target) + loss_fn(images_similarity, target))/2


            # # Calculating the Loss (Cross-Entropy (CE))
            # # Calculate logits
            # logits = (text_embeddings @ image_embeddings.T) / CFG.temperature
            # # Calculate dot product similarity between image embeddings and text embeddings
            # images_similarity = image_embeddings @ image_embeddings.T
            # # Calculate dot product similarity between text embeddings and image embeddings
            # texts_similarity = text_embeddings @ text_embeddings.T
            # # Generate target matrix
            # targets = F.softmax(
                # (images_similarity + texts_similarity) / 2 * CFG.temperature, dim=-1
            # )
            # # Calculate CE loss
            # texts_loss = cross_entropy(logits, targets, reduction='none')
            # images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            # loss =  (images_loss + texts_loss) / 2.0 
            # loss = loss.mean()

            # Update loss meter
            loss_meter.update(loss.item(), batch["image"].size(0))

            # Display progress with average validation loss
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    # Print the top-1 and top-5 accuracies
    print('acc@1 == {:.2f}%'.format((acc_1/total_number) * 100))
    print('acc@5 == {:.2f}%'.format((acc_5/total_number) * 100))

    # Return loss meter and top-1 accuracy
    return loss_meter, acc_1/total_number


def main():
    """
    Main function that trains and validates the model.
    """

    # Create directory for checkpoints
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    # # Set up logger
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler(os.path.join(CFG.checkpoint_dir, 'train.log'))
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)

    # Print and log experiment configuration
    # logger.info("Experiment Configuration:")
    # for key, value in vars(CFG).items():
        # logger.info(f"{key}: {value}")


    # Create training and validation dataframes
    train_df, valid_df = make_train_valid_dfs()
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

    # Build data loaders
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    test_loader = build_loaders(valid_df, tokenizer, mode="test")
    
    # Initialize model, optimizer, and LR scheduler
    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    # Define Loss function
    loss_fn = KLLoss()

    # Initialize best_loss and best accuracy    
    best_loss = float('inf')
    best_acc = 0

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        # Training phase
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, loss_fn)
        if (epoch+1)%20 == 0:
            state = {'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_scheduler': lr_scheduler.state_dict(),
                      'epoch': epoch}
            save_ckp(state, epoch, CFG.checkpoint_dir)
        writer.add_scalar('train_loss', train_loss.avg, epoch)

        # Validation phase
        model.eval()
        with torch.no_grad():
            valid_loss,acc = valid_epoch(model, valid_loader, test_loader, loss_fn)
            writer.add_scalar('valid_loss', valid_loss.avg, epoch)
            writer.add_scalar('acc@1', acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(CFG.checkpoint_dir, f'best_{round(best_acc, 2)}.pth'))
            print("Saved Best Model!")
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print("Best Loss")
        lr_scheduler.step(valid_loss.avg)

if __name__ == '__main__':
    main()