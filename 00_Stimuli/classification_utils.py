import pandas as pd
import numpy as np
import os
import pickle
import time 
import copy 

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from sklearn.model_selection import StratifiedKFold

from PIL import Image
from collections import OrderedDict
from tqdm import tqdm

import cornet

#### Loading helpers ####
def load_data(file):
    """
    Load data from a pickle file.
    ------
    Args:
        file (str): The path to the pickle file to be loaded.

    Returns:
        data: The data loaded from the pickle file.
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_data(data, filename):
    """
    Serializes and saves data to a file using pickle.
    ------
    Args:
        data (any): The data to be serialized and saved.
        filename (str): The path to the file where the data will be saved.

    Returns:
        None
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
#### CORNET helpers #### 
def load_cornet_s(pretrained=True, device="cpu"):
    """
    Load CORnet-S with pretrained weights (DataParallel-safe).
    """

    # Initialize model
    model = cornet.CORnet_S()

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            "https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth",
            map_location=device
        )

        # Fix DataParallel naming
        new_state_dict = OrderedDict()
        for k, v in state_dict["state_dict"].items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)

    return model

def cornet_two_class(device="cpu", freeze_backbone=True):
    model = load_cornet_s(device=device)

    in_features = model.decoder.linear.in_features
    model.decoder.linear = nn.Linear(in_features, 2)  # [animate, inanimate]

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("decoder"):
                param.requires_grad = False

    return model.to(device)


class ImagePathDataset(Dataset):
    def __init__(self, paths, label, transform=None):
        self.paths = paths
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        image_id = path  # stable ID
        return img, self.label, image_id
    
def split_inanimate_indices(n_inanimate, n_core, rng):
    """
    Splits inanimate indices into:
    - core (balanced subset)
    - extra (held-out)
    """

    perm = rng.permutation(n_inanimate)
    core_idx = perm[:n_core]
    extra_idx = perm[n_core:]

    return core_idx, extra_idx

def build_core_and_extra_datasets(animacy_paths, inanimacy_paths, rng, preprocess, batch_size, animate_label=0, inanimate_label=1):
    """
    Builds balanced core dataset (animate + core inanimate)
    and extra inanimate dataset for one shuffle.

    Returns
    -------
    core_dataset : torch.utils.data.Dataset
        Balanced dataset for CV (animate + core inanimate)

    labels : np.ndarray
        Labels aligned with core_dataset (for stratified CV)

    indices : np.ndarray
        Indices for core_dataset

    inanimate_ds_extra : torch.utils.data.Dataset
        Extra inanimate images (never trained on)
    """

    N_ANIMATE = len(animacy_paths)

    # Split inanimate indices
    core_idx, extra_idx = split_inanimate_indices(
        n_inanimate=len(inanimacy_paths),
        n_core=N_ANIMATE,
        rng=rng
    )

    # Paths
    core_animacy_paths = [animacy_paths[i] for i in  np.random.permutation(N_ANIMATE)]
    core_inanimate_paths = [inanimacy_paths[i] for i in core_idx]
    extra_inanimate_paths = [inanimacy_paths[i] for i in extra_idx]

    # Datasets
    animate_ds = ImagePathDataset(
        core_animacy_paths,
        label=animate_label,
        transform=preprocess
    )

    inanimate_ds_core = ImagePathDataset(
        core_inanimate_paths,
        label=inanimate_label,
        transform=preprocess
    )

    inanimate_ds_extra = ImagePathDataset(
        extra_inanimate_paths,
        label=inanimate_label,
        transform=preprocess
    )

    # Core dataset
    core_dataset = ConcatDataset([
        animate_ds,
        inanimate_ds_core
    ])

    # Labels for stratification / CV
    labels = np.array(
        [animate_label] * len(animate_ds) +
        [inanimate_label] * len(inanimate_ds_core)
    )

    indices = np.arange(len(labels))
    
    extra_in_dataset = DataLoader(
        inanimate_ds_extra,
        batch_size=batch_size,
        shuffle=False,      # VERY IMPORTANT — keep order fixed
        pin_memory=True
    )
    

    return core_dataset, labels, indices, extra_in_dataset

#### Train and testing helpers ####
def train_model(model, dataloader_train, criterion, optimizer, scheduler=None, num_epochs=50, device="cuda"):
    """
    Train-only model training loop.
    No validation, no model selection.
    Returns trained model + loss and accuracy curves.
    """

    since = time.time()

    loss_vec_train = []
    acc_vec_train = []

    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        since_epoch = time.time()

        model.train()

        running_loss = 0.0
        running_corrects = 0
        n_samples = 0

        # Iterate over training data
        for inputs, labels, _ in dataloader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            n_samples += batch_size

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples

        loss_vec_train.append(epoch_loss)
        acc_vec_train.append(epoch_acc.detach().cpu().numpy())

        #print('Train Loss: {:.4f} Acc: {:.4f}'.format(
            #epoch_loss, epoch_acc))

        time_elapsed_epoch = time.time() - since_epoch
        #print('Epoch complete in {:.0f}m {:.0f}s'.format(
            #time_elapsed_epoch // 60, time_elapsed_epoch % 60))
        #print()

    time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(
        #time_elapsed // 60, time_elapsed % 60))

    return model, loss_vec_train, acc_vec_train

def test_model(model, dataloader, device, results, shuffle_id, fold_id=None):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, image_paths in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            for i, img_path in enumerate(image_paths):
                true_label = int(labels[i].item())
                pred = int(preds[i].item())

                prob_animate = float(probs[i, 0].item())
                prob_inanimate = float(probs[i, 1].item())

                acc = int(pred == true_label)

                entry = results[img_path]
                entry["acc"].append(acc)
                entry["prob_animate"].append(prob_animate)
                entry["prob_inanimate"].append(prob_inanimate)
                entry["shuffle"].append(shuffle_id)
                entry["fold"].append(fold_id)

    return correct / total
