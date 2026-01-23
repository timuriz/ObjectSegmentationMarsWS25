import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MarsSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_files = sorted(os.listdir(images_dir))
        mask_files = set(os.listdir(masks_dir))
        for f in self.image_files:
            if f not in mask_files:
                raise FileNotFoundError(f"Missing mask for image: {f}") #returns error if no mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        image = Image.open(os.path.join(self.images_dir, filename)).convert("RGB") #fixed Mac/Windows/Linux compatibility
        mask = Image.open(os.path.join(self.masks_dir, filename))

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask) 

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] #different normalization
    )
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST), #interpolation
])

train_dataset = MarsSegmentationDataset(
    images_dir="data/images/train",
    masks_dir="data/masks/train",
    image_transform=image_transform,
    mask_transform=mask_transform
)

val_dataset = MarsSegmentationDataset(
    images_dir="data/images/val",
    masks_dir="data/masks/val",
    image_transform=image_transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

#faster data transfer

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


#i've applied different normalization method, also added a part which checks for a mask of a pic, 
#if there's no mask - turns back an error, so it all wont get ruined. also some minor changes like other
#normalization method, faster data loader, mask gets resized too

