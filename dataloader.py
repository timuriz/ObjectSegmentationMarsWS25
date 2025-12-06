import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class MarsSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # sorted list of image filenames
        self.image_files = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # load image
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # load mask
        mask_path = os.path.join(self.masks_dir, img_name)
        mask = Image.open(mask_path)

        # convert mask -> numpy int64 array (values 0..5)
        mask = np.array(mask, dtype=np.int64)

        # apply transforms to image only
        if self.transform:
            img = self.transform(img)

        # convert mask to tensor AFTER transforms
        mask = torch.from_numpy(mask).long()    # [H, W]

        return img, mask
    
###### transformation

# mask = mask.resize((256,256), Image.NEAREST, if we resize

from torchvision import transforms

img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),          # converts to [0,1]
])

### Data loaders

from torch.utils.data import DataLoader

train_dataset = MarsSegmentationDataset(
    images_dir="data/images/train",
    masks_dir="data/masks/train",
    transform=img_transform
)

val_dataset = MarsSegmentationDataset(
    images_dir="data/images/val",
    masks_dir="data/masks/val",
    transform=img_transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)


# data/
#   images/
#     train/
#       img001.png
#       img002.png
#       ...
#     val/
#       img101.png
#       img102.png

#   masks/
#     train/
#       img001.png
#       img002.png
#       ...
#     val/
#       img101.png
#       img102.png