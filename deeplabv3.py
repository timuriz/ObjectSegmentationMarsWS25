import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.optim as optim

num_classes = 4  # 0..3

model = deeplabv3_resnet50(weights="DEFAULT")  # pretrained on COCO

in_channels = model.classifier[4].in_channels  # usually 256
model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# loss function:
criterion = nn.CrossEntropyLoss()  # target: [B,H,W] с 0..3
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,     
    patience=3,     
    verbose=True
)


#training

from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)      # [B,3,H,W]
        masks = masks.to(device)        # [B,H,W]

        optimizer.zero_grad()

        outputs = model(images)         # dict: {"out": [B,C,H,W], ...}
        logits = outputs["out"]

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

#validation

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        logits = outputs["out"]

        loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# cycle

def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=20,
    patience=5,
    save_path="deeplab_best.pth"
):
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)

        print(f"  train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f}")

        # step scheduler for val loss
        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print("  ✅ New best model saved.")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("  ⛔ Early stopping triggered.")
            break