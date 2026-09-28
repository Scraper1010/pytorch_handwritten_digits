import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src import DigitCNN, get_dataloaders, get_device, save_training_curves


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total) * 100.0:.2f}%"
        })

    return running_loss / total, (correct / total) * 100.0


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct / total) * 100.0:.2f}%"
            })

    return running_loss / total, (correct / total) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Train DigitCNN on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Train batch size")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory storing raw MNIST data")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model weights")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    device = get_device()
    print(f"--> Using compute device: {device}")

    # 1. Prepare Data
    print("--> Preparing data...")
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        train_batch_size=args.batch_size
    )

    # 2. Initialize Model, Loss, Optimizer & Scheduler
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, "mnist_cnn.pth")

    print(f"--> Starting training for {args.epochs} epochs...\n")

    # 3. Main Loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    [+] Saved best model to {best_model_path} ({best_val_acc:.2f}%)")

    # 4. Generate & Save Artifacts
    print("\n--> Saving training curves...")
    save_training_curves(history, output_path="assets/training_curves.png")
    print(f"--> Done! Best Test Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()