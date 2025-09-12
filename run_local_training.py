import torch
from torch import optim
from models.cnn_model import create_model
from hospital_client.data_loader import get_dataloaders
from hospital_client.trainer import train_one_epoch, evaluate


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = create_model(in_channels=1, num_classes=2)
    model.to(device)

    train_loader, val_loader = get_dataloaders(batch_size=32, val_split=0.2, num_samples=800, image_size=224, in_channels=1, pos_ratio=0.35, seed=42)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        print({"epoch": epoch, "train_loss": train_loss, **metrics})

    from pathlib import Path
    save_path = Path(__file__).resolve().parent / "local_model.pth"
    torch.save(model.state_dict(), str(save_path))
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
