from typing import Dict, List, Tuple
import flwr as fl
import torch
from torch import optim
from models.cnn_model import create_model
from hospital_client.data_loader import get_dataloaders
from hospital_client.trainer import train_one_epoch, evaluate


def get_parameters(model: torch.nn.Module) -> List[torch.Tensor]:
    return [val.cpu().detach() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[torch.Tensor]) -> None:
    state_dict = model.state_dict()
    new_state = {}
    for (k, old_val), v in zip(state_dict.items(), parameters):
        if isinstance(v, torch.Tensor):
            t = v
        else:
            try:
                import numpy as np  # type: ignore
                if isinstance(v, np.ndarray):
                    t = torch.from_numpy(v)
                else:
                    t = torch.tensor(v)
            except Exception:
                t = torch.tensor(v)
        new_state[k] = t.to(dtype=old_val.dtype)
    model.load_state_dict(new_state, strict=True)


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, cid: str, num_samples: int = 600, seed: int = 0, batch_size: int = 32):
        self.cid = cid
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = create_model(in_channels=1, num_classes=2).to(self.device)
        # Make non-IID by shifting pos_ratio and seed per client
        pos_ratio = 0.25 + (int(cid) % 3) * 0.1
        self.train_loader, self.val_loader = get_dataloaders(batch_size=batch_size, val_split=0.2, num_samples=num_samples, image_size=224, in_channels=1, pos_ratio=pos_ratio, seed=seed + int(cid))
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config: Dict[str, str]):
        return [t.numpy() for t in get_parameters(self.model)]

    def fit(self, parameters, config):
        set_parameters(self.model, [torch.tensor(p) for p in parameters])
        local_epochs = int(config.get("local_epochs", 1))
        for _ in range(local_epochs):
            train_one_epoch(self.model, self.train_loader, self.optimizer, self.device)
        num_examples = len(self.train_loader.dataset)
        return self.get_parameters(config), num_examples, {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, [torch.tensor(p) for p in parameters])
        metrics = evaluate(self.model, self.val_loader, self.device)
        # Flower expects (loss, num_examples, metrics)
        return float(metrics["val_loss"]), len(self.val_loader.dataset), {"val_acc": float(metrics["val_acc"]), "val_auc": float(metrics["val_auc"])}


def main():
    import os
    import argparse
    cid = os.environ.get("CLIENT_ID", "0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    client = HospitalClient(cid=cid, num_samples=args.num_samples, seed=args.seed, batch_size=args.batch_size)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
