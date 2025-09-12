from typing import Dict, List, Tuple
import flwr as fl
import torch
from models.cnn_model import create_model
import argparse
from hospital_client.data_loader import get_dataloaders


def get_parameters(model: torch.nn.Module) -> List[torch.Tensor]:
    return [val.cpu().detach() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[torch.Tensor]) -> None:
    # Parameters may arrive as numpy arrays from Flower; convert to torch tensors
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


def get_evaluate_fn():
    # Build a validation set on the server for quick global eval (synthetic)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(in_channels=1, num_classes=2).to(device)
    _, val_loader = get_dataloaders(batch_size=64, val_split=0.5, num_samples=400, image_size=224, in_channels=1, pos_ratio=0.35, seed=123)
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(server_round: int, parameters: List[torch.Tensor], config: Dict[str, str]):
        set_parameters(model, parameters)
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return float(total_loss / total), {"accuracy": float(correct / total)}

    return evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2, help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum clients for fit/evaluate/available")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round")
    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda rnd: {"local_epochs": args.local_epochs},
        initial_parameters=fl.common.ndarrays_to_parameters([t.numpy() for t in get_parameters(create_model())]),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )


if __name__ == "__main__":
    main()
