from pathlib import Path
import torch


def save_model(model: torch.nn.Module, save_directory: str, model_name: str):
    target_directory = Path(save_directory)
    target_directory.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_path = target_directory / model_name

    torch.save(model.state_dict(), model_path)


def load_model(model: torch.nn.Module, model_state_path: str):
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state)
