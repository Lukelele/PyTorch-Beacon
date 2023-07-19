from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms


###########################   Saving and Loading   ###########################


def save_model(model: torch.nn.Module, save_directory: str, model_name: str):
    target_directory = Path(save_directory)
    target_directory.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_path = target_directory / model_name

    torch.save(model.state_dict(), model_path)


def load_model(model: torch.nn.Module, model_state_path: str):
    model_state = torch.load(model_state_path)
    model.load_state_dict(model_state)


########################   For Image Classification   ########################


def get_image_tensor_from_path(image_path: str):
    image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image).unsqueeze_(0)
    image.close()

    return image_tensor


def get_image_from_path(image_path: str):
    image = Image.open(image_path)
    return image


def get_image_from_tensor(image_tensor: torch.Tensor):
    image = transforms.ToPILImage()(image_tensor)
    return image


def display_image_from_path(image_path: str):
    image = Image.open(image_path)
    image.show()
    

def display_image_from_tensor(image_tensor: torch.Tensor):
    image = transforms.ToPILImage()(image_tensor)
    image.show()


def classify(model: torch.nn.Module, image: torch.Tensor, labels: list, device="cpu"):
    model.to(device)
    model.eval()

    with torch.inference_mode():
        y_pred = model(image.to(device))
        y_prob = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_prob, dim=1)

    labels_list = torch.zeros(len(y_pred))
        
    for i, y in enumerate(y_pred):
        labels_list[i] = labels[y.item()].item()
    return labels_list, y_prob


################################   Misc   ################################


def dataloader_to_tensor(dataloader: torch.utils.data.DataLoader, batch_count: int = 0):
    data_tensor = None
    label_tensor = None
    
    for batch, (data, label) in enumerate(dataloader):
        if batch_count == 0 or batch < batch_count:
            if data_tensor is None:
                data_tensor = data
            else:
                data_tensor = torch.cat((data_tensor, data), dim=0)
        
            if label_tensor is None:
                label_tensor = label
            else:
                label_tensor = torch.cat((label_tensor, label), dim=0)

    return data_tensor, label_tensor
