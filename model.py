import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes):
    model = models.resnet18(pretrained=True)

    # freeze feature extractor
    for param in model.parameters():
        param.requires_grad = False

    # replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def save_model(model, path="models/model.pth"):
    torch.save(model.state_dict(), path)


def load_model(num_classes, path="models/model.pth", device="cpu"):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model