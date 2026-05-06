import torch
import torch.nn as nn
import torchvision.models as models
import os

def get_model(num_classes):
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(num_classes, path="models/model.pth", device="cpu"):
    model = get_model(num_classes)
    model.to(device)

    # ✅ SAFE CHECK
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    else:
        print("⚠️ Model file not found. Returning untrained model.")
        model.eval()
        return model