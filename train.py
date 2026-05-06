import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model, save_model
from data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, valid_loader, _, class_names = get_data_loaders()
model = get_model(len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

save_model(model)
print("Model saved!")