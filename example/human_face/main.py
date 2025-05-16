import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt

dataset = fetch_olivetti_faces(shuffle=True)
X = dataset.images  # Shape: (400, 64, 64)
y = dataset.target  # Labels: 40 unique people

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Add channel dim (N, 1, 64, 64)
y = torch.tensor(y, dtype=torch.long)

# Split into training (80%) and test (20%) sets
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_X, test_X = random_split(list(zip(X, y)), [train_size, test_size])

# Create DataLoader
batch_size = 16
train_loader = DataLoader(train_X, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_X, batch_size=batch_size)

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust based on pooling
        self.fc2 = nn.Linear(128, 40)  # 40 classes (people)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x=self.conv1(x)
        x = F.max_pool2d(x, 2)  # Downscale (64 → 32)
        # x = F.relu(self.conv2(x))
        x=self.conv2(x)
        x = F.max_pool2d(x, 2)  # Downscale (32 → 16)
        x = x.view(x.size(0), -1)  # Flatten
        x=self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)  # No softmax (handled by loss function)

# Instantiate model
model = FaceCNN()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs = 20
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         images, labels = batch
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
    
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# # Convert train and test data to tensors
# train_data = (torch.stack([x for x, _ in train_X]), torch.tensor([y for _, y in train_X]))
# test_data = (torch.stack([x for x, _ in test_X]), torch.tensor([y for _, y in test_X]))
# torch.save(train_data, "train_data.pt")
# torch.save(test_data, "test_data.pt")
# torch.save(model.state_dict(), "face_cnn.pth")

# model_p=torch.load("face_cnn.pth")
# model.eval()
# correct, total = 0, 0
# with torch.no_grad():
#     for batch in test_loader:
#         images, labels = batch
#         images, labels = images.to(device), labels.to(device)

#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)  # Get class index with highest score
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")


model_p=torch.load("face_cnn.pth")
model.load_state_dict(model_p)
model.eval()
test_data = torch.load("test_data.pt")
test_images, test_labels = test_data
correct, total = 0, 0
with torch.no_grad():
    inputs=test_images[0]
    labels=test_labels[0]
    inputs=inputs[:,None,:,:]
    images, labels = inputs.to(device), labels.to(device)

    outputs = model(images)
    print(outputs)
    _, predicted = torch.max(outputs, 1)  # Get class index with highest score
    total += 1
    correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")