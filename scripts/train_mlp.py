import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    logs_dir = os.path.join(base_dir, "logs")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.flatten(x)
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
            return x

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("Training baseline MLP...")
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    
    model_path = os.path.join(outputs_dir, "baseline_model.pth")
    torch.save(model.state_dict(), model_path)
    
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    log_content = f"Baseline accuracy: {accuracy:.2f}%\nModel size: {model_size_mb:.2f} MB\n"
    print(log_content)
    with open(os.path.join(logs_dir, "training_log.txt"), "w") as f:
        f.write(log_content)

if __name__ == "__main__":
    main()
