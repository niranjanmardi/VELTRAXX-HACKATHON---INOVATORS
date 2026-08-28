import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

def get_threshold(model, target_sparsity):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.abs().cpu().numpy().flatten())
    all_weights = np.concatenate(all_weights)
    threshold = np.percentile(all_weights, target_sparsity * 100)
    return threshold

def create_masks(model, threshold):
    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            masks[name] = (param.data.abs() >= threshold).float()
    return masks

def apply_masks(model, masks):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data *= masks[name]

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    logs_dir = os.path.join(base_dir, "logs")

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
    model.load_state_dict(torch.load(os.path.join(outputs_dir, "baseline_model.pth")))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    pruning_schedule = [(0.30, 3), (0.50, 3), (0.65, 5)]
    masks = {}

    log_file = open(os.path.join(logs_dir, "pruning_log.txt"), "w")

    for round_idx, (target_sparsity, fine_tune_epochs) in enumerate(pruning_schedule):
        print(f"--- Pruning Round {round_idx+1}: Target Sparsity {target_sparsity*100}% ---")
        threshold = get_threshold(model, target_sparsity)
        masks = create_masks(model, threshold)
        apply_masks(model, masks)

        for epoch in range(fine_tune_epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                # Apply mask to gradients
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        param.grad.data *= masks[name]
                optimizer.step()
                apply_masks(model, masks)
        
        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Sparsity per layer
        total_params = 0
        total_zeros = 0
        log_file.write(f"Round {round_idx+1} (Sparsity {target_sparsity*100}%)\n")
        print(f"Round {round_idx+1} Accuracy: {accuracy:.2f}%")
        for name, param in model.named_parameters():
            if 'weight' in name:
                zeros = (param.data == 0).sum().item()
                total = param.numel()
                total_params += total
                total_zeros += zeros
                layer_sparsity = zeros / total
                log_file.write(f"  {name} sparsity: {layer_sparsity:.4f}\n")
        
        overall_sparsity = total_zeros / total_params
        log_file.write(f"  Overall sparsity: {overall_sparsity:.4f}\n")
        log_file.write(f"  Accuracy: {accuracy:.2f}%\n\n")
        print(f"Overall sparsity achieved: {overall_sparsity*100:.2f}%")
    
    log_file.close()

    torch.save(model.state_dict(), os.path.join(outputs_dir, "pruned_model.pth"))
    torch.save(masks, os.path.join(outputs_dir, "pruning_masks.pth"))
    print("Pruning finished. Pruned model saved.")

if __name__ == "__main__":
    main()
