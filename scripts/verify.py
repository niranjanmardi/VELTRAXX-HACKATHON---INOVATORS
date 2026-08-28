import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import struct
import numpy as np

def read_sparse_bin(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b'VCSR', "Invalid magic bytes in CSR file"
        version = struct.unpack('<I', f.read(4))[0]
        num_layers = struct.unpack('<I', f.read(4))[0]
        
        layers = []
        for _ in range(num_layers):
            nr, nc, nnz = struct.unpack('<III', f.read(12))
            
            vals = np.frombuffer(f.read(nnz * 4), dtype=np.float32)
            col_ind = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
            row_ptr = np.frombuffer(f.read((nr + 1) * 4), dtype=np.int32)
            bias = np.frombuffer(f.read(nr * 4), dtype=np.float32)
            
            # Reconstruct dense
            dense_w = np.zeros((nr, nc), dtype=np.float32)
            for r in range(nr):
                start = row_ptr[r]
                end = row_ptr[r+1]
                for idx in range(start, end):
                    c = col_ind[idx]
                    dense_w[r, c] = vals[idx]
            
            layers.append((dense_w, bias))
    return layers

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

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    logs_dir = os.path.join(base_dir, "logs")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    baseline_model = MLP().to(device)
    baseline_model.load_state_dict(torch.load(os.path.join(outputs_dir, "baseline_model.pth")))
    baseline_acc = evaluate(baseline_model, test_loader, device)

    sparse_layers = read_sparse_bin(os.path.join(outputs_dir, "sparse_weights.bin"))
    sparse_model = MLP().to(device)
    state_dict = sparse_model.state_dict()
    
    state_dict['fc1.weight'] = torch.from_numpy(sparse_layers[0][0]).to(device)
    state_dict['fc1.bias'] = torch.from_numpy(sparse_layers[0][1]).to(device)
    state_dict['fc2.weight'] = torch.from_numpy(sparse_layers[1][0]).to(device)
    state_dict['fc2.bias'] = torch.from_numpy(sparse_layers[1][1]).to(device)
    state_dict['fc3.weight'] = torch.from_numpy(sparse_layers[2][0]).to(device)
    state_dict['fc3.bias'] = torch.from_numpy(sparse_layers[2][1]).to(device)
    
    sparse_model.load_state_dict(state_dict)
    sparse_acc = evaluate(sparse_model, test_loader, device)

    acc_drop = baseline_acc - sparse_acc
    status = "PASS (within 1% threshold)" if acc_drop <= 1.0 else "FAIL"

    report = f"""Dense Model Accuracy:   {baseline_acc:.2f}%
Sparse Model Accuracy:  {sparse_acc:.2f}%
Accuracy Drop:          {acc_drop:.2f}%
Status: {status}
"""
    print(report)
    with open(os.path.join(logs_dir, "accuracy_report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
