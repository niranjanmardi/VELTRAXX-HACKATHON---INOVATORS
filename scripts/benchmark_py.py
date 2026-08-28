import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
import json
import struct
import numpy as np

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

def read_sparse_bin(path):
    layers = []
    with open(path, "rb") as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        num_layers = struct.unpack('<I', f.read(4))[0]
        for _ in range(num_layers):
            nr, nc, nnz = struct.unpack('<III', f.read(12))
            vals = np.frombuffer(f.read(nnz * 4), dtype=np.float32)
            col_ind = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
            row_ptr = np.frombuffer(f.read((nr + 1) * 4), dtype=np.int32)
            bias = np.frombuffer(f.read(nr * 4), dtype=np.float32)
            dense_w = np.zeros((nr, nc), dtype=np.float32)
            for r in range(nr):
                start = row_ptr[r]
                end = row_ptr[r+1]
                for idx in range(start, end):
                    c = col_ind[idx]
                    dense_w[r, c] = vals[idx]
            layers.append((dense_w, bias))
    return layers

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    logs_dir = os.path.join(base_dir, "logs")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    subset = torch.utils.data.Subset(test_dataset, range(1000))
    test_loader = torch.utils.data.DataLoader(subset, batch_size=1000, shuffle=False)
    data_batch = next(iter(test_loader))[0].to(device)

    # Dense time
    model_dense = MLP().to(device)
    model_dense.load_state_dict(torch.load(os.path.join(outputs_dir, "baseline_model.pth")))
    model_dense.eval()
    
    # Warmup
    with torch.no_grad():
        model_dense(data_batch)
    
    start = time.perf_counter()
    with torch.no_grad():
        model_dense(data_batch)
    dense_time_ms = (time.perf_counter() - start) * 1000

    # Sparse time (reconstruct then run)
    start_recon = time.perf_counter()
    sparse_layers = read_sparse_bin(os.path.join(outputs_dir, "sparse_weights.bin"))
    model_sparse = MLP().to(device)
    state_dict = model_sparse.state_dict()
    state_dict['fc1.weight'] = torch.from_numpy(sparse_layers[0][0]).to(device)
    state_dict['fc1.bias'] = torch.from_numpy(sparse_layers[0][1]).to(device)
    state_dict['fc2.weight'] = torch.from_numpy(sparse_layers[1][0]).to(device)
    state_dict['fc2.bias'] = torch.from_numpy(sparse_layers[1][1]).to(device)
    state_dict['fc3.weight'] = torch.from_numpy(sparse_layers[2][0]).to(device)
    state_dict['fc3.bias'] = torch.from_numpy(sparse_layers[2][1]).to(device)
    model_sparse.load_state_dict(state_dict)
    model_sparse.eval()
    
    with torch.no_grad():
        model_sparse(data_batch)
    sparse_time_ms = (time.perf_counter() - start_recon) * 1000

    speedup = dense_time_ms / sparse_time_ms

    timing_report = f"""=== Timing Report (Python Baseline) ===
Dense inference:  {dense_time_ms:.2f} ms / 1000 samples
Sparse inference: {sparse_time_ms:.2f} ms / 1000 samples
Speedup:          {speedup:.2f}x
Note: C++ engine speedup reported separately
"""
    print(timing_report)
    with open(os.path.join(logs_dir, "timing_report.txt"), "w") as f:
        f.write(timing_report)

    # Memory
    with open(os.path.join(outputs_dir, "sparse_metadata.json")) as f:
        meta = json.load(f)
    
    d_size = meta["dense_size_bytes"]
    s_size = meta["sparse_size_bytes"]
    ratio = meta["memory_reduction_ratio"]
    sparsity = meta["overall_sparsity"] * 100
    
    status = "PASS (target: 2.0x reduction)" if ratio >= 2.0 else "FAIL"

    memory_report = f"""=== Memory Report ===
Dense model weight size:  {d_size:,} bytes ({d_size/1024/1024:.2f} MB)
Sparse CSR file size:     {s_size:,} bytes ({s_size/1024/1024:.2f} MB)
Memory reduction ratio:   {ratio:.2f}x
Sparsity achieved:        {sparsity:.1f}%
Status: {status}
"""
    print(memory_report)
    with open(os.path.join(logs_dir, "memory_report.txt"), "w") as f:
        f.write(memory_report)

if __name__ == "__main__":
    main()
