import os
import torch
import numpy as np
import scipy.sparse as sp
import struct
import json

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    
    state_dict = torch.load(os.path.join(outputs_dir, "pruned_model.pth"), map_location='cpu')
    
    # Layer order
    layer_names = ["fc1", "fc2", "fc3"]
    layers_data = []
    
    total_dense_size = 0
    total_sparse_size = 0
    total_params = 0
    total_zeros = 0
    
    layer_sparsities = []
    
    # Size of header
    total_sparse_size += 4 + 4 + 4 # magic, version, num_layers
    
    for name in layer_names:
        w = state_dict[name + ".weight"].numpy()
        b = state_dict[name + ".bias"].numpy()
        
        w[np.abs(w) < 1e-6] = 0.0
        
        zeros = np.sum(w == 0)
        params = w.size
        total_params += params
        total_zeros += zeros
        layer_sparsities.append(float(zeros / params))
        
        csr = sp.csr_matrix(w)
        num_rows, num_cols = csr.shape
        nnz = csr.nnz
        
        layers_data.append({
            "num_rows": num_rows,
            "num_cols": num_cols,
            "nnz": nnz,
            "values": csr.data.astype(np.float32),
            "col_indices": csr.indices.astype(np.int32),
            "row_ptr": csr.indptr.astype(np.int32),
            "bias": b.astype(np.float32)
        })
        
        # size logic
        total_dense_size += (params + b.size) * 4
        # num_rows, num_cols, nnz (4x3) + values + col_indices + row_ptr + bias
        total_sparse_size += 12 + nnz * 4 + nnz * 4 + (num_rows + 1) * 4 + num_rows * 4

    overall_sparsity = float(total_zeros / total_params)
    
    # Write binary
    bin_path = os.path.join(outputs_dir, "sparse_weights.bin")
    with open(bin_path, "wb") as f:
        f.write(b'VCSR')
        f.write(struct.pack('<I', 1)) # version
        f.write(struct.pack('<I', len(layers_data))) # num_layers
        
        for ld in layers_data:
            f.write(struct.pack('<III', ld['num_rows'], ld['num_cols'], ld['nnz']))
            f.write(ld['values'].tobytes())
            f.write(ld['col_indices'].tobytes())
            f.write(ld['row_ptr'].tobytes())
            f.write(ld['bias'].tobytes())
            
    reduction_ratio = total_dense_size / total_sparse_size
    
    meta = {
        "architecture": [784, 256, 128, 10],
        "sparsity_per_layer": layer_sparsities,
        "overall_sparsity": overall_sparsity,
        "dense_size_bytes": total_dense_size,
        "sparse_size_bytes": total_sparse_size,
        "memory_reduction_ratio": reduction_ratio,
        "baseline_accuracy": 0.0,
        "pruned_accuracy": 0.0
    }
    
    # Try to extract accuracy from pruning logs
    logs_dir = os.path.join(base_dir, "logs")
    if os.path.exists(os.path.join(logs_dir, "training_log.txt")):
        with open(os.path.join(logs_dir, "training_log.txt")) as f:
            for line in f:
                if "Baseline accuracy" in line:
                    try:
                        meta["baseline_accuracy"] = float(line.split(":")[1].strip().replace("%", ""))
                    except:
                        pass
    if os.path.exists(os.path.join(logs_dir, "pruning_log.txt")):
        with open(os.path.join(logs_dir, "pruning_log.txt")) as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Accuracy:" in line:
                    try:
                        meta["pruned_accuracy"] = float(line.split(":")[1].strip().replace("%", ""))
                        break
                    except:
                        pass

    with open(os.path.join(outputs_dir, "sparse_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"Dense model size: {total_dense_size} bytes")
    print(f"Sparse file size: {total_sparse_size} bytes")
    print(f"Memory reduction ratio: {reduction_ratio:.2f}x")
    print(f"Per-layer sparsity: {layer_sparsities}")
    print(f"Overall sparsity: {overall_sparsity:.4f}")

if __name__ == "__main__":
    main()
