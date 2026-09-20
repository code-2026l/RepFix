"""
RepFix: Inference and Evaluation Demo

This script loads a pretrained model and runs inference on synthetic data.
For full training, see train.py (requires the complete backend pipeline).

Usage:
    python eval.py --model ../models/model_fold1.pth --size base
"""

import argparse, sys, warnings
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import StudentModel

def load_model(model_path: str, size: str = "base", device: str = "cpu"):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = StudentModel(size=size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def prepare_data(batch_size: int, seq_len: int = 60, n_feat: int = 28, device: str = "cpu"):
    rng = np.random.RandomState(42)
    X = torch.from_numpy(rng.randn(batch_size, seq_len, n_feat).astype(np.float32)).to(device)
    aux = torch.from_numpy(rng.randn(batch_size, 16).astype(np.float32)).to(device)
    return X, aux

def main():
    parser = argparse.ArgumentParser(description="RepFix Evaluation")
    parser.add_argument("--model", type=str, default="../models/model_fold1.pth")
    parser.add_argument("--size", type=str, default="base", choices=["tiny","base","large","xl","xxl"])
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=10, help="Number of inference iterations")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[RepFix] Loading {args.size} model from {args.model}")
    model = load_model(args.model, args.size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[RepFix] Parameters: {n_params:,}")

    X, aux = prepare_data(args.batch, device=device)
    print(f"[RepFix] Input: X {list(X.shape)}, aux {list(aux.shape)}")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(X, aux)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(args.iters):
                _ = model(X, aux)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iters
        throughput = args.batch * 1000 / ms
        print(f"[RepFix] Latency: {ms:.1f}ms/batch | Throughput: {throughput:.0f} samples/sec")
    else:
        import time
        t0 = time.time()
        with torch.no_grad():
            for _ in range(args.iters):
                _ = model(X, aux)
        ms = (time.time() - t0) * 1000 / args.iters
        print(f"[RepFix] Latency: {ms:.1f}ms/batch (CPU)")

    # Output shapes
    out = model(X, aux)
    print(f"[RepFix] Output keys: {list(out.keys())}")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")

if __name__ == "__main__":
    main()
