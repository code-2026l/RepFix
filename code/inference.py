"""
RepFix Inference Example
========================
Minimal example demonstrating how to load a pretrained model and run inference.

Usage:
python inference.py --model-path../models/model_fold1.pth --size base
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import StudentModel


def load_model(model_path: str, size: str = "tiny", device: str = "cpu"):
    """Load a pretrained V11_6Student model.

    Args:
    model_path: Path to checkpoint (.pth file)
    size: Model size - "tiny"(94K), "base"(3M), "large"(10M), "xl"(22M), "xxl"(60M)
    device: "cpu" or "cuda"

    Returns:
    Loaded model in eval mode.
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = StudentModel(size=size).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def prepare_dummy_input(batch_size: int = 1, device: str = "cpu"):
    """Create a dummy input tensor simulating stock features.

    The model expects:
    X: (batch, 60, 28) - 60 time steps, 28 features per step
    aux: (batch, 16) - auxiliary features

    Returns:
    X, aux tensors on the specified device.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((batch_size, 60, 28)).astype(np.float32)
    aux = rng.standard_normal((batch_size, 16)).astype(np.float32)
    return torch.from_numpy(X).to(device), torch.from_numpy(aux).to(device)


def main():
    parser = argparse.ArgumentParser(description="RepFix Inference Demo")
    parser.add_argument("--model-path", type=str, default="../models/model_fold1.pth",
        help="Path to model checkpoint")
    parser.add_argument("--size", type=str, default="tiny", choices=["tiny", "base", "large", "xl", "xxl"],
        help="Model size variant")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[RepFix] Loading {args.size} model from {args.model_path} on {device}...")

    model = load_model(args.model_path, size=args.size, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[RepFix] Model loaded: {n_params:,} parameters")

    X, aux = prepare_dummy_input(batch_size=args.batch, device=device)
    print(f"[RepFix] Input shape: X={X.shape}, aux={aux.shape}")

    with torch.no_grad():
        output = model(X, aux)

        print(f"[RepFix] Output keys: {list(output.keys())}")
        if "point" in output:
            print(f"[RepFix] Point prediction shape: {output['point'].shape}")
            print(f"[RepFix] Point prediction (first 8): {output['point'][0, :8].tolist()}")
            if "quantiles" in output:
                print(f"[RepFix] Quantiles shape: {output['quantiles'].shape}")
                print("[RepFix] Inference complete!")


if __name__ == "__main__":
    main()
