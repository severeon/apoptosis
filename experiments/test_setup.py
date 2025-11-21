#!/usr/bin/env python3
"""Quick test to verify setup and basic functionality."""

import sys
import torch
import numpy as np

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        import matplotlib.pyplot as plt
        from torch.utils.tensorboard import SummaryWriter
        from tqdm import tqdm
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return False

def test_device():
    """Check available compute device."""
    print("\nTesting device...")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using device: CUDA")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Using device: MPS (Apple Silicon GPU)")
        print(f"  Expected speedup: 5-10x over CPU on M2 Max")
    else:
        device = torch.device('cpu')
        print(f"✓ Using device: CPU")
        print("  Note: GPU not available, training will be slower")

    return True

def test_model_creation():
    """Test creating a small model."""
    print("\nTesting model creation...")
    try:
        # Simple test model
        class TinyTransformer(torch.nn.Module):
            def __init__(self, vocab_size=100, d_model=64):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab_size, d_model)
                self.proj = torch.nn.Linear(d_model, vocab_size)

            def forward(self, x):
                return self.proj(self.embed(x))

        model = TinyTransformer()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Test forward pass
        test_input = torch.randint(0, 100, (2, 10)).to(device)
        output = model(test_input)

        assert output.shape == (2, 10, 100), "Output shape incorrect"
        print(f"✓ Model creation successful")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_gradient_flow():
    """Test that gradients flow correctly."""
    print("\nTesting gradient flow...")
    try:
        # Simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )

        # Forward pass
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = all(p.grad is not None for p in model.parameters())
        assert has_gradients, "Some parameters don't have gradients"

        print("✓ Gradient flow working")
        return True
    except Exception as e:
        print(f"✗ Gradient test failed: {e}")
        return False

def test_apoptosis_concept():
    """Test the basic apoptosis concept."""
    print("\nTesting apoptosis mechanics...")
    try:
        # Simulate layer vitality decay
        vitality = 1.0
        max_lifespan = 100

        for age in range(max_lifespan + 1):
            vitality = max(0.0, 1.0 - (age / max_lifespan))

        assert vitality == 0.0, f"Vitality should decay to zero, got {vitality}"
        print("✓ Vitality decay working")

        # Simulate layer growth
        vitality = 0.0
        maturation_period = 50

        for age in range(maturation_period):
            vitality = min(1.0, age / maturation_period)

        assert vitality < 1.0, "Vitality should grow"
        print("✓ Vitality growth working")

        return True
    except Exception as e:
        print(f"✗ Apoptosis test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("APOPTOSIS EXPERIMENT - SETUP TEST")
    print("="*60)

    tests = [
        test_imports,
        test_device,
        test_model_creation,
        test_gradient_flow,
        test_apoptosis_concept
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append(False)

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} tests passed!")
        print("\nYou're ready to run the experiment:")
        print("  jupyter notebook apoptosis_experiment.ipynb")
        return 0
    else:
        print(f"✗ {total - passed}/{total} tests failed")
        print("\nPlease fix the issues above before running the experiment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
