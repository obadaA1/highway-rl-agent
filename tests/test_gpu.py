"""
GPU diagnostic script.
Run: python tests/test_gpu.py
"""

import sys

print("\n" + "="*70)
print("GPU DIAGNOSTIC REPORT")
print("="*70)

# Test 1: Check if torch is installed
print("\n[1] PyTorch Installation")
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch not installed: {e}")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n[2] CUDA Availability")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if not cuda_available:
    print("❌ CUDA not available!")
    print("\nPossible reasons:")
    print("  1. PyTorch CPU-only version installed")
    print("  2. CUDA drivers not installed")
    print("  3. CUDA version mismatch")
    
    # Check PyTorch build info
    print(f"\nPyTorch build: {torch.version.cuda}")
    print("If 'None', you have CPU-only PyTorch installed!")
else:
    print("✅ CUDA is available!")

# Test 3: GPU details
if cuda_available:
    print("\n[3] GPU Information")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")

# Test 4: Simple CUDA operation
if cuda_available:
    print("\n[4] CUDA Operation Test")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x @ y
        print("✅ Successfully performed matrix multiplication on GPU")
    except Exception as e:
        print(f"❌ CUDA operation failed: {e}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

if not cuda_available:
    print("\n⚠️  ACTION REQUIRED: Install PyTorch with CUDA support")
    print("\nFor RTX 3050 (CUDA 11.8 or 12.1):")
    print("\nRun this command:")
    print("pip uninstall torch")
    print("pip install torch --index-url https://download.pytorch.org/whl/cu118")
else:
    print("\n✅ GPU setup is correct!")