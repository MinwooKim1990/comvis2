#!/usr/bin/env python3
"""
CUDA Environment Diagnostic Tool
Check CUDA availability and compatibility for RTX 4090
"""

import sys
import subprocess

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_nvidia_smi():
    """Check nvidia-smi output"""
    print_section("1. NVIDIA GPU Detection (nvidia-smi)")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi works!")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi failed")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found")
        print("NVIDIA drivers may not be installed")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_cuda_toolkit():
    """Check CUDA toolkit installation"""
    print_section("2. CUDA Toolkit Detection")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA toolkit installed!")
            print(result.stdout)
            return True
        else:
            print("✗ nvcc not found")
            return False
    except FileNotFoundError:
        print("✗ CUDA toolkit (nvcc) not found")
        print("Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_numba_cuda():
    """Check Numba CUDA"""
    print_section("3. Numba CUDA Detection")
    try:
        from numba import cuda
        print("✓ Numba imported successfully")

        if cuda.is_available():
            print("✓ Numba CUDA is available!")
            try:
                cuda.select_device(0)
                device = cuda.get_current_device()
                print(f"  Device name: {device.name.decode()}")
                print(f"  Compute capability: {device.compute_capability}")
                print(f"  PCI bus ID: {device.id}")
                return True
            except Exception as e:
                print(f"✗ CUDA device access failed: {e}")
                return False
        else:
            print("✗ Numba CUDA is NOT available")
            print("\nPossible reasons:")
            print("1. CUDA_PATH environment variable not set")
            print("2. Numba can't find CUDA libraries")
            print("3. CUDA version incompatible with Numba")
            return False
    except ImportError as e:
        print(f"✗ Cannot import numba: {e}")
        print("Install: pip install numba")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_cupy():
    """Check CuPy"""
    print_section("4. CuPy Detection (Alternative)")
    try:
        import cupy as cp
        print("✓ CuPy imported successfully")

        # Test basic operation
        arr = cp.array([1, 2, 3])
        print(f"✓ CuPy GPU operations work!")
        print(f"  CuPy version: {cp.__version__}")
        print(f"  CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        return True
    except ImportError:
        print("✗ CuPy not installed")
        print("Install: pip install cupy-cuda12x  (or cupy-cuda11x)")
        return False
    except Exception as e:
        print(f"✗ CuPy test failed: {e}")
        return False

def check_pytorch():
    """Check PyTorch CUDA"""
    print_section("5. PyTorch CUDA Detection (Alternative)")
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print("✓ PyTorch CUDA is available!")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")

            # Test basic operation
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print("✓ PyTorch GPU operations work!")
            return True
        else:
            print("✗ PyTorch CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        print("Install: pip install torch")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def print_recommendations(results):
    """Print recommendations based on results"""
    print_section("RECOMMENDATIONS")

    gpu_ok = results['nvidia_smi']
    toolkit_ok = results['cuda_toolkit']
    numba_ok = results['numba']
    cupy_ok = results['cupy']
    pytorch_ok = results['pytorch']

    if not gpu_ok:
        print("❌ CRITICAL: No NVIDIA GPU detected")
        print("   → Install NVIDIA drivers")
        print("   → Restart computer")
        return

    if not toolkit_ok:
        print("⚠️  WARNING: CUDA toolkit not found")
        print("   → Install CUDA Toolkit 12.x for RTX 4090")
        print("   → Add to PATH: /usr/local/cuda/bin")
        print("   → Set CUDA_HOME=/usr/local/cuda")

    if numba_ok:
        print("✅ Numba CUDA works - use raytracing_cuda.py")
    elif cupy_ok:
        print("✅ CuPy works - use CuPy version (recommended)")
        print("   → I'll create a CuPy-based raytracer for you")
    elif pytorch_ok:
        print("✅ PyTorch CUDA works - use PyTorch version")
        print("   → I can create a PyTorch-based raytracer")
    else:
        print("❌ No GPU frameworks work")
        print("\nTROUBLESHOOTING STEPS:")
        print("1. Ubuntu: Check CUDA_PATH")
        print("   export CUDA_PATH=/usr/local/cuda")
        print("   export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH")
        print("   export PATH=$CUDA_PATH/bin:$PATH")
        print("")
        print("2. Windows: Check Environment Variables")
        print("   CUDA_PATH = C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x")
        print("   Add to PATH: %CUDA_PATH%\\bin")
        print("")
        print("3. Reinstall CUDA-capable package")
        print("   pip uninstall numba")
        print("   pip install numba")
        print("   OR")
        print("   pip install cupy-cuda12x  # For CUDA 12.x")
        print("")
        print("4. Use CPU version for now:")
        print("   python raytracing_fast.py")

def main():
    print("="*60)
    print("  CUDA Diagnostic Tool for RTX 4090")
    print("="*60)

    results = {
        'nvidia_smi': check_nvidia_smi(),
        'cuda_toolkit': check_cuda_toolkit(),
        'numba': check_numba_cuda(),
        'cupy': check_cupy(),
        'pytorch': check_pytorch()
    }

    print_recommendations(results)

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name:20s}: {'WORKING' if status else 'NOT WORKING'}")
    print("="*60)

if __name__ == "__main__":
    main()
