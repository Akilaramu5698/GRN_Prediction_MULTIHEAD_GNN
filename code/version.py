# ============================================================================
# GET YOUR ACTUAL SYSTEM AND PACKAGE VERSIONS
# ============================================================================
import sys
import platform
import numpy as np
import pandas as pd
import torch
import torch_geometric
import sklearn
import networkx as nx
import scipy
import matplotlib
import seaborn as sns

print("\n" + "="*50)
print("YOUR ACTUAL SYSTEM & PACKAGE VERSIONS")
print("="*50)

# System info
print(f"\n  SYSTEM:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Machine: {platform.machine()}")

# Package versions
print(f"\n PACKAGES:")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  PyTorch: {torch.__version__}")
print(f"  PyTorch Geometric: {torch_geometric.__version__}")
print(f"  Scikit-learn: {sklearn.__version__}")
print(f"  NetworkX: {nx.__version__}")
print(f"  SciPy: {scipy.__version__}")
print(f"  Matplotlib: {matplotlib.__version__}")
print(f"  Seaborn: {sns.__version__}")

# CUDA info (if available)
if torch.cuda.is_available():
    print(f"\n CUDA:")
    print(f"  CUDA Available: Yes")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"\n CUDA: Not available (running on CPU)")

print("\n" + "="*50)
