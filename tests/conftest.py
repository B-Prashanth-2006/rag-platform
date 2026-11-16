"""Root test configuration and environment setup."""

from __future__ import annotations

import os

# Prevent OpenMP runtime conflict on macOS when PyTorch and FAISS both load libomp
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
