"""Hydrointelligence System, Part A.

Coupled 1-D/2-D shallow-water flood engine (ground truth) and the GeoKAN-PINO
surrogate trained on it. Results on synthetic terrain demonstrate numerics only;
they are not flood predictions for any real place.
"""
import os

# cuBLAS needs this before the CUDA context is created for deterministic kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
SYNTHETIC_WATERMARK = "SYNTHETIC TERRAIN — NOT A PREDICTION FOR REAL ROXAS CITY"
