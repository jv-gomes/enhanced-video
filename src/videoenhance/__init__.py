"""Upscale and interpolate video on AMD GPUs via NCNN + Vulkan binaries.

The package is an orchestrator only: it shells out to FFmpeg,
``realesrgan-ncnn-vulkan`` and ``rife-ncnn-vulkan``. It deliberately carries no
PyTorch or CUDA dependency.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
