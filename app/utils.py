import io
import base64
from PIL import Image
from typing import Tuple

def validate_image_size(file_bytes: bytes, max_mb: int) -> None:
    """Validate that the image file size is within the allowed limit."""
    if len(file_bytes) > max_mb * 1024 * 1024:
        raise ValueError(f"Image size exceeds the maximum allowed of {max_mb}MB")

def load_pil_from_bytes(file_bytes: bytes) -> Image.Image:
    """Load a PIL Image from bytes and convert to RGB."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def load_pil_from_base64(base64_str: str) -> Image.Image:
    """Load a PIL Image from base64 string."""
    try:
        file_bytes = base64.b64decode(base64_str)
        return load_pil_from_bytes(file_bytes)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")

def validate_image_format(img: Image.Image) -> None:
    """Validate that the image is in a supported format."""
    if img.mode not in ['RGB', 'RGBA', 'L']:
        raise ValueError(f"Unsupported image mode: {img.mode}. Supported modes: RGB, RGBA, L")
    
    if img.size[0] < 32 or img.size[1] < 32:
        raise ValueError("Image too small. Minimum size: 32x32 pixels")
    
    if img.size[0] > 4096 or img.size[1] > 4096:
        raise ValueError("Image too large. Maximum size: 4096x4096 pixels")