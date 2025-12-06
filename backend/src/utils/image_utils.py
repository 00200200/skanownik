

import cv2
import numpy as np


def align_to_reference(
    image: np.ndarray, reference: np.ndarray | None = None
) -> np.ndarray:
    """Placeholder for Image Registration.

    Aligns the input image to a 'Golden Master' reference.

    Args:
        image: Input image (H, W, C) or (H, W)
        reference: Reference image (optional)

    Returns:
        Aligned image.
    """
    # TODO: Implement actual registration (e.g., ORB features + Homography)
    # For now, just return the image as is.
    return image


def is_background(
    patch: np.ndarray, threshold: float = 240, ratio: float = 0.9
) -> bool:
    """Check if the patch is mostly background (white).

    Args:
        patch: Image patch (H, W, C) or (H, W)
        threshold: Pixel value threshold to consider as 'white' (0-255).
        ratio: Fraction of pixels that must be > threshold to be considered background.

    Returns:
        True if patch is background, False otherwise.
    """
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = patch

    # Count pixels above threshold (assuming white background)
    white_pixels = np.sum(gray > threshold)
    total_pixels = gray.size

    return (white_pixels / total_pixels) > ratio


def tile_image(
    image: np.ndarray,
    patch_size: int = 224,
    stride: int = 112,
    filter_background: bool = True,
) -> list[tuple[np.ndarray, int, int]]:
    """Crop the large image into fixed-size patches with overlap.

    Args:
        image: Input image (H, W, C)
        patch_size: Size of the square patch.
        stride: Stride for sliding window.
        filter_background: Whether to drop background patches.

    Returns:
        List of tuples (patch, x, y).
    """
    h, w = image.shape[:2]
    patches = []

    # Handle case where image is smaller than patch
    if h < patch_size or w < patch_size:
        # Resize or pad? For now, let's just resize if too small, or skip.
        # Assuming high-res images as per requirements.
        return []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]

            if filter_background and is_background(patch):
                continue

            patches.append((patch, x, y))

    return patches


def stitch_heatmap(
    scores: list[float],
    coords: list[tuple[int, int]],
    image_shape: tuple[int, int],
    patch_size: int = 224,
) -> np.ndarray:
    """Stitch patch anomaly scores back into a full-sized heatmap.

    Args:
        scores: List of anomaly scores for each patch.
        coords: List of (x, y) coordinates for each patch.
        image_shape: (H, W) of the original image.
        patch_size: Size of the patches used.

    Returns:
        Heatmap (H, W) with float values.
    """
    heatmap = np.zeros(image_shape, dtype=np.float32)
    counts = np.zeros(image_shape, dtype=np.float32)

    for score, (x, y) in zip(scores, coords, strict=False):
        heatmap[y : y + patch_size, x : x + patch_size] += score
        counts[y : y + patch_size, x : x + patch_size] += 1

    # Average the scores in overlapping regions
    # Avoid division by zero
    counts[counts == 0] = 1
    heatmap = heatmap / counts

    # Apply Gaussian smoothing
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4.0, sigmaY=4.0)

    return heatmap
