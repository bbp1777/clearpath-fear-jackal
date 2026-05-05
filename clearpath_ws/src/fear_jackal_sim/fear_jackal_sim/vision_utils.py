"""
Shared RGB and depth utilities used by the trainer, goal monitor, and offline dataset
pipeline.
"""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


class ImageDecodingError(RuntimeError):
    """
    Raised when a ROS image uses an encoding this project does not currently support.
    """
    pass


def rgb_image_to_numpy(msg: Image) -> np.ndarray:
    """
    Decode a ROS color image into a standard uint8 numpy array.
    """
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ImageDecodingError(f'Unsupported color encoding: {msg.encoding}')

    # ROS image rows can include padding, so the step field is safer than assuming
    # width * 3 bytes exactly when reconstructing the array.
    channels = max(int(msg.step / msg.width), 3)
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)[..., :3]
    if msg.encoding == 'bgr8':
        image = image[..., ::-1]
    return image.copy()


def depth_image_to_numpy(msg: Image) -> np.ndarray:
    """
    Decode a ROS depth image into meters as float32.
    """
    if msg.encoding == '32FC1':
        dtype = np.float32
        width = int(msg.step / np.dtype(dtype).itemsize)
        image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, width)[:, :msg.width]
        return image.copy()

    if msg.encoding == '16UC1':
        dtype = np.uint16
        width = int(msg.step / np.dtype(dtype).itemsize)
        image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, width)[:, :msg.width]
        return image.astype(np.float32) / 1000.0

    raise ImageDecodingError(f'Unsupported depth encoding: {msg.encoding}')


def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Resize an image or depth map with nearest-neighbor sampling.
    """
    if image.shape[0] == height and image.shape[1] == width:
        return image.copy()

    y_index = np.linspace(0, image.shape[0] - 1, height).astype(np.int32)
    x_index = np.linspace(0, image.shape[1] - 1, width).astype(np.int32)
    return image[y_index][:, x_index].copy()


def format_rgbd_timestep(
    color_msg: Image,
    depth_msg: Image,
    height: int = 84,
    width: int = 84,
    depth_clip_m: float = 5.0,
) -> np.ndarray:
    """
    Merge one RGB frame and one depth frame into a single channel-first [4, H, W] tensor.
    """
    rgb = resize_nearest(rgb_image_to_numpy(color_msg), height, width)
    depth = resize_nearest(depth_image_to_numpy(depth_msg), height, width)
    depth = np.clip(depth, 0.0, depth_clip_m)
    depth = depth / max(depth_clip_m, 1.0e-6)
    rgb = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)
    depth = np.expand_dims((depth * 255.0).astype(np.uint8), axis=0)
    return np.concatenate((rgb, depth), axis=0)


def merge_rgb_depth_window(
    rgb_window: np.ndarray,
    depth_window: np.ndarray,
    depth_clip_m: float = 5.0,
) -> np.ndarray:
    """
    Merge one saved manual RGB window and one saved depth window into [look_back, 4, H, W].
    """
    rgb = np.asarray(rgb_window, dtype=np.uint8)
    depth = np.asarray(depth_window, dtype=np.float32)
    if rgb.ndim != 4:
        raise ValueError(f'Expected RGB window [look_back, 3, H, W], got {rgb.shape}.')
    if depth.ndim != 3:
        raise ValueError(f'Expected depth window [look_back, H, W], got {depth.shape}.')
    if rgb.shape[0] != depth.shape[0]:
        raise ValueError('RGB and depth windows must use the same look_back.')

    # The Jackal MANN path is intentionally hard-coded to 4 channels: RGB + 1 depth layer.
    depth = np.clip(depth, 0.0, depth_clip_m)
    depth = depth / max(depth_clip_m, 1.0e-6)
    depth = np.expand_dims((depth * 255.0).astype(np.uint8), axis=1)
    return np.concatenate((rgb, depth), axis=1)


def merge_rgb_depth_dataset(
    rgb_windows: np.ndarray,
    depth_windows: np.ndarray,
    depth_clip_m: float = 5.0,
) -> np.ndarray:
    """
    Merge the full manual Jackal dataset into the Sanchez-style [N, look_back, 4, H, W] format.
    """
    rgb = np.asarray(rgb_windows, dtype=np.uint8)
    depth = np.asarray(depth_windows, dtype=np.float32)
    if rgb.ndim != 5:
        raise ValueError(f'Expected RGB dataset [N, look_back, 3, H, W], got {rgb.shape}.')
    if depth.ndim != 4:
        raise ValueError(f'Expected depth dataset [N, look_back, H, W], got {depth.shape}.')
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError('RGB and depth datasets must agree on N and look_back.')

    depth = np.clip(depth, 0.0, depth_clip_m)
    depth = depth / max(depth_clip_m, 1.0e-6)
    depth = np.expand_dims((depth * 255.0).astype(np.uint8), axis=2)
    return np.concatenate((rgb, depth), axis=2)

def _green_goal_mask(
    rgb_image: np.ndarray,
    min_green: int = 120,
    max_red: int = 45,
    max_blue: int = 45,
    min_green_minus_red: int = 100,
    min_green_minus_blue: int = 100,
) -> np.ndarray:
    red = rgb_image[..., 0].astype(np.int16)
    green = rgb_image[..., 1].astype(np.int16)
    blue = rgb_image[..., 2].astype(np.int16)
    return (
        (green >= min_green)
        & (red <= max_red)
        & (blue <= max_blue)
        & ((green - red) >= min_green_minus_red)
        & ((green - blue) >= min_green_minus_blue)
    )


def compute_green_goal_coverage(
    rgb_image: np.ndarray,
    min_green: int = 120,
    max_red: int = 45,
    max_blue: int = 45,
    min_green_minus_red: int = 100,
    min_green_minus_blue: int = 100,
) -> float:
    """
    Measure how much of the frame is occupied by the goal-marker green mask.
    """
    mask = _green_goal_mask(
        rgb_image,
        min_green=min_green,
        max_red=max_red,
        max_blue=max_blue,
        min_green_minus_red=min_green_minus_red,
        min_green_minus_blue=min_green_minus_blue,
    )
    return float(mask.mean())


def compute_green_goal_offset(
    rgb_image: np.ndarray,
    min_green: int = 120,
    max_red: int = 45,
    max_blue: int = 45,
    min_green_minus_red: int = 100,
    min_green_minus_blue: int = 100,
) -> float:
    """
    Measure how far left or right the green goal centroid sits from image center.
    """
    mask = _green_goal_mask(
        rgb_image,
        min_green=min_green,
        max_red=max_red,
        max_blue=max_blue,
        min_green_minus_red=min_green_minus_red,
        min_green_minus_blue=min_green_minus_blue,
    )
    _, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0

    image_center = (rgb_image.shape[1] - 1) / 2.0
    if image_center <= 0.0:
        return 0.0

    centroid_x = float(xs.mean())
    return float(np.clip((centroid_x - image_center) / image_center, -1.0, 1.0))


def summarize_depth_sectors(depth_image: np.ndarray, max_depth_m: float = 3.0) -> dict[str, float]:
    """
    Reduce a depth image to coarse left, center, right, and minimum-distance summaries.
    """
    default_summary = {
        'left': 1.0,
        'center': 1.0,
        'right': 1.0,
        'minimum': 1.0,
    }
    if depth_image.ndim != 2:
        return default_summary

    height, width = depth_image.shape
    if height == 0 or width == 0:
        return default_summary

    y_start = int(height * 0.30)
    y_end = int(height * 0.85)
    focus = depth_image[y_start:y_end, :]
    sections = np.array_split(focus, 3, axis=1)

    def normalize_region(region: np.ndarray) -> float:
        valid = region[np.isfinite(region)]
        valid = valid[valid > 0.0]
        if valid.size == 0:
            return 1.0
        percentile = float(np.percentile(valid, 15))
        clipped = np.clip(percentile, 0.0, max_depth_m)
        return float(clipped / max_depth_m)

    left = normalize_region(sections[0])
    center = normalize_region(sections[1])
    right = normalize_region(sections[2])

    valid_all = focus[np.isfinite(focus)]
    valid_all = valid_all[valid_all > 0.0]
    if valid_all.size == 0:
        minimum = 1.0
    else:
        minimum = float(np.clip(valid_all.min(), 0.0, max_depth_m) / max_depth_m)

    return {
        'left': left,
        'center': center,
        'right': right,
        'minimum': minimum,
    }


def compute_collision_from_depth(
    depth_image: np.ndarray,
    collision_distance_m: float,
    center_height_fraction: float = 0.45,
    center_width_fraction: float = 0.40,
    occupancy_threshold: float = 0.02,
) -> bool:
    """
    Declare a near-collision when enough center pixels fall within the collision distance.
    """
    height, width = depth_image.shape[:2]
    y_margin = int(height * (1.0 - center_height_fraction) / 2.0)
    x_margin = int(width * (1.0 - center_width_fraction) / 2.0)
    center_window = depth_image[y_margin:height - y_margin, x_margin:width - x_margin]

    valid = center_window[np.isfinite(center_window)]
    valid = valid[valid > 0.0]
    if valid.size == 0:
        return False

    close_fraction = float((valid <= collision_distance_m).mean())
    return bool(close_fraction >= occupancy_threshold)
