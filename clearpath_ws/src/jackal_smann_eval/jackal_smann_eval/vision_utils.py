# This imports numpy for image math and array reshaping.
import numpy as np
# This imports the ROS image type used by the helper functions.
from sensor_msgs.msg import Image


# This marks unsupported ROS image encodings with a clear error.
class ImageDecodingError(RuntimeError):
    """Raise a readable error when an image encoding is not supported."""


# This converts a ROS color image into a uint8 RGB array.
def rgb_image_to_numpy(msg: Image) -> np.ndarray:
    """Decode a ROS color message into a normal RGB numpy image."""

    # This rejects unsupported color encodings.
    if msg.encoding not in ('rgb8', 'bgr8'):
        # This raises a readable error for unsupported color data.
        raise ImageDecodingError(f'Unsupported color encoding: {msg.encoding}')

    # This computes the stored channel width from the ROS step field.
    channels = max(int(msg.step / msg.width), 3)
    # This reshapes the raw bytes into an image array.
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)[..., :3]

    # This flips BGR into RGB when needed.
    if msg.encoding == 'bgr8':
        # This reverses the channel order.
        image = image[..., ::-1]

    # This returns a standalone copy of the RGB image.
    return image.copy()


# This converts a ROS depth image into meters as float32.
def depth_image_to_numpy(msg: Image) -> np.ndarray:
    """Decode a ROS depth message into a float32 depth map in meters."""

    # This handles float depth images directly.
    if msg.encoding == '32FC1':
        # This sets the dtype used by the message.
        dtype = np.float32
        # This computes the padded row width.
        width = int(msg.step / np.dtype(dtype).itemsize)
        # This reshapes the raw bytes into a 2D depth image.
        image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, width)[:, :msg.width]
        # This returns a standalone copy of the depth image.
        return image.copy()

    # This handles uint16 depth images stored in millimeters.
    if msg.encoding == '16UC1':
        # This sets the dtype used by the message.
        dtype = np.uint16
        # This computes the padded row width.
        width = int(msg.step / np.dtype(dtype).itemsize)
        # This reshapes the raw bytes into a 2D depth image.
        image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, width)[:, :msg.width]
        # This converts millimeters into meters.
        return image.astype(np.float32) / 1000.0

    # This raises a readable error for unsupported depth data.
    raise ImageDecodingError(f'Unsupported depth encoding: {msg.encoding}')


# This resizes an image with nearest-neighbor sampling.
def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a color image or depth map with simple nearest-neighbor sampling."""

    # This returns early when the image already has the target size.
    if image.shape[0] == height and image.shape[1] == width:
        # This returns a standalone copy of the input.
        return image.copy()

    # This builds the target y indexes.
    y_index = np.linspace(0, image.shape[0] - 1, height).astype(np.int32)
    # This builds the target x indexes.
    x_index = np.linspace(0, image.shape[1] - 1, width).astype(np.int32)
    # This samples the resized image.
    return image[y_index][:, x_index].copy()


# This merges one RGB message and one depth message into one [4, H, W] tensor.
def format_rgbd_timestep(color_msg: Image, depth_msg: Image, height: int = 84, width: int = 84, depth_clip_m: float = 5.0) -> np.ndarray:
    """Convert one live Jackal RGB-D pair into the SMANN input layout."""

    # This decodes and resizes the RGB image.
    rgb = resize_nearest(rgb_image_to_numpy(color_msg), height, width)
    # This decodes and resizes the depth image.
    depth = resize_nearest(depth_image_to_numpy(depth_msg), height, width)
    # This clips very far depth values.
    depth = np.clip(depth, 0.0, depth_clip_m)
    # This normalizes depth into the 0 to 1 range.
    depth = depth / max(depth_clip_m, 1.0e-6)
    # This converts RGB into channel-first layout.
    rgb = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)
    # This converts depth into one uint8 channel.
    depth = np.expand_dims((depth * 255.0).astype(np.uint8), axis=0)
    # This concatenates RGB and depth into 4 channels.
    return np.concatenate((rgb, depth), axis=0)


# This measures how much of the frame matches the green goal color.
def compute_green_goal_coverage(rgb_image: np.ndarray, min_green: int = 160, max_red: int = 140, max_blue: int = 140) -> float:
    """Measure the fraction of pixels that match the green goal mask."""

    # This extracts the red channel.
    red = rgb_image[..., 0]
    # This extracts the green channel.
    green = rgb_image[..., 1]
    # This extracts the blue channel.
    blue = rgb_image[..., 2]
    # This builds the green goal mask.
    mask = (green >= min_green) & (red <= max_red) & (blue <= max_blue)
    # This returns the fraction of goal-colored pixels.
    return float(mask.mean())


# This measures whether the goal blob is left or right of center.
def compute_green_goal_offset(rgb_image: np.ndarray, min_green: int = 160, max_red: int = 140, max_blue: int = 140) -> float:
    """Measure the left-right offset of the green goal mask centroid."""

    # This extracts the red channel.
    red = rgb_image[..., 0]
    # This extracts the green channel.
    green = rgb_image[..., 1]
    # This extracts the blue channel.
    blue = rgb_image[..., 2]
    # This builds the green goal mask.
    mask = (green >= min_green) & (red <= max_red) & (blue <= max_blue)
    # This finds the goal pixel coordinates.
    _, xs = np.nonzero(mask)

    # This returns zero when the goal is not visible.
    if xs.size == 0:
        # This returns a centered offset when no goal is found.
        return 0.0

    # This computes the image center position.
    image_center = (rgb_image.shape[1] - 1) / 2.0

    # This returns zero when the width is invalid.
    if image_center <= 0.0:
        # This returns a safe default offset.
        return 0.0

    # This computes the goal centroid x position.
    centroid_x = float(xs.mean())
    # This returns the normalized left-right offset.
    return float(np.clip((centroid_x - image_center) / image_center, -1.0, 1.0))


# This reduces a depth image into coarse obstacle sectors.
def summarize_depth_sectors(depth_image: np.ndarray, max_depth_m: float = 3.0) -> dict[str, float]:
    """Summarize depth into left, center, right, and minimum normalized distances."""

    # This builds the default safe summary.
    summary = {'left': 1.0, 'center': 1.0, 'right': 1.0, 'minimum': 1.0}

    # This returns the default summary when the depth image is invalid.
    if depth_image.ndim != 2:
        # This returns the default summary.
        return summary

    # This reads the image size.
    height, width = depth_image.shape

    # This returns the default summary for empty images.
    if height == 0 or width == 0:
        # This returns the default summary.
        return summary

    # This crops to the forward-looking part of the image.
    focus = depth_image[int(height * 0.30):int(height * 0.85), :]
    # This splits the focus image into three horizontal sectors.
    sections = np.array_split(focus, 3, axis=1)

    # This normalizes one region into the 0 to 1 range.
    def normalize(region: np.ndarray) -> float:
        """Convert one depth region into one robust normalized distance."""

        # This keeps only valid finite depth values.
        valid = region[np.isfinite(region)]
        # This removes zero or negative depth values.
        valid = valid[valid > 0.0]

        # This returns a safe default when no valid depth exists.
        if valid.size == 0:
            # This returns the safest normalized value.
            return 1.0

        # This picks a robust near-obstacle statistic.
        percentile = float(np.percentile(valid, 15))
        # This clips the distance to the chosen max depth.
        clipped = np.clip(percentile, 0.0, max_depth_m)
        # This normalizes the distance.
        return float(clipped / max_depth_m)

    # This fills the left summary.
    summary['left'] = normalize(sections[0])
    # This fills the center summary.
    summary['center'] = normalize(sections[1])
    # This fills the right summary.
    summary['right'] = normalize(sections[2])

    # This keeps only valid depth values from the focus window.
    valid_all = focus[np.isfinite(focus)]
    # This removes zero or negative depth values.
    valid_all = valid_all[valid_all > 0.0]

    # This sets the minimum distance summary.
    if valid_all.size == 0:
        # This stores the safest normalized value.
        summary['minimum'] = 1.0
    else:
        # This stores the normalized minimum distance.
        summary['minimum'] = float(np.clip(valid_all.min(), 0.0, max_depth_m) / max_depth_m)

    # This returns the final sector summary.
    return summary


# This detects center-window collisions from depth alone.
def compute_collision_from_depth(depth_image: np.ndarray, collision_distance_m: float, occupancy_threshold: float = 0.02) -> bool:
    """Detect a near collision when enough center pixels are closer than the threshold."""

    # This reads the image size.
    height, width = depth_image.shape[:2]
    # This computes the vertical margin for the center crop.
    y_margin = int(height * 0.275)
    # This computes the horizontal margin for the center crop.
    x_margin = int(width * 0.30)
    # This extracts the center depth window.
    center_window = depth_image[y_margin:height - y_margin, x_margin:width - x_margin]
    # This keeps only finite depth values.
    valid = center_window[np.isfinite(center_window)]
    # This removes zero or negative depth values.
    valid = valid[valid > 0.0]

    # This returns no collision when no valid depth exists.
    if valid.size == 0:
        # This returns the safe default.
        return False

    # This measures the fraction of close pixels.
    close_fraction = float((valid <= collision_distance_m).mean())
    # This returns whether the close fraction exceeds the threshold.
    return bool(close_fraction >= occupancy_threshold)
