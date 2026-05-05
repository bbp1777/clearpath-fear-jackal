# -*- coding: utf-8 -*-
"""
Utilities for converting Jackal RGB and depth observations into the channel-first
arrays expected by the MANN code.
"""

import cv2
import numpy as np


class ImageDecodingError(RuntimeError):
    pass



def rgb_image_to_numpy(msg):
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ImageDecodingError(f"Unsupported color encoding: {msg.encoding}")

    channels = max(int(msg.step / msg.width), 3)
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)[..., :3]
    if msg.encoding == 'bgr8':
        image = image[..., ::-1]
    return image.copy()



def depth_image_to_numpy(msg, depth_clip=5.0):
    if msg.encoding == '32FC1':
        dtype = np.float32
        width = int(msg.step / np.dtype(dtype).itemsize)
        image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, width)[:, :msg.width]
        depth = image.copy()
    elif msg.encoding == '16UC1':
        dtype = np.uint16
        width = int(msg.step / np.dtype(dtype).itemsize)
        image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, width)[:, :msg.width]
        depth = image.astype(np.float32) / 1000.0
    else:
        raise ImageDecodingError(f"Unsupported depth encoding: {msg.encoding}")

    depth = np.clip(depth, 0.0, depth_clip)
    depth = depth / max(depth_clip, 1.0e-6)
    return depth



def format_rgbd_observation(color_msg, depth_msg, shape_change=(100, 100), depth_clip=5.0):
    # JACKAL RGB-D SHAPE NOTE:
    # This function formats one timestep only and returns [4, H, W]. Dataset
    # collection should still stack look_back timesteps outside this helper so
    # each memory sample becomes [look_back, 4, H, W].
    #
    # JACKAL DATASET COLLECTION BLOCK:
    # Once the ROS subscriber has one RGB message and one depth message for a
    # single timestep, the dataset code can do:
    #
    timestep = format_rgbd_observation(color_msg, depth_msg, shape_change=(100, 100))
    dataset_keeper.append_current(timestep)
    #
    # After self.look_back timesteps have been collected, DatasetKeeper can
    # stack them into one sample shaped [look_back, 4, H, W].
    # Keep the output channel-first so the existing MANN path only needs a 3->4
    # channel change instead of a larger structural rewrite.
    rgb = rgb_image_to_numpy(color_msg)
    depth = depth_image_to_numpy(depth_msg, depth_clip=depth_clip)

    rgb = cv2.resize(rgb, shape_change, interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, shape_change, interpolation=cv2.INTER_NEAREST)

    rgb = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)
    depth = np.expand_dims((depth * 255.0).astype(np.uint8), axis=0)
    return np.concatenate((rgb, depth), axis=0)
