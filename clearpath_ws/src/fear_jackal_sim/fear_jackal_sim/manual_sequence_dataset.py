"""
Disk format helpers for manual three-step RGB-D samples collected for low-shot offline fear
training.
"""
from __future__ import annotations

import json
import os
import time
from glob import glob

import numpy as np

UNSAFE_LABEL = 'unsafe'
SAFE_LABEL = 'safe'
DEFAULT_UNSAFE_REWARD = -1.0
DEFAULT_SAFE_REWARD = 0.0


def default_reward_for_label(label: str) -> float:
    # Unsafe memories default to -1.0 so the live agent can scale fear directly by
    # similarity to those terminal three-step sequences.
    """
    Return the default stored reward for a manual sample label.
    """
    normalized = str(label).strip().lower()
    if normalized == UNSAFE_LABEL:
        return DEFAULT_UNSAFE_REWARD
    if normalized == SAFE_LABEL:
        return DEFAULT_SAFE_REWARD
    raise ValueError(f'Unsupported label: {label}')


def save_manual_sequence_sample(
    output_dir: str,
    rgb_window: np.ndarray,
    depth_window: np.ndarray,
    label: str,
    reward: float | None = None,
    note: str = '',
    source: str = 'manual_capture',
) -> dict[str, str | float | int]:
    """
    Validate and persist one manual three-step sample to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    normalized_label = str(label).strip().lower()
    sample_reward = default_reward_for_label(normalized_label) if reward is None else float(reward)

    rgb = np.asarray(rgb_window, dtype=np.uint8)
    depth = np.asarray(depth_window, dtype=np.float32)
    if rgb.ndim != 4:
        raise ValueError(f'Expected RGB window shape [lookback, 3, H, W], got {rgb.shape}.')
    if depth.ndim != 3:
        raise ValueError(f'Expected depth window shape [lookback, H, W], got {depth.shape}.')
    if rgb.shape[0] != depth.shape[0]:
        raise ValueError('RGB and depth windows must use the same lookback length.')

    existing = sorted(glob(os.path.join(output_dir, 'sample_*.npz')))
    sample_index = len(existing) + 1
    sample_path = os.path.join(output_dir, f'sample_{sample_index:06d}.npz')
    timestamp = time.time()

    np.savez_compressed(
        sample_path,
        rgb=rgb,
        depth=depth,
        label=np.asarray(normalized_label),
        reward=np.asarray(sample_reward, dtype=np.float32),
        lookback=np.asarray(rgb.shape[0], dtype=np.int32),
        source=np.asarray(str(source)),
        note=np.asarray(str(note)),
        timestamp=np.asarray(timestamp, dtype=np.float64),
    )

    metadata = {
        'path': sample_path,
        'label': normalized_label,
        'reward': float(sample_reward),
        'lookback': int(rgb.shape[0]),
        'image_height': int(rgb.shape[-2]),
        'image_width': int(rgb.shape[-1]),
        'note': str(note),
    }
    metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w', encoding='ascii') as handle:
        json.dump(summarize_manual_sequence_dataset(output_dir), handle, indent=2)
    metadata['metadata_path'] = metadata_path
    return metadata


def iter_manual_sequence_samples(dataset_dir: str) -> list[str]:
    """
    List saved manual sample files in order.
    """
    return sorted(glob(os.path.join(dataset_dir, 'sample_*.npz')))


def load_manual_sequence_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """
    Load all saved manual samples into stacked numpy arrays.
    """
    sample_paths = iter_manual_sequence_samples(dataset_dir)
    stats = summarize_manual_sequence_dataset(dataset_dir)
    if not sample_paths:
        return (
            np.empty((0, 0, 3, 0, 0), dtype=np.uint8),
            np.empty((0, 0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype='<U16'),
            np.empty((0,), dtype=np.float32),
            stats,
        )

    rgb_windows = []
    depth_windows = []
    labels = []
    rewards = []
    for sample_path in sample_paths:
        with np.load(sample_path, allow_pickle=False) as data:
            rgb_windows.append(np.asarray(data['rgb'], dtype=np.uint8))
            depth_windows.append(np.asarray(data['depth'], dtype=np.float32))
            labels.append(str(data['label']))
            rewards.append(float(data['reward']))

    return (
        np.stack(rgb_windows, axis=0),
        np.stack(depth_windows, axis=0),
        np.asarray(labels, dtype='<U16'),
        np.asarray(rewards, dtype=np.float32),
        stats,
    )


# JACKAL RGB-D IMPLEMENTATION BLOCK:
# Uncomment this export helper if you want the manual Jackal dataset on disk to
# directly match the Sanchez-style loader shape contract [N, look_back, 4, H, W].
# This uses the existing manual capture samples, which currently store RGB and
# depth separately inside each sample_*.npz file.
#
def export_manual_dataset(
    dataset_dir: str,
    output_dir: str,
    environment_name: str = 'Jackal-v0',
    depth_clip_m: float = 5.0,
) -> dict[str, int | float | str]:
    from fear_jackal_sim.dataset_tools import (
        DANGER_CLASS_NUMBER,
        DANGER_REWARD_LABEL,
        SAFE_CLASS_NUMBER,
        SAFE_REWARD_LABEL,
    )
    from fear_jackal_sim.vision_utils import merge_rgb_depth_dataset
    rgb_windows, depth_windows, labels, rewards, stats = load_manual_sequence_dataset(dataset_dir)
    observations = merge_rgb_depth_dataset(rgb_windows, depth_windows, depth_clip_m=depth_clip_m)
    class_numbers = np.asarray(
        [DANGER_CLASS_NUMBER if label == UNSAFE_LABEL else SAFE_CLASS_NUMBER for label in labels],
        dtype=np.int64,
    )
    reward_labels = np.asarray(
        [DANGER_REWARD_LABEL if label == UNSAFE_LABEL else SAFE_REWARD_LABEL for label in labels],
        dtype=np.int64,
    )
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f'{environment_name}_lookback_{observations.shape[1]}')
    np.save(prefix + 'observations.npy', observations)
    np.save(prefix + 'class.npy', labels)
    np.save(prefix + 'class_number.npy', class_numbers)
    np.save(prefix + 'reward.npy', rewards)
    np.save(prefix + 'reward_labels.npy', reward_labels)
    canonical_metadata_path = os.path.join(output_dir, 'metadata.json')
    canonical_observations_path = os.path.join(output_dir, 'observations.npy')
    canonical_class_path = os.path.join(output_dir, 'class.npy')
    canonical_class_number_path = os.path.join(output_dir, 'class_number.npy')
    canonical_reward_labels_path = os.path.join(output_dir, 'reward_labels.npy')
    np.save(canonical_observations_path, observations)
    np.save(canonical_class_path, labels)
    np.save(canonical_class_number_path, class_numbers)
    np.save(canonical_reward_labels_path, reward_labels)
    metadata = {
        'dataset_dir': output_dir,
        'source': 'manual_jackal_low_shot',
        'source_dataset_dir': dataset_dir,
        'samples': int(observations.shape[0]),
        'look_back': int(observations.shape[1]),
        'channels': int(observations.shape[2]),
        'image_height': int(observations.shape[3]),
        'image_width': int(observations.shape[4]),
        'unsafe_samples': int(np.sum(labels == UNSAFE_LABEL)),
        'safe_samples': int(np.sum(labels == SAFE_LABEL)),
        'mean_reward': float(np.mean(rewards)) if rewards.size else 0.0,
        'reward_label_map': {
            UNSAFE_LABEL: int(DANGER_REWARD_LABEL),
            SAFE_LABEL: int(SAFE_REWARD_LABEL),
        },
        'class_number_map': {
            UNSAFE_LABEL: int(DANGER_CLASS_NUMBER),
            SAFE_LABEL: int(SAFE_CLASS_NUMBER),
        },
        'raw_stats': stats,
    }
    with open(canonical_metadata_path, 'w', encoding='ascii') as handle:
        json.dump(metadata, handle, indent=2)
    return {
        'observations_path': prefix + 'observations.npy',
        'class_path': prefix + 'class.npy',
        'class_number_path': prefix + 'class_number.npy',
        'reward_path': prefix + 'reward.npy',
        'reward_labels_path': prefix + 'reward_labels.npy',
        'canonical_metadata_path': canonical_metadata_path,
        'samples': int(observations.shape[0]),
        'look_back': int(observations.shape[1]),
        'channels': int(observations.shape[2]),
        'image_height': int(observations.shape[3]),
        'image_width': int(observations.shape[4]),
        'unsafe_samples': int(np.sum(labels == UNSAFE_LABEL)),
        'safe_samples': int(np.sum(labels == SAFE_LABEL)),
        'mean_reward': float(np.mean(rewards)) if rewards.size else 0.0,
        'source_dataset_dir': dataset_dir,
        'raw_stats': stats,
    }

def summarize_manual_sequence_dataset(dataset_dir: str) -> dict[str, int | float | str]:
    """
    Compute simple counts and reward statistics for the manual dataset.
    """
    sample_paths = iter_manual_sequence_samples(dataset_dir)
    unsafe_count = 0
    safe_count = 0
    rewards: list[float] = []
    for sample_path in sample_paths:
        with np.load(sample_path, allow_pickle=False) as data:
            label = str(data['label']).strip().lower()
            reward = float(data['reward'])
        rewards.append(reward)
        if label == UNSAFE_LABEL:
            unsafe_count += 1
        elif label == SAFE_LABEL:
            safe_count += 1

    return {
        'dataset_dir': dataset_dir,
        'samples': len(sample_paths),
        'unsafe_samples': unsafe_count,
        'safe_samples': safe_count,
        'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
    }
