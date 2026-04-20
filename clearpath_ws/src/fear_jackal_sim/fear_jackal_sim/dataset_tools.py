"""
Helpers for archiving episodes and exporting balanced offline datasets from recorded
transitions.
"""
from __future__ import annotations

import json
import os
from glob import glob
from typing import Sequence

import numpy as np

from fear_jackal_sim.rl_types import Transition


DANGER_CLASS_NAME = 'danger'
SAFE_CLASS_NAME = 'safe'
DANGER_CLASS_NUMBER = 0
SAFE_CLASS_NUMBER = 1
DANGER_REWARD_LABEL = -1
SAFE_REWARD_LABEL = 0


def reward_labels_from_class_numbers(class_numbers: np.ndarray) -> np.ndarray:
    """
    Convert SMANN class numbers into the paper reward-label convention.
    """
    class_numbers = np.asarray(class_numbers, dtype=np.int64)
    return np.where(class_numbers == DANGER_CLASS_NUMBER, DANGER_REWARD_LABEL, SAFE_REWARD_LABEL).astype(np.int64)


def class_numbers_from_reward_labels(reward_labels: np.ndarray) -> np.ndarray:
    """
    Convert manual reward labels into Sanchez classifier class ids.
    """
    reward_labels = np.asarray(reward_labels, dtype=np.int64)
    return np.where(reward_labels == DANGER_REWARD_LABEL, DANGER_CLASS_NUMBER, SAFE_CLASS_NUMBER).astype(np.int64)


def archive_episode_transitions(
    output_dir: str,
    episode_index: int,
    transitions: Sequence[Transition],
) -> dict[str, int | str | bool]:
    """
    Archive one episode of labeled transitions into a compact .npz file.
    """
    os.makedirs(output_dir, exist_ok=True)
    archived = [
        transition
        for transition in transitions
        if transition.vision_window is not None and transition.danger_label is not None
    ]

    summary: dict[str, int | str | bool] = {
        'saved': False,
        'path': '',
        'windows': len(archived),
        'danger_windows': 0,
        'safe_windows': 0,
    }
    if not archived:
        return summary

    # Each archived transition already contains a ready-to-train lookback window,
    # so dataset export mainly becomes stacking arrays while keeping labels aligned.
    observations = np.stack(
        [
            np.asarray(
                transition.vision_window['rgb'] if isinstance(transition.vision_window, dict) else transition.vision_window,
                dtype=np.uint8,
            )
            for transition in archived
        ],
        axis=0,
    )
    danger_labels = np.asarray([int(transition.danger_label) for transition in archived], dtype=np.int64)
    class_numbers = np.where(danger_labels == 1, DANGER_CLASS_NUMBER, SAFE_CLASS_NUMBER).astype(np.int64)
    class_names = np.where(danger_labels == 1, DANGER_CLASS_NAME, SAFE_CLASS_NAME).astype('<U8')
    reward_labels = reward_labels_from_class_numbers(class_numbers)
    step_indices = np.asarray(
        [int(transition.state_summary.get('step_index', 0)) for transition in archived],
        dtype=np.int32,
    )
    external_rewards = np.asarray([float(transition.external_reward) for transition in archived], dtype=np.float32)
    intrinsic_rewards = np.asarray([float(transition.intrinsic_reward) for transition in archived], dtype=np.float32)
    combined_rewards = np.asarray([float(transition.combined_reward) for transition in archived], dtype=np.float32)
    terminals = np.asarray([bool(transition.terminal) for transition in archived], dtype=np.bool_)
    truncateds = np.asarray([bool(transition.truncated) for transition in archived], dtype=np.bool_)

    archive_path = os.path.join(output_dir, f'episode_{int(episode_index):06d}.npz')
    np.savez_compressed(
        archive_path,
        observations=observations,
        danger_labels=danger_labels,
        class_names=class_names,
        class_numbers=class_numbers,
        reward_labels=reward_labels,
        step_indices=step_indices,
        external_rewards=external_rewards,
        intrinsic_rewards=intrinsic_rewards,
        combined_rewards=combined_rewards,
        terminals=terminals,
        truncateds=truncateds,
    )

    summary['saved'] = True
    summary['path'] = archive_path
    summary['danger_windows'] = int((danger_labels == 1).sum())
    summary['safe_windows'] = int((danger_labels == 0).sum())
    return summary


def iter_episode_archives(archive_dir: str) -> list[str]:
    """
    List archived episode files in chronological order.
    """
    pattern = os.path.join(archive_dir, 'episode_*.npz')
    return sorted(glob(pattern))


def build_smann_dataset_from_archives(
    archive_dir: str,
    safe_to_danger_ratio: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    """
    Load archives, balance safe and danger samples, and return arrays ready for offline
    training.
    """
    archive_files = iter_episode_archives(archive_dir)
    stats: dict[str, int | float] = {
        'episodes': len(archive_files),
        'raw_windows': 0,
        'raw_danger_windows': 0,
        'raw_safe_windows': 0,
        'selected_windows': 0,
        'selected_danger_windows': 0,
        'selected_safe_windows': 0,
        'safe_to_danger_ratio': float(safe_to_danger_ratio),
    }

    if not archive_files:
        return (
            np.empty((0, 3, 4, 84, 84), dtype=np.uint8),
            np.empty((0,), dtype='<U8'),
            np.empty((0,), dtype=np.int64),
            stats,
        )

    observations_list = []
    danger_labels_list = []
    for archive_path in archive_files:
        with np.load(archive_path, allow_pickle=False) as data:
            observations_list.append(np.asarray(data['observations'], dtype=np.uint8))
            danger_labels_list.append(np.asarray(data['danger_labels'], dtype=np.int64))

    observations = np.concatenate(observations_list, axis=0) if observations_list else np.empty((0,), dtype=np.uint8)
    danger_labels = np.concatenate(danger_labels_list, axis=0) if danger_labels_list else np.empty((0,), dtype=np.int64)

    stats['raw_windows'] = int(len(observations))
    stats['raw_danger_windows'] = int((danger_labels == 1).sum())
    stats['raw_safe_windows'] = int((danger_labels == 0).sum())

    if len(observations) == 0:
        return (
            np.empty((0, 3, 4, 84, 84), dtype=np.uint8),
            np.empty((0,), dtype='<U8'),
            np.empty((0,), dtype=np.int64),
            stats,
        )

    rng = np.random.default_rng(seed)
    danger_indices = np.flatnonzero(danger_labels == 1)
    safe_indices = np.flatnonzero(danger_labels == 0)

    if len(danger_indices) == 0:
        selected_indices = safe_indices
    else:
        max_safe = max(1, int(round(len(danger_indices) * max(float(safe_to_danger_ratio), 0.0))))
        if len(safe_indices) > max_safe:
            safe_indices = np.sort(rng.choice(safe_indices, size=max_safe, replace=False))
        selected_indices = np.concatenate((danger_indices, safe_indices), axis=0)

    if len(selected_indices) == 0:
        return (
            np.empty((0, 3, 4, 84, 84), dtype=np.uint8),
            np.empty((0,), dtype='<U8'),
            np.empty((0,), dtype=np.int64),
            stats,
        )

    rng.shuffle(selected_indices)
    selected_observations = observations[selected_indices]
    selected_danger_labels = danger_labels[selected_indices]
    class_numbers = np.where(selected_danger_labels == 1, DANGER_CLASS_NUMBER, SAFE_CLASS_NUMBER).astype(np.int64)
    class_names = np.where(selected_danger_labels == 1, DANGER_CLASS_NAME, SAFE_CLASS_NAME).astype('<U8')

    stats['selected_windows'] = int(len(selected_indices))
    stats['selected_danger_windows'] = int((selected_danger_labels == 1).sum())
    stats['selected_safe_windows'] = int((selected_danger_labels == 0).sum())
    return selected_observations, class_names, class_numbers, stats


def export_smann_dataset(
    archive_dir: str,
    output_dir: str,
    safe_to_danger_ratio: float = 1.0,
    seed: int = 0,
) -> dict[str, int | float | str]:
    """
    Write the balanced offline dataset arrays and metadata to disk.
    """
    observations, class_names, class_numbers, stats = build_smann_dataset_from_archives(
        archive_dir=archive_dir,
        safe_to_danger_ratio=safe_to_danger_ratio,
        seed=seed,
    )
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'observations.npy'), observations)
    np.save(os.path.join(output_dir, 'class.npy'), class_names)
    np.save(os.path.join(output_dir, 'class_number.npy'), class_numbers)
    reward_labels = reward_labels_from_class_numbers(class_numbers)
    np.save(os.path.join(output_dir, 'reward_labels.npy'), reward_labels)

    metadata = {
        'archive_dir': archive_dir,
        'output_dir': output_dir,
        'source': 'episode_archive_export',
        'reward_label_map': {
            DANGER_CLASS_NAME: DANGER_REWARD_LABEL,
            SAFE_CLASS_NAME: SAFE_REWARD_LABEL,
        },
        'class_number_map': {
            DANGER_CLASS_NAME: DANGER_CLASS_NUMBER,
            SAFE_CLASS_NAME: SAFE_CLASS_NUMBER,
        },
        **stats,
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='ascii') as handle:
        json.dump(metadata, handle, indent=2)

    metadata['metadata_path'] = metadata_path
    return metadata


def load_exported_smann_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """
    Load a previously exported dataset from disk.
    """
    observations_path = _resolve_dataset_file(dataset_dir, 'observations.npy')
    class_path = _resolve_dataset_file(dataset_dir, 'class.npy')
    class_number_path = _resolve_dataset_file(dataset_dir, 'class_number.npy')
    observations = np.load(observations_path, allow_pickle=False)
    class_names = np.load(class_path, allow_pickle=False)
    class_numbers = np.load(class_number_path, allow_pickle=False)
    reward_labels_path = os.path.join(dataset_dir, 'reward_labels.npy')
    legacy_reward_path = _resolve_dataset_file(dataset_dir, 'reward.npy', required=False)
    if os.path.isfile(reward_labels_path):
        reward_labels = np.load(reward_labels_path, allow_pickle=False)
    elif legacy_reward_path and os.path.isfile(legacy_reward_path):
        reward_labels = np.load(legacy_reward_path, allow_pickle=False)
    else:
        reward_labels = reward_labels_from_class_numbers(class_numbers)

    metadata_path = os.path.join(dataset_dir, 'metadata.json')
    metadata: dict[str, int | float | str] = {'dataset_dir': dataset_dir}
    if os.path.isfile(metadata_path):
        with open(metadata_path, 'r', encoding='ascii') as handle:
            metadata.update(json.load(handle))
    metadata['reward_label_map'] = {
        DANGER_CLASS_NAME: DANGER_REWARD_LABEL,
        SAFE_CLASS_NAME: SAFE_REWARD_LABEL,
    }
    metadata['class_number_map'] = {
        DANGER_CLASS_NAME: DANGER_CLASS_NUMBER,
        SAFE_CLASS_NAME: SAFE_CLASS_NUMBER,
    }
    metadata['unsafe_reward_label_count'] = int((np.asarray(reward_labels) == DANGER_REWARD_LABEL).sum())
    metadata['safe_reward_label_count'] = int((np.asarray(reward_labels) == SAFE_REWARD_LABEL).sum())
    return observations, class_names, class_numbers.astype(np.int64), metadata


def _resolve_dataset_file(dataset_dir: str, canonical_name: str, required: bool = True) -> str:
    """
    Resolve canonical dataset files, falling back to Rodney's environment-prefixed names.
    """
    canonical_path = os.path.join(dataset_dir, canonical_name)
    if os.path.isfile(canonical_path):
        return canonical_path

    matches = sorted(glob(os.path.join(dataset_dir, f'*{canonical_name}')))
    if matches:
        return matches[0]

    if required:
        raise FileNotFoundError(
            f'Could not find {canonical_name} or a prefixed *{canonical_name} file in {dataset_dir}.'
        )
    return ''
