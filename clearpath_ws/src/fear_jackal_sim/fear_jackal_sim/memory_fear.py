"""
Offline memory-similarity fear model that compares the current three-step RGB-D window to
stored examples.
"""
from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

from fear_jackal_sim.manual_sequence_dataset import load_manual_sequence_dataset
from fear_jackal_sim.rl_types import ObservationBundle
from fear_jackal_sim.vision_utils import depth_image_to_numpy, resize_nearest, rgb_image_to_numpy


class SequenceMemoryRewardModel:
    """Offline three-step memory bank used for low-shot fear rewards.

    We keep Rodney Sanchez's fixed lookback idea here: every memory item is a short
    sequence of consecutive observations instead of a single image. The difference is
    that this offline-first path uses a similarity bank built from labeled robot data
    rather than training the full SMANN online during simulator runs.
    """

    def __init__(
        self,
        dataset_dir: str = '',
        bank_path: str = '',
        lookback: int = 3,
        image_size: int = 84,
        depth_clip_m: float = 5.0,
    ) -> None:
        """
        Configure where the bank or raw dataset lives and how windows are encoded.
        """
        self.dataset_dir = str(dataset_dir)
        self.bank_path = str(bank_path)
        self.lookback = int(lookback)
        self.image_size = int(image_size)
        self.depth_clip_m = float(depth_clip_m)
        self._memory_vectors = np.empty((0, 0), dtype=np.float32)
        self._memory_rewards = np.empty((0,), dtype=np.float32)
        self._memory_labels = np.empty((0,), dtype='<U16')
        self._loaded = False

    def load(self, logger) -> None:
        """
        Load a bank from disk or build one from manual samples.
        """
        if self.bank_path and os.path.isfile(self.bank_path):
            with np.load(self.bank_path, allow_pickle=False) as data:
                # Support the new all-samples bank format and the earlier unsafe/safe split.
                if 'memory_vectors' in data:
                    self._memory_vectors = np.asarray(data['memory_vectors'], dtype=np.float32)
                    self._memory_rewards = np.asarray(data['memory_rewards'], dtype=np.float32)
                    self._memory_labels = np.asarray(data['memory_labels']).astype('<U16')
                else:
                    unsafe_vectors = np.asarray(data['unsafe_vectors'], dtype=np.float32)
                    safe_vectors = np.asarray(data['safe_vectors'], dtype=np.float32)
                    vector_dim = unsafe_vectors.shape[1] if unsafe_vectors.size else (safe_vectors.shape[1] if safe_vectors.size else 0)
                    self._memory_vectors = np.concatenate(
                        (
                            unsafe_vectors if unsafe_vectors.size else np.empty((0, vector_dim), dtype=np.float32),
                            safe_vectors if safe_vectors.size else np.empty((0, vector_dim), dtype=np.float32),
                        ),
                        axis=0,
                    )
                    self._memory_rewards = np.concatenate(
                        (
                            np.full((len(unsafe_vectors),), -1.0, dtype=np.float32),
                            np.zeros((len(safe_vectors),), dtype=np.float32),
                        ),
                        axis=0,
                    )
                    self._memory_labels = np.concatenate(
                        (
                            np.full((len(unsafe_vectors),), 'unsafe', dtype='<U16'),
                            np.full((len(safe_vectors),), 'safe', dtype='<U16'),
                        ),
                        axis=0,
                    )
            self._loaded = len(self._memory_vectors) > 0
            unsafe_count = int((self._memory_rewards < 0.0).sum())
            safe_count = int((self._memory_rewards >= 0.0).sum())
            logger.info(
                f'Loaded offline memory-similarity fear bank from {self.bank_path} '
                f'(samples={len(self._memory_vectors)} unsafe={unsafe_count} safe={safe_count}).'
            )
            if safe_count == 0 and unsafe_count > 0:
                logger.warning('Offline memory bank currently contains only unsafe samples; add safe sidewalk sequences so unknown states do not all look mildly dangerous.')
            return

        if not self.dataset_dir:
            logger.info('Memory-similarity fear dataset/bank was not provided; intrinsic fear will stay at 0.0.')
            return

        bank = build_memory_bank_from_dataset(
            dataset_dir=self.dataset_dir,
            image_size=self.image_size,
            depth_clip_m=self.depth_clip_m,
        )
        self._memory_vectors = bank['memory_vectors']
        self._memory_rewards = bank['memory_rewards']
        self._memory_labels = bank['memory_labels']
        self._loaded = len(self._memory_vectors) > 0
        logger.info(
            'Loaded memory-similarity fear dataset '
            f"samples={bank['metadata']['samples']} unsafe={bank['metadata']['unsafe_samples']} safe={bank['metadata']['safe_samples']} path={self.dataset_dir}"
        )
        if int(bank['metadata']['safe_samples']) == 0 and int(bank['metadata']['unsafe_samples']) > 0:
            logger.warning('Offline memory dataset currently contains only unsafe samples; add safe sidewalk sequences so unknown states do not all look mildly dangerous.')

    def prepare_window(self, observation_window: Sequence[ObservationBundle], logger):
        """
        Convert the latest observations into the RGB-D layout expected by the bank.
        """
        if len(observation_window) < self.lookback:
            return None

        recent_window = observation_window[-self.lookback:]
        rgb_frames = []
        depth_frames = []
        for observation in recent_window:
            if observation.color_msg is None or observation.depth_msg is None:
                return None
            try:
                rgb = resize_nearest(rgb_image_to_numpy(observation.color_msg), self.image_size, self.image_size)
                depth = resize_nearest(depth_image_to_numpy(observation.depth_msg), self.image_size, self.image_size)
            except Exception as exc:
                logger.warning(f'Unable to prepare memory-similarity fear window: {exc}')
                return None
            rgb_frames.append(rgb.transpose(2, 0, 1).astype(np.uint8, copy=False))
            depth_frames.append(depth.astype(np.float32, copy=False))

        return {
            'rgb': np.stack(rgb_frames, axis=0),
            'depth': np.stack(depth_frames, axis=0),
        }

    def predict_reward(self, observation_window: Sequence[ObservationBundle], logger) -> float:
        """
        Predict the intrinsic reward for the current observation window.
        """
        prepared_window = self.prepare_window(observation_window, logger)
        return self.predict_reward_prepared(prepared_window, logger)

    def predict_reward_prepared(self, prepared_window, logger) -> float:
        """
        Predict the intrinsic reward from an already prepared RGB-D window.
        """
        if prepared_window is None or not self._loaded or len(self._memory_vectors) == 0:
            return 0.0

        try:
            # The live three-step window is projected into the same normalized space
            # as the stored memories before similarity lookup is performed.
            current_vector = encode_rgbd_windows(
                prepared_window['rgb'],
                prepared_window['depth'],
                depth_clip_m=self.depth_clip_m,
            )
            similarities = np.clip(self._memory_vectors @ current_vector, 0.0, 1.0)
            if not np.any(similarities > 0.0):
                return 0.0

            # Use a small top-k weighted lookup so rewards stay tied to stored examples.
            top_k = min(5, len(similarities))
            top_indices = np.argsort(similarities)[-top_k:]
            top_similarities = similarities[top_indices]
            weight_total = float(np.sum(top_similarities))
            if weight_total <= 1e-8:
                return 0.0

            normalized_weights = top_similarities / weight_total
            predicted_reward = float(np.sum(normalized_weights * self._memory_rewards[top_indices]))

            # Scale the stored reward by match confidence so a bank that only contains
            # unsafe examples does not make every unfamiliar state look weakly terminal.
            match_confidence = float(np.max(top_similarities))
            predicted_reward *= match_confidence
            return float(np.clip(predicted_reward, -1.0, 0.0))
        except Exception as exc:
            logger.warning(f'Memory-similarity fear reward lookup failed; returning zero intrinsic fear for this step: {exc}')
            return 0.0

    def score(self, observation_window: Sequence[ObservationBundle], logger) -> float:
        """
        Return the positive fear score for the current observation window.
        """
        prepared_window = self.prepare_window(observation_window, logger)
        return self.score_prepared(prepared_window, logger)

    def score_prepared(self, prepared_window, logger) -> float:
        """
        Convert a prepared-window reward prediction into a [0, 1] fear score.
        """
        reward = self.predict_reward_prepared(prepared_window, logger)
        fear_score = float(np.clip(-reward, 0.0, 1.0))
        return 0.0 if abs(fear_score) < 1e-8 else fear_score


def encode_rgbd_windows(rgb_window: np.ndarray, depth_window: np.ndarray, depth_clip_m: float = 5.0) -> np.ndarray:
    """
    Flatten and normalize a three-step RGB-D window into the bank vector space.
    """
    rgb = np.asarray(rgb_window, dtype=np.float32) / 255.0
    depth = np.asarray(depth_window, dtype=np.float32)
    depth = np.clip(depth, 0.0, depth_clip_m) / max(depth_clip_m, 1e-6)
    vector = np.concatenate((rgb.reshape(-1), depth.reshape(-1)), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def build_memory_bank_from_dataset(
    dataset_dir: str,
    image_size: int = 84,
    depth_clip_m: float = 5.0,
) -> dict[str, np.ndarray | dict[str, int | float | str]]:
    """
    Encode every manual sample into a bank of memory vectors plus metadata.
    """
    rgb_windows, depth_windows, labels, rewards, metadata = load_manual_sequence_dataset(dataset_dir)
    if len(labels) == 0:
        return {
            'memory_vectors': np.empty((0, 0), dtype=np.float32),
            'memory_rewards': np.empty((0,), dtype=np.float32),
            'memory_labels': np.empty((0,), dtype='<U16'),
            'metadata': metadata,
        }

    memory_vectors = []
    for rgb_window, depth_window in zip(rgb_windows, depth_windows):
        memory_vectors.append(encode_rgbd_windows(rgb_window, depth_window, depth_clip_m=depth_clip_m))

    vector_dim = len(memory_vectors[0]) if memory_vectors else 0
    bank_metadata = {
        **metadata,
        'image_size': int(image_size),
        'depth_clip_m': float(depth_clip_m),
        'vector_dim': int(vector_dim),
        'mean_reward': float(np.mean(rewards)) if len(rewards) else 0.0,
    }
    return {
        'memory_vectors': np.stack(memory_vectors, axis=0).astype(np.float32) if memory_vectors else np.empty((0, vector_dim), dtype=np.float32),
        'memory_rewards': np.asarray(rewards, dtype=np.float32),
        'memory_labels': np.asarray(labels, dtype='<U16'),
        'metadata': bank_metadata,
    }


def save_memory_bank(bank_path: str, bank: dict[str, np.ndarray | dict[str, int | float | str]]) -> dict[str, str | int | float]:
    """
    Persist the encoded memory bank to a compressed .npz file.
    """
    os.makedirs(os.path.dirname(bank_path), exist_ok=True)
    metadata = dict(bank['metadata'])
    metadata_json = json.dumps(metadata)
    np.savez_compressed(
        bank_path,
        memory_vectors=np.asarray(bank['memory_vectors'], dtype=np.float32),
        memory_rewards=np.asarray(bank['memory_rewards'], dtype=np.float32),
        memory_labels=np.asarray(bank['memory_labels'], dtype='<U16'),
        metadata_json=np.asarray(metadata_json),
    )
    reward_array = np.asarray(bank['memory_rewards'], dtype=np.float32)
    return {
        'path': bank_path,
        'samples': int(len(reward_array)),
        'unsafe_samples': int(np.sum(reward_array < 0.0)),
        'safe_samples': int(np.sum(reward_array >= 0.0)),
    }


