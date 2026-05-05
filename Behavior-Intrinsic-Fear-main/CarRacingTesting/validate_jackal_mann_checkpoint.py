# Validate a Jackal RGB-D SMANN checkpoint with the same fresh-state call used live.
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from aio_complex import EncapsulatedNTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate Jackal RGB-D SMANN checkpoint calibration.')
    parser.add_argument('--dataset-dir', default='/workspaces/clearpath_docker/clearpath_ws/logs/rodney_dataset/')
    parser.add_argument('--checkpoint', default='/workspaces/clearpath_docker/clearpath_ws/logs/rodney_training/jackal_mann_independent/weights')
    return parser.parse_args()


def dataset_prefix(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, 'Jackal-v0_lookback_3')


def load_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = dataset_prefix(dataset_dir)
    observations = np.load(prefix + 'observations.npy')
    labels = np.load(prefix + 'class_number.npy').astype(np.int64)
    names = np.load(prefix + 'class.npy')
    if observations.ndim != 5:
        raise ValueError(f'Expected observations [N, lookback, channels, H, W], got {observations.shape}.')
    if observations.shape[2] != 4:
        raise ValueError(f'Expected RGB-D observations with 4 channels, got {observations.shape}.')
    return observations.astype(np.uint8), labels, names


def build_model(channels: int, image_size: int) -> EncapsulatedNTM:
    return EncapsulatedNTM(
        [channels, image_size, image_size],
        2,
        1,
        controller_size=250,
        controller_layers=7,
        num_read_heads=30,
        num_write_heads=30,
        N=128,
        M=60,
    )


def load_checkpoint(model, checkpoint: str) -> None:
    model.ntm.fc.load_state_dict(torch.load(os.path.join(checkpoint, 'decision_layer.pth'), map_location='cpu'))
    model.ntm.controller.complexlstm.load_state_dict(torch.load(os.path.join(checkpoint, 'controller_weights.pth'), map_location='cpu'))
    model.ntm.heads.load_state_dict(torch.load(os.path.join(checkpoint, 'heads.pth'), map_location='cpu'))
    memory_path = os.path.join(checkpoint, 'memory.pth')
    if os.path.exists(memory_path):
        model.memory.load_memory(checkpoint + os.sep)
    model.eval()


def score_window(model, window: np.ndarray) -> float:
    lookback, channels, height, width = window.shape
    batch = np.zeros((2, lookback, channels, height, width), dtype=np.float32)
    batch[0] = window.astype(np.float32) / 255.0
    inputs = torch.from_numpy(batch).permute(1, 0, 2, 3, 4).contiguous()
    delimiter = torch.zeros((2, 2), dtype=torch.float32)
    with torch.no_grad():
        model.init_sequence(2)
        logits, _ = model(x=inputs, delimeter=delimiter, previous_state=None, seq=lookback)
        probabilities = torch.softmax(logits[0], dim=-1)
    return float(probabilities[0].item())


def print_stats(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f'{name}: count=0')
        return
    print(
        f'{name}: count={values.size} min={values.min():.6f} '
        f'mean={values.mean():.6f} max={values.max():.6f} std={values.std():.6f}'
    )


def main() -> None:
    args = parse_args()
    observations, labels, names = load_dataset(args.dataset_dir)
    model = build_model(int(observations.shape[2]), int(observations.shape[3]))
    load_checkpoint(model, args.checkpoint)
    scores = np.asarray([score_window(model, observations[index]) for index in range(len(observations))], dtype=np.float32)
    unsafe_scores = scores[labels == 0]
    safe_scores = scores[labels == 1]
    print(f'dataset={args.dataset_dir}')
    print(f'checkpoint={args.checkpoint}')
    print(f'observations_shape={observations.shape}')
    print_stats('all', scores)
    print_stats('unsafe_class_0', unsafe_scores)
    print_stats('safe_class_1', safe_scores)
    print('first_10=', [(str(names[i]), int(labels[i]), float(scores[i])) for i in range(min(10, len(scores)))])
    for threshold in [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95]:
        predicted_unsafe = scores >= threshold
        true_unsafe = labels == 0
        true_positive = int(np.logical_and(predicted_unsafe, true_unsafe).sum())
        false_positive = int(np.logical_and(predicted_unsafe, ~true_unsafe).sum())
        true_negative = int(np.logical_and(~predicted_unsafe, ~true_unsafe).sum())
        false_negative = int(np.logical_and(~predicted_unsafe, true_unsafe).sum())
        accuracy = float((true_positive + true_negative) / len(scores))
        print(
            f'threshold={threshold:.2f} accuracy={accuracy:.3f} '
            f'tp={true_positive} fp={false_positive} tn={true_negative} fn={false_negative}'
        )
    if unsafe_scores.size and safe_scores.size:
        margin = float(unsafe_scores.mean() - safe_scores.mean())
        print(f'unsafe_minus_safe_mean={margin:.6f}')
        if abs(margin) < 0.05 or float(scores.std()) < 0.01:
            print('WARNING: scores are still poorly separated; do not use this checkpoint for threshold sweeps yet.')


if __name__ == '__main__':
    main()
