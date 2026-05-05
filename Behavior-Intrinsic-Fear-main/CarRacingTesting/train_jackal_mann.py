"""
Train the Jackal RGB-D memory-augmented fear model on the converted Sanchez-style dataset.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from train_and_set_model_complex_car_racing import MANN_Handler


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for offline Jackal MANN training.
    """
    parser = argparse.ArgumentParser(description='Train the Jackal RGB-D MANN on the converted offline dataset.')
    parser.add_argument(
        '--dataset-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs/rodney_dataset/',
        help='Directory containing Jackal-v0_lookback_3observations.npy and matching labels.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Fixed batch size used by Rodney Sanchez\'s training loop.',
    )
    parser.add_argument(
        '--output-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs/rodney_training',
        help='Directory where losses, accuracies, metadata, and model weights will be written.',
    )
    parser.add_argument(
        '--run-name',
        default='jackal_mann',
        help='Subdirectory name used under --output-dir for this training run.',
    )
    return parser.parse_args()


def ensure_trailing_sep(path: str) -> str:
    """
    Normalize dataset directories so the underlying loader can append filenames safely.
    """
    if path.endswith(os.sep):
        return path
    return path + os.sep


def as_float_list(values) -> list[float]:
    """
    Convert tensor-like metric values into JSON- and numpy-friendly Python floats.
    """
    floats: list[float] = []
    for value in values:
        if hasattr(value, 'item'):
            floats.append(float(value.item()))
        else:
            floats.append(float(value))
    return floats


def main() -> None:
    """
    Train the Jackal RGB-D MANN and persist the resulting metrics and weights.
    """
    args = parse_args()
    dataset_dir = ensure_trailing_sep(args.dataset_dir)

    run_dir = os.path.join(args.output_dir, args.run_name)
    weights_dir = os.path.join(run_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    handler = MANN_Handler(dataset_dir, args.batch_size)
    losses, accuracies, _ = handler.train_model()

    loss_values = as_float_list(losses)
    accuracy_values = as_float_list(accuracies)

    # Rodney's save path expects a directory prefix that ends with a separator.
    handler.model.save(weights_dir + os.sep)
    np.save(os.path.join(run_dir, 'losses.npy'), np.asarray(loss_values, dtype=np.float32))
    np.save(os.path.join(run_dir, 'accuracies.npy'), np.asarray(accuracy_values, dtype=np.float32))

    metadata = {
        'dataset_dir': dataset_dir,
        'batch_size': int(args.batch_size),
        'samples_loaded': int(handler.num_data),
        'look_back': int(handler.look_back),
        'channels': int(handler.channels),
        'input_size': int(handler.input_size),
        'epochs': int(len(loss_values)),
        'final_accuracy': float(accuracy_values[-1]) if accuracy_values else None,
        'weights_dir': weights_dir,
    }
    with open(os.path.join(run_dir, 'metadata.json'), 'w', encoding='ascii') as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"Saved metrics to {run_dir}")
    print(f"Saved model weights to {weights_dir}")


if __name__ == '__main__':
    main()
