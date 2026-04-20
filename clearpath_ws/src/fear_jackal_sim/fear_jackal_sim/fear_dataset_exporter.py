"""
CLI for converting archived training episodes into an offline fear dataset.
"""
from __future__ import annotations

import argparse
import json

from fear_jackal_sim.dataset_tools import export_smann_dataset


def parse_args() -> argparse.Namespace:
    """
    Parse exporter CLI arguments.
    """
    parser = argparse.ArgumentParser(description='Export archived Jackal episodes into a Sanchez-style fear dataset.')
    parser.add_argument(
        '--archive-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs/episode_archives',
        help='Directory containing archived episode_*.npz files from the fear trainer.',
    )
    parser.add_argument(
        '--output-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs/smann_dataset',
        help='Directory where observations.npy, class.npy, and class_number.npy will be written.',
    )
    parser.add_argument(
        '--safe-to-danger-ratio',
        type=float,
        default=1.0,
        help='Maximum ratio of safe windows to danger windows in the exported dataset.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed used when subsampling safe windows.',
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the archive export and print a JSON summary.
    """
    args = parse_args()
    metadata = export_smann_dataset(
        archive_dir=args.archive_dir,
        output_dir=args.output_dir,
        safe_to_danger_ratio=args.safe_to_danger_ratio,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
