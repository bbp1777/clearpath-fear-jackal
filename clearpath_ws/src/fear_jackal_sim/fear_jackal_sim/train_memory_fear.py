"""
CLI for building an offline memory-similarity bank from manually captured three-step
samples.
"""
from __future__ import annotations

import argparse
import json
import logging

from fear_jackal_sim.memory_fear import build_memory_bank_from_dataset, save_memory_bank


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for memory-bank building.
    """
    parser = argparse.ArgumentParser(description='Build an offline memory-similarity fear bank from manual 3-step RGB+depth samples.')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--depth-clip-m', type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    """
    Build the memory bank and log the resulting summary.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger('train_memory_fear')

    bank = build_memory_bank_from_dataset(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        depth_clip_m=args.depth_clip_m,
    )
    summary = save_memory_bank(args.output_path, bank)
    logger.info('Built offline memory fear bank: %s', json.dumps(summary, indent=2))
    logger.info('Dataset metadata: %s', json.dumps(bank['metadata'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
