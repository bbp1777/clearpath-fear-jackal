"""
CLI for offline supervised training of the Sanchez SMANN model on exported Jackal windows.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from fear_jackal_sim.dataset_tools import load_exported_smann_dataset
from fear_jackal_sim.smann import SMANNAdapter

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for offline SMANN training.
    """
    parser = argparse.ArgumentParser(description='Train the Sanchez fear model on exported Jackal episode windows.')
    parser.add_argument(
        '--dataset-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs/rodney_dataset',
        help='Directory containing observations.npy, class.npy, and class_number.npy.',
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs/smann_checkpoint',
        help='Directory where decision_layer.pth, controller_weights.pth, and heads.pth will be written.',
    )
    parser.add_argument(
        '--fear-repo-path',
        default='/workspaces/Behavior-Intrinsic-Fear-main/CarRacingTesting',
        help='Path to the Behavior-Intrinsic-Fear CarRacingTesting source directory inside Docker.',
    )
    parser.add_argument(
        '--warm-start-checkpoint',
        default='',
        help='Optional existing checkpoint directory to load before training.',
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lookback', type=int, default=3)
    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--fear-threshold', type=float, default=0.5)
    parser.add_argument(
        '--tensorboard-log-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs/tensorboard',
        help='TensorBoard root used for offline SMANN training metrics.',
    )
    parser.add_argument(
        '--run-name',
        default='smann_offline_manual_63',
        help='TensorBoard run name for offline SMANN training.',
    )
    return parser.parse_args()


def main() -> int:
    """
    Load the exported dataset, train the adapter, and save the checkpoint.
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger('train_smann')

    observations, class_names, class_numbers, metadata = load_exported_smann_dataset(args.dataset_dir)
    logger.info(
        'Loaded dataset from %s with %d windows (%d danger / %d safe).',
        args.dataset_dir,
        len(observations),
        int((class_numbers == 0).sum()),
        int((class_numbers == 1).sum()),
    )
    logger.info('Dataset metadata: %s', json.dumps(metadata, indent=2))
    writer = None
    if SummaryWriter is not None and args.tensorboard_log_dir:
        run_dir = os.path.join(args.tensorboard_log_dir, args.run_name)
        writer = SummaryWriter(log_dir=run_dir)
        writer.add_text('run/metadata', f'```json\n{json.dumps(metadata, indent=2)}\n```', 0)

    adapter = SMANNAdapter(
        checkpoint_path=args.warm_start_checkpoint,
        repo_path=args.fear_repo_path,
        image_size=args.image_size,
        lookback=args.lookback,
        fear_threshold=args.fear_threshold,
    )
    adapter.load(logger)
    metrics = adapter.train_supervised_dataset(
        observations=observations,
        class_numbers=class_numbers,
        logger=logger,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    if not bool(metrics.get('trained', False)):
        logger.error('SMANN supervised training did not run. Check the dataset size and repo path.')
        if writer is not None:
            writer.close()
        return 1

    if not adapter.save_checkpoint(args.checkpoint_dir, logger):
        logger.error('Training completed but checkpoint export failed.')
        if writer is not None:
            writer.close()
        return 1

    if writer is not None:
        writer.add_scalar('smann_offline/loss', float(metrics.get('loss', 0.0)), 0)
        writer.add_scalar('smann_offline/samples', float(metrics.get('samples', 0)), 0)
        writer.add_scalar('smann_offline/epochs', float(metrics.get('epochs', 0)), 0)
        writer.add_scalar('smann_offline/batches', float(metrics.get('batches', 0)), 0)
        writer.close()

    logger.info('Final training metrics: %s', json.dumps(metrics, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
