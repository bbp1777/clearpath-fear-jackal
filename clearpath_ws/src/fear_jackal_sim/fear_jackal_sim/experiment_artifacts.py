"""
Utilities for preserving experiment outputs before fresh paper runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


DEFAULT_ARCHIVE_TARGETS = [
    'tensorboard',
    'episode_archives',
    'rodney_training',
    'smann_checkpoint',
    'smann_dataset',
]


def archive_experiment_outputs(
    logs_dir: str,
    archive_root: str | None = None,
    targets: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """
    Move generated experiment outputs aside while preserving the manual low-shot dataset.
    """
    logs_path = Path(logs_dir).resolve()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    archive_path = Path(archive_root).resolve() if archive_root else logs_path / 'archived_runs' / timestamp
    selected_targets = targets or DEFAULT_ARCHIVE_TARGETS

    summary: dict[str, object] = {
        'logs_dir': str(logs_path),
        'archive_dir': str(archive_path),
        'dry_run': bool(dry_run),
        'moved': [],
        'skipped': [],
        'preserved_manual_dataset': str(logs_path / 'rodney_dataset'),
    }

    for target in selected_targets:
        source = logs_path / target
        if target == 'rodney_dataset':
            summary['skipped'].append({'target': target, 'reason': 'manual low-shot dataset is preserved'})
            continue
        if not source.exists():
            summary['skipped'].append({'target': target, 'reason': 'not found'})
            continue

        destination = archive_path / target
        summary['moved'].append({'source': str(source), 'destination': str(destination)})
        if dry_run:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f'Archive destination already exists: {destination}')
        shutil.move(str(source), str(destination))

    if not dry_run:
        archive_path.mkdir(parents=True, exist_ok=True)
        with open(archive_path / 'archive_summary.json', 'w', encoding='ascii') as handle:
            json.dump(summary, handle, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Archive generated fear-training outputs before a fresh run.')
    parser.add_argument(
        '--logs-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs',
        help='Root logs directory containing TensorBoard, episode archives, and SMANN outputs.',
    )
    parser.add_argument(
        '--archive-root',
        default='',
        help='Optional explicit archive directory. Defaults to logs/archived_runs/<timestamp>.',
    )
    parser.add_argument(
        '--target',
        action='append',
        dest='targets',
        help='Specific generated output directory to archive. May be repeated.',
    )
    parser.add_argument('--dry-run', action='store_true', help='Print what would move without changing files.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = archive_experiment_outputs(
        logs_dir=args.logs_dir,
        archive_root=args.archive_root or None,
        targets=args.targets,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
