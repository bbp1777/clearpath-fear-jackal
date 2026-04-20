"""
Audit local Sanchez Behavior-Intrinsic-Fear code against the canonical upstream repo.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


CORE_FILES = {
    'aio_complex.py',
    'ntm_complex.py',
    'complexcontroller.py',
    'head.py',
    'memory.py',
}

JACKAL_ADAPTER_FILES = {
    'DatasetMaker.py',
    'train_and_set_model_complex_car_racing.py',
    'vision_utils.py',
}


def _run(command: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        joined = ' '.join(command)
        raise RuntimeError(f'Command failed ({joined}):\n{result.stdout}')
    return result


def _changed_files_from_diff(diff_text: str, local_root: str) -> list[str]:
    changed: set[str] = set()
    local_prefix = os.path.abspath(local_root) + os.sep
    for line in diff_text.splitlines():
        if not line.startswith('diff -ruN '):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        local_path = parts[-1]
        if local_path == '/dev/null':
            continue
        abs_local = os.path.abspath(local_path)
        if abs_local.startswith(local_prefix):
            changed.add(os.path.relpath(abs_local, local_root))
    return sorted(changed)


def _classify_changed_file(path: str) -> str:
    basename = os.path.basename(path)
    if basename in CORE_FILES:
        return 'unnecessary_core_drift'
    if basename in JACKAL_ADAPTER_FILES:
        return 'required_jackal_adapter'
    if path.startswith('__pycache__') or path.endswith('.pyc'):
        return 'ignored_runtime_artifact'
    return 'needs_manual_classification'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Diff local Sanchez code against the canonical upstream repository.')
    parser.add_argument(
        '--upstream-url',
        default='https://github.com/ras8047/Behavior-Intrinsic-Fear.git',
        help='Canonical upstream Behavior-Intrinsic-Fear Git repository.',
    )
    parser.add_argument(
        '--upstream-commit',
        default='',
        help='Optional upstream commit SHA to pin before diffing.',
    )
    parser.add_argument(
        '--local-path',
        default='/workspaces/Behavior-Intrinsic-Fear-main/CarRacingTesting',
        help='Local Sanchez CarRacingTesting directory used by the Jackal adapter.',
    )
    parser.add_argument(
        '--output-dir',
        default='/workspaces/clearpath_docker/clearpath_ws/logs/source_audits',
        help='Directory where audit reports and raw diffs are written.',
    )
    parser.add_argument(
        '--work-dir',
        default='/tmp/fear_sanchez_upstream_audit',
        help='Temporary clone directory for the upstream repository.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_path = os.path.abspath(args.local_path)
    output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.abspath(args.work_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    if not os.path.isdir(local_path):
        raise SystemExit(f'Local Sanchez path does not exist: {local_path}')

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        _run(['git', 'clone', '--depth', '1', args.upstream_url, work_dir])
    except Exception as exc:
        failure = {
            'status': 'failed',
            'reason': str(exc),
            'upstream_url': args.upstream_url,
            'local_path': local_path,
        }
        report_path = os.path.join(output_dir, f'sanchez_source_audit_failed_{timestamp}.md')
        json_path = os.path.join(output_dir, f'sanchez_source_audit_failed_{timestamp}.json')
        with open(json_path, 'w', encoding='ascii') as handle:
            json.dump(failure, handle, indent=2)
        with open(report_path, 'w', encoding='ascii') as handle:
            handle.write('# Sanchez Source Audit Failed\n\n')
            handle.write(f"- Upstream repo: `{args.upstream_url}`\n")
            handle.write(f"- Local path: `{local_path}`\n")
            handle.write(f"- JSON summary: `{json_path}`\n\n")
            handle.write('## Failure\n\n')
            handle.write('```text\n')
            handle.write(str(exc))
            handle.write('\n```\n')
            handle.write(
                '\nThe most common cause is that the GitHub repository is private '
                'or the container does not have credentials for it.\n'
            )
        print(json.dumps(failure, indent=2))
        return 2

    if args.upstream_commit:
        _run(['git', 'fetch', '--depth', '1', 'origin', args.upstream_commit], cwd=work_dir)
        _run(['git', 'checkout', args.upstream_commit], cwd=work_dir)

    upstream_commit = _run(['git', 'rev-parse', 'HEAD'], cwd=work_dir).stdout.strip()
    upstream_path = os.path.join(work_dir, 'CarRacingTesting')
    if not os.path.isdir(upstream_path):
        raise SystemExit(f'Upstream repository does not contain CarRacingTesting at {upstream_path}')

    diff_result = _run(
        [
            'diff',
            '-ruN',
            '--exclude=__pycache__',
            '--exclude=*.pyc',
            upstream_path,
            local_path,
        ],
        check=False,
    )
    if diff_result.returncode not in (0, 1):
        raise SystemExit(f'diff failed:\n{diff_result.stdout}')

    diff_path = os.path.join(output_dir, f'sanchez_source_diff_{timestamp}.diff')
    report_path = os.path.join(output_dir, f'sanchez_source_audit_{timestamp}.md')
    json_path = os.path.join(output_dir, f'sanchez_source_audit_{timestamp}.json')

    with open(diff_path, 'w', encoding='utf-8') as handle:
        handle.write(diff_result.stdout)

    changed_files = _changed_files_from_diff(diff_result.stdout, local_path)
    classifications = [
        {
            'path': path,
            'classification': _classify_changed_file(path),
        }
        for path in changed_files
    ]
    summary = {
        'upstream_url': args.upstream_url,
        'upstream_commit': upstream_commit,
        'local_path': local_path,
        'upstream_path': upstream_path,
        'diff_path': diff_path,
        'report_path': report_path,
        'changed_files': classifications,
        'has_core_drift': any(item['classification'] == 'unnecessary_core_drift' for item in classifications),
    }

    with open(json_path, 'w', encoding='ascii') as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        '# Sanchez Source Audit',
        '',
        f'- Upstream repo: `{args.upstream_url}`',
        f'- Upstream commit: `{upstream_commit}`',
        f'- Local path: `{local_path}`',
        f'- Raw diff: `{diff_path}`',
        f'- JSON summary: `{json_path}`',
        '',
        '## Changed Files',
        '',
    ]
    if classifications:
        for item in classifications:
            lines.append(f"- `{item['path']}`: `{item['classification']}`")
    else:
        lines.append('- No differences found.')

    lines.extend(
        [
            '',
            '## Classification Rules',
            '',
            '- `required_jackal_adapter`: expected wrapper, dataset, or training-script changes for Jackal RGB-D compatibility.',
            '- `unnecessary_core_drift`: model core drift in Sanchez architecture, memory, controller, or heads; review and revert/isolate before paper runs.',
            '- `needs_manual_classification`: inspect the raw diff and document whether it is a Jackal compatibility change.',
        ]
    )
    with open(report_path, 'w', encoding='ascii') as handle:
        handle.write('\n'.join(lines) + '\n')

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
