"""
Run a small 5-fold SMANN hyperparameter grid and train the selected final checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict

import numpy as np

from fear_jackal_sim.dataset_tools import load_exported_smann_dataset
from fear_jackal_sim.smann import SMANNAdapter

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(',') if part.strip()]


def _stratified_folds(class_numbers: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = max(int(folds), 2)
    fold_indices = [[] for _ in range(folds)]
    for class_id in sorted(set(int(value) for value in class_numbers.tolist())):
        indices = np.flatnonzero(class_numbers == class_id)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            fold_indices[offset % folds].append(int(index))
    return [np.asarray(sorted(indices), dtype=np.int64) for indices in fold_indices]


def _write_loss_curve(path: str, train_losses: list[float], validation_losses: list[float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='ascii') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_loss', 'validation_loss'])
        for epoch in range(1, len(train_losses) + 1):
            validation_loss = validation_losses[epoch - 1] if epoch - 1 < len(validation_losses) else ''
            writer.writerow([epoch, f'{train_losses[epoch - 1]:.10f}', validation_loss])


def _write_rows(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, 'w', newline='', encoding='ascii') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_pyplot():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, output_dir: str, basename: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, basename + '.png'), dpi=200)
    fig.savefig(os.path.join(output_dir, basename + '.pdf'))


def _summarize_grid(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row['epochs']), float(row['learning_rate']))].append(row)

    summaries: list[dict[str, object]] = []
    for (epochs, learning_rate), group in sorted(grouped.items()):
        validation_losses = np.asarray([float(row['validation_loss']) for row in group], dtype=np.float64)
        accuracies = np.asarray([float(row['validation_accuracy']) for row in group], dtype=np.float64)
        f1_scores = np.asarray([float(row['validation_unsafe_f1']) for row in group], dtype=np.float64)
        summaries.append(
            {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'folds': len(group),
                'validation_loss_mean': float(np.mean(validation_losses)),
                'validation_loss_std': float(np.std(validation_losses)),
                'validation_accuracy_mean': float(np.mean(accuracies)),
                'validation_accuracy_std': float(np.std(accuracies)),
                'validation_unsafe_f1_mean': float(np.mean(f1_scores)),
                'validation_unsafe_f1_std': float(np.std(f1_scores)),
            }
        )
    return summaries


def _select_hyperparameters(summaries: list[dict[str, object]]) -> dict[str, object]:
    if not summaries:
        raise RuntimeError('No SMANN grid summaries were produced.')
    return sorted(
        summaries,
        key=lambda row: (
            -float(row['validation_unsafe_f1_mean']),
            float(row['validation_loss_mean']),
            int(row['epochs']),
            float(row['learning_rate']),
        ),
    )[0]


def _plot_grid_outputs(
    output_dir: str,
    summaries: list[dict[str, object]],
    final_metrics: dict[str, object],
    final_evaluation: dict[str, object],
) -> list[str]:
    plt = _load_pyplot()
    outputs: list[str] = []
    labels = [f"e{row['epochs']}\nlr{row['learning_rate']:g}" for row in summaries]
    x_values = np.arange(len(summaries))

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x_values, [float(row['validation_unsafe_f1_mean']) for row in summaries])
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, rotation=45, ha='right')
    axis.set_ylabel('Mean validation unsafe F1')
    axis.set_title('SMANN hyperparameter grid F1')
    axis.grid(True, axis='y', alpha=0.3)
    _save_figure(fig, output_dir, 'smann_grid_validation_f1')
    plt.close(fig)
    outputs.extend([
        os.path.join(output_dir, 'smann_grid_validation_f1.png'),
        os.path.join(output_dir, 'smann_grid_validation_f1.pdf'),
    ])

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x_values, [float(row['validation_loss_mean']) for row in summaries])
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, rotation=45, ha='right')
    axis.set_ylabel('Mean validation loss')
    axis.set_title('SMANN hyperparameter grid loss')
    axis.grid(True, axis='y', alpha=0.3)
    _save_figure(fig, output_dir, 'smann_grid_validation_loss')
    plt.close(fig)
    outputs.extend([
        os.path.join(output_dir, 'smann_grid_validation_loss.png'),
        os.path.join(output_dir, 'smann_grid_validation_loss.pdf'),
    ])

    train_losses = [float(value) for value in final_metrics.get('epoch_losses', [])]
    if train_losses:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(np.arange(1, len(train_losses) + 1), train_losses)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Training loss')
        axis.set_title('Selected SMANN final training curve')
        axis.grid(True, alpha=0.3)
        _save_figure(fig, output_dir, 'smann_selected_training_loss')
        plt.close(fig)
        outputs.extend([
            os.path.join(output_dir, 'smann_selected_training_loss.png'),
            os.path.join(output_dir, 'smann_selected_training_loss.pdf'),
        ])

    confusion = np.asarray(
        [
            [int(final_evaluation.get('true_unsafe', 0)), int(final_evaluation.get('false_safe', 0))],
            [int(final_evaluation.get('false_unsafe', 0)), int(final_evaluation.get('true_safe', 0))],
        ],
        dtype=np.int64,
    )
    fig, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(confusion, cmap='Blues')
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(['pred unsafe', 'pred safe'])
    axis.set_yticklabels(['true unsafe', 'true safe'])
    for y in range(2):
        for x in range(2):
            axis.text(x, y, str(confusion[y, x]), ha='center', va='center')
    axis.set_title('Selected SMANN confusion matrix')
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save_figure(fig, output_dir, 'smann_selected_confusion_matrix')
    plt.close(fig)
    outputs.extend([
        os.path.join(output_dir, 'smann_selected_confusion_matrix.png'),
        os.path.join(output_dir, 'smann_selected_confusion_matrix.pdf'),
    ])

    probabilities = np.asarray(final_evaluation.get('unsafe_probabilities', []), dtype=np.float64)
    labels_array = np.asarray(final_evaluation.get('class_numbers', []), dtype=np.int64)
    if len(probabilities) == len(labels_array) and len(probabilities) > 0:
        fig, axis = plt.subplots(figsize=(6.5, 4.5))
        axis.hist(probabilities[labels_array == 1], bins=10, alpha=0.65, label='safe')
        axis.hist(probabilities[labels_array == 0], bins=10, alpha=0.65, label='unsafe')
        axis.set_xlabel('Predicted P(unsafe)')
        axis.set_ylabel('Samples')
        axis.set_title('Selected SMANN unsafe probability distribution')
        axis.legend(loc='best')
        axis.grid(True, alpha=0.3)
        _save_figure(fig, output_dir, 'smann_selected_unsafe_probability')
        plt.close(fig)
        outputs.extend([
            os.path.join(output_dir, 'smann_selected_unsafe_probability.png'),
            os.path.join(output_dir, 'smann_selected_unsafe_probability.pdf'),
        ])
    return outputs


def run_grid(args: argparse.Namespace) -> dict[str, object]:
    logger = logging.getLogger('train_smann_grid')
    observations, _class_names, class_numbers, metadata = load_exported_smann_dataset(args.dataset_dir)
    class_numbers = np.asarray(class_numbers, dtype=np.int64)
    folds = _stratified_folds(class_numbers, args.folds, args.seed)
    epochs_values = _parse_int_list(args.epochs)
    learning_rates = _parse_float_list(args.learning_rates)
    root_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(root_dir, exist_ok=True)

    fold_rows: list[dict[str, object]] = []
    writer = None
    if SummaryWriter is not None and args.tensorboard_log_dir:
        writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_log_dir, args.run_name))
        writer.add_text('run/metadata', f'```json\n{json.dumps(metadata, indent=2)}\n```', 0)

    for epochs in epochs_values:
        for learning_rate in learning_rates:
            combo_name = f'epochs_{epochs}_lr_{learning_rate:g}'.replace('.', 'p')
            logger.info('Running SMANN grid combo %s.', combo_name)
            for fold_index, validation_indices in enumerate(folds, start=1):
                train_indices = np.setdiff1d(np.arange(len(observations)), validation_indices)
                fold_dir = os.path.join(root_dir, combo_name, f'fold_{fold_index:02d}')
                adapter = SMANNAdapter(
                    checkpoint_path='',
                    repo_path=args.fear_repo_path,
                    image_size=args.image_size,
                    lookback=args.lookback,
                    fear_threshold=args.fear_threshold,
                )
                adapter.load(logger)
                train_metrics = adapter.train_supervised_dataset(
                    observations=observations[train_indices],
                    class_numbers=class_numbers[train_indices],
                    logger=logger,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    learning_rate=learning_rate,
                    validation_observations=observations[validation_indices],
                    validation_class_numbers=class_numbers[validation_indices],
                )
                validation_metrics = adapter.evaluate_supervised_dataset(
                    observations[validation_indices],
                    class_numbers[validation_indices],
                    logger,
                    batch_size=args.batch_size,
                )
                train_losses = [float(value) for value in train_metrics.get('epoch_losses', [])]
                validation_losses = [float(value) for value in train_metrics.get('validation_epoch_losses', [])]
                _write_loss_curve(os.path.join(fold_dir, 'loss_curve.csv'), train_losses, validation_losses)

                row = {
                    'combo': combo_name,
                    'fold': fold_index,
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': int(args.batch_size),
                    'train_samples': int(len(train_indices)),
                    'validation_samples': int(len(validation_indices)),
                    'train_final_loss': float(train_metrics.get('final_epoch_loss', 0.0)),
                    'validation_loss': float(validation_metrics.get('loss', 0.0)),
                    'validation_accuracy': float(validation_metrics.get('accuracy', 0.0)),
                    'validation_unsafe_precision': float(validation_metrics.get('unsafe_precision', 0.0)),
                    'validation_unsafe_recall': float(validation_metrics.get('unsafe_recall', 0.0)),
                    'validation_unsafe_f1': float(validation_metrics.get('unsafe_f1', 0.0)),
                    'true_unsafe': int(validation_metrics.get('true_unsafe', 0)),
                    'false_unsafe': int(validation_metrics.get('false_unsafe', 0)),
                    'true_safe': int(validation_metrics.get('true_safe', 0)),
                    'false_safe': int(validation_metrics.get('false_safe', 0)),
                    'loss_curve': os.path.join(fold_dir, 'loss_curve.csv'),
                }
                fold_rows.append(row)
                with open(os.path.join(fold_dir, 'validation_metrics.json'), 'w', encoding='ascii') as handle:
                    json.dump({**row, 'validation_details': validation_metrics}, handle, indent=2)
                if writer is not None:
                    step = len(fold_rows)
                    writer.add_scalar('smann_grid/validation_loss', row['validation_loss'], step)
                    writer.add_scalar('smann_grid/validation_accuracy', row['validation_accuracy'], step)
                    writer.add_scalar('smann_grid/validation_unsafe_f1', row['validation_unsafe_f1'], step)

    fold_csv = os.path.join(root_dir, 'grid_fold_results.csv')
    _write_rows(fold_csv, fold_rows)
    summaries = _summarize_grid(fold_rows)
    summary_csv = os.path.join(root_dir, 'grid_summary.csv')
    _write_rows(summary_csv, summaries)
    selected = _select_hyperparameters(summaries)
    selected_path = os.path.join(root_dir, 'selected_hyperparameters.json')
    with open(selected_path, 'w', encoding='ascii') as handle:
        json.dump(selected, handle, indent=2)

    final_checkpoint_dir = args.final_checkpoint_dir or os.path.join(root_dir, 'final_selected', 'weights')
    final_adapter = SMANNAdapter(
        checkpoint_path='',
        repo_path=args.fear_repo_path,
        image_size=args.image_size,
        lookback=args.lookback,
        fear_threshold=args.fear_threshold,
    )
    final_adapter.load(logger)
    final_metrics = final_adapter.train_supervised_dataset(
        observations=observations,
        class_numbers=class_numbers,
        logger=logger,
        epochs=int(selected['epochs']),
        batch_size=args.batch_size,
        learning_rate=float(selected['learning_rate']),
    )
    final_evaluation = final_adapter.evaluate_supervised_dataset(
        observations=observations,
        class_numbers=class_numbers,
        logger=logger,
        batch_size=args.batch_size,
    )
    final_adapter.save_checkpoint(final_checkpoint_dir, logger)
    final_loss_curve = os.path.join(final_checkpoint_dir, 'loss_curve.csv')
    _write_loss_curve(final_loss_curve, [float(value) for value in final_metrics.get('epoch_losses', [])], [])
    final_metrics_path = os.path.join(final_checkpoint_dir, 'training_metrics.json')
    with open(final_metrics_path, 'w', encoding='ascii') as handle:
        json.dump(
            {
                **final_metrics,
                'selected_hyperparameters': selected,
                'dataset_metadata': metadata,
                'final_evaluation': final_evaluation,
            },
            handle,
            indent=2,
        )
    final_evaluation_path = os.path.join(final_checkpoint_dir, 'final_evaluation.json')
    with open(final_evaluation_path, 'w', encoding='ascii') as handle:
        json.dump(final_evaluation, handle, indent=2)
    plot_outputs = _plot_grid_outputs(root_dir, summaries, final_metrics, final_evaluation)

    if writer is not None:
        writer.flush()
        writer.close()

    result = {
        'dataset_dir': args.dataset_dir,
        'output_dir': root_dir,
        'fold_results_csv': fold_csv,
        'grid_summary_csv': summary_csv,
        'selected_hyperparameters_path': selected_path,
        'selected_hyperparameters': selected,
        'final_checkpoint_dir': final_checkpoint_dir,
        'final_loss_curve': final_loss_curve,
        'final_metrics_path': final_metrics_path,
        'final_evaluation_path': final_evaluation_path,
        'plot_outputs': plot_outputs,
    }
    result_path = os.path.join(root_dir, 'grid_run_summary.json')
    with open(result_path, 'w', encoding='ascii') as handle:
        json.dump(result, handle, indent=2)
    result['summary_path'] = result_path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the SMANN hyperparameter grid and train the selected checkpoint.')
    parser.add_argument('--dataset-dir', default='/workspaces/clearpath_docker/clearpath_ws/logs/manual_dataset')
    parser.add_argument('--fear-repo-path', default='/workspaces/clearpath_docker/Behavior-Intrinsic-Fear-main/CarRacingTesting')
    parser.add_argument('--output-dir', default='/workspaces/clearpath_docker/clearpath_ws/logs/smann_training/smann_grid')
    parser.add_argument('--final-checkpoint-dir', default='')
    parser.add_argument('--tensorboard-log-dir', default='/workspaces/clearpath_docker/clearpath_ws/logs/tensorboard')
    parser.add_argument('--run-name', default='smann_grid_manual_63')
    parser.add_argument('--epochs', default='250,500,1000')
    parser.add_argument('--learning-rates', default='3e-5,1e-4,3e-4')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lookback', type=int, default=3)
    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--fear-threshold', type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
    summary = run_grid(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
