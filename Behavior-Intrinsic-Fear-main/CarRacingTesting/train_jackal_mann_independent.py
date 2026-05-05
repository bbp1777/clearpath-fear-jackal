# Train a Jackal RGB-D SMANN with the same fresh-state call used at live inference.
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn, optim

from aio_complex import EncapsulatedNTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Jackal RGB-D MANN with independent fresh-state windows.')
    parser.add_argument('--dataset-dir', default='/workspaces/clearpath_docker/clearpath_ws/logs/rodney_dataset/')
    parser.add_argument('--output-dir', default='/workspaces/clearpath_docker/clearpath_ws/logs/rodney_training')
    parser.add_argument('--run-name', default='jackal_mann_independent')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--learning-rate', type=float, default=7.0e-5)
    parser.add_argument('--seed', type=int, default=12345)
    return parser.parse_args()


def dataset_prefix(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, 'Jackal-v0_lookback_3')


def load_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = dataset_prefix(dataset_dir)
    observations = np.load(prefix + 'observations.npy')
    labels = np.load(prefix + 'class_number.npy').astype(np.int64)
    if observations.ndim != 5:
        raise ValueError(f'Expected observations [N, lookback, channels, H, W], got {observations.shape}.')
    if observations.shape[2] != 4:
        raise ValueError(f'Expected RGB-D observations with 4 channels, got {observations.shape}.')
    if observations.shape[0] != labels.shape[0]:
        raise ValueError('Observation and label counts do not match.')
    return observations.astype(np.uint8), labels


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


def make_inputs(batch_windows: np.ndarray) -> torch.Tensor:
    batch_windows = batch_windows.astype(np.float32) / 255.0
    return torch.from_numpy(batch_windows).permute(1, 0, 2, 3, 4).contiguous()


def balanced_epoch_order(labels: np.ndarray, batch_size: int) -> np.ndarray:
    unsafe = np.where(labels == 0)[0]
    safe = np.where(labels == 1)[0]
    if unsafe.size == 0 or safe.size == 0:
        return np.random.permutation(len(labels))

    unsafe_per_batch = max(1, batch_size // 2)
    safe_per_batch = max(1, batch_size - unsafe_per_batch)
    batches = int(np.ceil(max(unsafe.size / unsafe_per_batch, safe.size / safe_per_batch)))
    order = []
    for _ in range(batches):
        batch = []
        batch.extend(np.random.choice(unsafe, unsafe_per_batch, replace=True).tolist())
        batch.extend(np.random.choice(safe, safe_per_batch, replace=True).tolist())
        np.random.shuffle(batch)
        order.extend(batch)
    return np.asarray(order, dtype=np.int64)


def train_epoch(model, optimizer, criterion, observations, labels, batch_size, lookback):
    order = balanced_epoch_order(labels, batch_size)
    total_loss = 0.0
    correct = 0
    used = 0
    for start in range(0, len(order) - batch_size + 1, batch_size):
        idx = order[start:start + batch_size]
        x = make_inputs(observations[idx])
        y = torch.tensor(labels[idx], dtype=torch.long)
        delimiter = torch.zeros((batch_size, 2), dtype=torch.float32)
        model.train()
        model.init_sequence(batch_size)
        logits, _ = model(x=x, delimeter=delimiter, previous_state=None, seq=lookback)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == y).sum().item())
        used += batch_size
    return total_loss / max(used, 1), 100.0 * correct / max(used, 1)


def score_live_style(model, window: np.ndarray) -> float:
    lookback, channels, height, width = window.shape
    batch = np.zeros((2, lookback, channels, height, width), dtype=np.float32)
    batch[0] = window.astype(np.float32) / 255.0
    inputs = torch.from_numpy(batch).permute(1, 0, 2, 3, 4).contiguous()
    delimiter = torch.zeros((2, 2), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        model.init_sequence(2)
        logits, _ = model(x=inputs, delimeter=delimiter, previous_state=None, seq=lookback)
        probabilities = torch.softmax(logits[0], dim=-1)
    return float(probabilities[0].item())


def summarize_scores(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unsafe = scores[labels == 0]
    safe = scores[labels == 1]
    return {
        'score_min': float(scores.min()),
        'score_max': float(scores.max()),
        'score_mean': float(scores.mean()),
        'unsafe_mean': float(unsafe.mean()) if unsafe.size else 0.0,
        'safe_mean': float(safe.mean()) if safe.size else 0.0,
        'unsafe_count': int(unsafe.size),
        'safe_count': int(safe.size),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    observations, labels = load_dataset(args.dataset_dir)
    lookback = int(observations.shape[1])
    channels = int(observations.shape[2])
    image_size = int(observations.shape[3])
    batch_size = min(max(2, int(args.batch_size)), int(len(labels)))
    model = build_model(channels, image_size)
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate))
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    for epoch in range(int(args.epochs)):
        loss, accuracy = train_epoch(model, optimizer, criterion, observations, labels, batch_size, lookback)
        losses.append(loss)
        accuracies.append(accuracy)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(args.epochs):
            print(f'epoch={epoch + 1} loss={loss:.6f} accuracy={accuracy:.2f}')
    scores = np.asarray([score_live_style(model, observations[index]) for index in range(len(observations))], dtype=np.float32)
    score_summary = summarize_scores(scores, labels)
    predictions = (scores < 0.5).astype(np.int64)
    live_accuracy = float((predictions == labels).mean() * 100.0)
    run_dir = os.path.join(args.output_dir, args.run_name)
    weights_dir = os.path.join(run_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    model.save(weights_dir + os.sep)
    np.save(os.path.join(run_dir, 'losses.npy'), np.asarray(losses, dtype=np.float32))
    np.save(os.path.join(run_dir, 'accuracies.npy'), np.asarray(accuracies, dtype=np.float32))
    np.save(os.path.join(run_dir, 'live_scores.npy'), scores)
    metadata = {
        'dataset_dir': args.dataset_dir,
        'batch_size': int(batch_size),
        'epochs': int(args.epochs),
        'learning_rate': float(args.learning_rate),
        'balanced_batches': True,
        'samples_loaded': int(len(labels)),
        'look_back': int(lookback),
        'channels': int(channels),
        'input_size': int(image_size),
        'final_train_accuracy': float(accuracies[-1]) if accuracies else 0.0,
        'live_style_accuracy_at_0_5': live_accuracy,
        'weights_dir': weights_dir,
        'score_summary': score_summary,
    }
    with open(os.path.join(run_dir, 'metadata.json'), 'w', encoding='ascii') as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    print(f'Saved model weights to {weights_dir}')


if __name__ == '__main__':
    main()
