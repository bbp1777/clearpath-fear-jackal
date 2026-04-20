"""
Adapter around the Behavior-Intrinsic-Fear repository so the Jackal stack can reuse Rodney
Sanchez's sequence model.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Sequence

import numpy as np

from fear_jackal_sim.rl_types import ObservationBundle, Transition
from fear_jackal_sim.vision_utils import ImageDecodingError, format_rgbd_timestep

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn, optim
except Exception:
    torch = None
    functional = None
    nn = None
    optim = None


class SMANNAdapter:
    """Adapter around the Behavior-Intrinsic-Fear sequence model.

    The upstream repository ships a complex memory-augmented model geared toward
    RGB lookback windows. This adapter keeps our ROS-facing interface stable,
    handles repo imports inside Docker, and lets us score, fine-tune, and export
    the Sanchez fear model using replay windows captured from the simulator.
    """

    def __init__(
        self,
        checkpoint_path: str = '',
        repo_path: str = '/workspaces/Behavior-Intrinsic-Fear-main/CarRacingTesting',
        image_size: int = 84,
        lookback: int = 3,
        fear_threshold: float = 0.5,
    ) -> None:
        """
        Configure checkpoint paths, repo import settings, and sequence-model
        hyperparameters.
        """
        self.checkpoint_path = checkpoint_path
        self.repo_path = repo_path
        self.image_size = int(image_size)
        self.lookback = int(lookback)
        self.fear_threshold = float(fear_threshold)

        self.loaded = False
        self._trained_batches = 0
        self._repo_import_error = ''
        self._model_class = None
        self._model = None
        self._optimizer = None
        self._criterion = nn.CrossEntropyLoss() if nn is not None else None
        self.last_raw_score = 0.0
        self.last_thresholded_score = 0.0
        self.last_fear_active = False

    def load(self, logger) -> None:
        """
        Import the upstream model and optionally load a checkpoint from disk.
        """
        if torch is None or nn is None or optim is None:
            logger.warning('PyTorch is unavailable; intrinsic fear defaults to 0.0.')
            return

        if not self._ensure_model(logger):
            logger.warning('SMANN model import failed; intrinsic fear defaults to 0.0.')
            return

        if not self.checkpoint_path:
            logger.info(
                'SMANN checkpoint not provided; the Sanchez fear model architecture is ready, '
                'but intrinsic fear will stay at 0.0 until a checkpoint is supplied or '
                'vicarious conditioning has trained it.'
            )
            return

        prefix = self._resolve_checkpoint_prefix(self.checkpoint_path)
        if prefix is None:
            logger.warning(
                f'SMANN checkpoint path {self.checkpoint_path} was not found; intrinsic fear defaults to 0.0.'
            )
            return

        try:
            # This loads the SMANN decision layer from the offline Jackal training run.
            self._model.ntm.fc.load_state_dict(torch.load(prefix + 'decision_layer.pth', map_location='cpu'))
            # This loads the recurrent controller weights from the offline Jackal training run.
            self._model.ntm.controller.complexlstm.load_state_dict(
                torch.load(prefix + 'controller_weights.pth', map_location='cpu')
            )
            # This loads the read and write heads from the offline Jackal training run.
            self._model.ntm.heads.load_state_dict(torch.load(prefix + 'heads.pth', map_location='cpu'))
            # This reloads the external memory contents when the checkpoint includes them.
            if os.path.exists(prefix + 'memory.pth'):
                # This restores the trained memory state so live evaluation matches offline training more closely.
                self._model.memory.load_memory(prefix)
            # This marks the adapter as ready to score live Jackal windows.
            self.loaded = True
            # This log makes it obvious which checkpoint directory is active for the run.
            logger.info(f'SMANN checkpoint loaded from {prefix}.')
        except FileNotFoundError as exc:
            logger.warning(f'SMANN checkpoint is incomplete: {exc}. Intrinsic fear defaults to 0.0.')
        except Exception as exc:
            logger.warning(f'Failed to load SMANN checkpoint from {prefix}: {exc}')

    def prepare_window(self, observation_window: Sequence[ObservationBundle], logger):
        """
        Convert the recent RGB-D observation history into the fixed [look_back, 4, H, W]
        tensor layout expected by the Jackal SMANN model.
        """
        # This guard waits until the full lookback window is available.
        if len(observation_window) < self.lookback:
            return None

        # This trims the window to the latest frames the model was trained on.
        recent_window = observation_window[-self.lookback:]
        # This list collects one [4, H, W] Jackal RGB-D frame at a time.
        frames = []
        # This loop converts each ROS observation into the exact model input layout.
        for observation in recent_window:
            # This guard keeps live inference aligned with the RGB-D training setup.
            if observation.color_msg is None or observation.depth_msg is None:
                return None

            try:
                # This merges one RGB frame and one depth frame into one [4, H, W] timestep.
                rgbd_timestep = format_rgbd_timestep(
                    observation.color_msg,
                    observation.depth_msg,
                    height=self.image_size,
                    width=self.image_size,
                )
            except ImageDecodingError as exc:
                # This log explains why the current live window could not be scored.
                logger.warning(str(exc))
                return None

            # This appends the converted timestep to the short sequence window.
            frames.append(rgbd_timestep)

        # This stacks the short RGB-D history into [look_back, 4, H, W].
        return np.stack(frames, axis=0)

    def score(self, observation_window: Sequence[ObservationBundle], logger) -> float:
        """
        Score the current observation history after preparing the lookback window.
        """
        prepared_window = self.prepare_window(observation_window, logger)
        return self.score_prepared(prepared_window, logger)

    def score_prepared(self, prepared_window, logger) -> float:
        """
        Run SMANN inference and return the thresholded Sanchez fear activation.
        """
        unsafe_probability = self.predict_unsafe_probability(prepared_window, logger)
        if unsafe_probability < self.fear_threshold:
            self.last_thresholded_score = 0.0
            self.last_fear_active = False
            return 0.0

        self.last_thresholded_score = float(unsafe_probability)
        self.last_fear_active = True
        return float(unsafe_probability)

    def compute_thresholded_intrinsic_reward(self, prepared_window, logger) -> float:
        """
        Return the Sanchez-style intrinsic penalty: 0 below threshold, -P(unsafe) above it.
        """
        return -float(self.score_prepared(prepared_window, logger))

    def predict_unsafe_probability(self, prepared_window, logger) -> float:
        """
        Run SMANN inference on a prepared lookback window and return raw P(unsafe).
        """
        if prepared_window is None:
            self.last_raw_score = 0.0
            self.last_thresholded_score = 0.0
            self.last_fear_active = False
            return 0.0

        if not self._can_infer():
            self.last_raw_score = 0.0
            self.last_thresholded_score = 0.0
            self.last_fear_active = False
            return 0.0

        self._model.eval()
        try:
            with torch.no_grad():
                inputs = self._prepared_window_to_model_input(prepared_window)
                delimiter = torch.zeros((2, 2), dtype=torch.float32)
                self._model.init_sequence(2)
                logits, _ = self._model(
                    x=inputs,
                    delimeter=delimiter,
                    previous_state=None,
                    seq=self.lookback,
                )
                probabilities = functional.softmax(logits[0], dim=-1)
                unsafe_probability = float(probabilities[0].item())
                self.last_raw_score = unsafe_probability
                return unsafe_probability
        except Exception as exc:
            self.last_raw_score = 0.0
            self.last_thresholded_score = 0.0
            self.last_fear_active = False
            logger.warning(f'SMANN inference failed; returning zero intrinsic fear for this step: {exc}')
            return 0.0

    def train_with_vicarious_conditioning(
        self,
        replay_buffer: Sequence[Transition],
        logger,
    ) -> dict[str, float | int | bool]:
        """
        Fine-tune the SMANN model online from labeled replay windows.
        """
        metrics: dict[str, float | int | bool] = {
            'trained': False,
            'samples': 0,
            'loss': 0.0,
        }

        if torch is None or nn is None or optim is None:
            return metrics

        if not self._ensure_model(logger):
            return metrics

        eligible = []
        for transition in replay_buffer:
            if transition.vision_window is None or transition.danger_label is None:
                continue
            fear_class = 0 if bool(transition.danger_label) else 1
            eligible.append((transition.vision_window, fear_class))

        if len(eligible) < 2:
            logger.info(
                f'SMANN vicarious-conditioning skipped; only {len(eligible)} labeled RGB windows are available.'
            )
            return metrics

        eligible = eligible[-64:]
        if len(eligible) % 2 == 1:
            eligible = eligible[:-1]

        total_loss = 0.0
        batches = 0
        self._model.train()
        self._ensure_optimizer(1e-4)

        try:
            for index in range(0, len(eligible), 2):
                batch_windows = np.stack((eligible[index][0], eligible[index + 1][0]), axis=0)
                batch_windows = batch_windows.astype(np.float32) / 255.0
                inputs = torch.from_numpy(batch_windows).permute(1, 0, 2, 3, 4).contiguous()
                labels = torch.tensor([eligible[index][1], eligible[index + 1][1]], dtype=torch.long)
                delimiter = torch.zeros((2, 2), dtype=torch.float32)

                self._model.init_sequence(2)
                logits, _ = self._model(
                    x=inputs,
                    delimeter=delimiter,
                    previous_state=None,
                    seq=self.lookback,
                )
                loss = self._criterion(logits, labels)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_loss += float(loss.item())
                batches += 1
        except Exception as exc:
            logger.warning(
                'SMANN vicarious-conditioning failed; leaving the current model weights unchanged for now: '
                f'{exc}'
            )
            return metrics

        if batches == 0:
            return metrics

        self._trained_batches += batches
        self.loaded = True
        mean_loss = total_loss / float(batches)
        metrics['trained'] = True
        metrics['samples'] = len(eligible)
        metrics['loss'] = mean_loss
        logger.info(
            f'SMANN vicarious-conditioning updated on {len(eligible)} RGB windows with mean loss {mean_loss:.4f}.'
        )
        return metrics

    def train_supervised_dataset(
        self,
        observations: np.ndarray,
        class_numbers: np.ndarray,
        logger,
        epochs: int = 20,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
    ) -> dict[str, float | int | bool]:
        """
        Train the SMANN model offline on exported Jackal windows.
        """
        metrics: dict[str, float | int | bool] = {
            'trained': False,
            'samples': int(len(observations)) if observations is not None else 0,
            'epochs': int(max(epochs, 0)),
            'batches': 0,
            'loss': 0.0,
        }

        if torch is None or nn is None or optim is None:
            logger.warning('PyTorch is unavailable; supervised SMANN training cannot run.')
            return metrics

        if not self._ensure_model(logger):
            return metrics

        if observations is None or class_numbers is None:
            logger.warning('No SMANN dataset was provided for supervised training.')
            return metrics

        observations = np.asarray(observations, dtype=np.uint8)
        class_numbers = np.asarray(class_numbers, dtype=np.int64)
        if observations.ndim != 5:
            logger.warning(
                f'Expected observations with shape [N, lookback, 4, H, W], but got {observations.shape}.'
            )
            return metrics

        if len(observations) < 2:
            logger.warning('At least two labeled windows are required to train the fear model.')
            return metrics

        self._ensure_optimizer(float(learning_rate))
        total_loss = 0.0
        total_batches = 0
        effective_batch_size = max(int(batch_size), 1)
        epochs = max(int(epochs), 1)
        rng = np.random.default_rng(0)
        self._model.train()

        try:
            for epoch_index in range(epochs):
                indices = np.arange(len(observations))
                rng.shuffle(indices)
                epoch_loss = 0.0
                epoch_batches = 0

                for start in range(0, len(indices), effective_batch_size):
                    batch_indices = indices[start:start + effective_batch_size]
                    if len(batch_indices) == 0:
                        continue

                    batch_windows = observations[batch_indices].astype(np.float32) / 255.0
                    inputs = torch.from_numpy(batch_windows).permute(1, 0, 2, 3, 4).contiguous()
                    labels = torch.from_numpy(class_numbers[batch_indices].astype(np.int64))
                    delimiter = torch.zeros((len(batch_indices), 2), dtype=torch.float32)

                    self._model.init_sequence(len(batch_indices))
                    logits, _ = self._model(
                        x=inputs,
                        delimeter=delimiter,
                        previous_state=None,
                        seq=self.lookback,
                    )
                    loss = self._criterion(logits, labels)

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    epoch_loss += float(loss.item())
                    epoch_batches += 1

                if epoch_batches == 0:
                    continue

                epoch_mean_loss = epoch_loss / float(epoch_batches)
                logger.info(
                    f'SMANN supervised epoch {epoch_index + 1}/{epochs} mean loss {epoch_mean_loss:.4f}.'
                )
                total_loss += epoch_loss
                total_batches += epoch_batches
        except Exception as exc:
            logger.warning(f'SMANN supervised training failed; keeping the current weights: {exc}')
            return metrics

        if total_batches == 0:
            return metrics

        self._trained_batches += total_batches
        self.loaded = True
        metrics['trained'] = True
        metrics['batches'] = total_batches
        metrics['loss'] = total_loss / float(total_batches)
        return metrics

    def save_checkpoint(self, checkpoint_dir: str, logger) -> bool:
        """
        Export the model weights in the file layout expected by the Sanchez repository.
        """
        if torch is None or self._model is None:
            logger.warning('SMANN checkpoint export skipped because the fear model is not initialized.')
            return False

        os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            # This saves the SMANN decision layer using Rodney's checkpoint naming.
            torch.save(self._model.ntm.fc.state_dict(), os.path.join(checkpoint_dir, 'decision_layer.pth'))
            # This saves the recurrent controller using Rodney's checkpoint naming.
            torch.save(
                self._model.ntm.controller.complexlstm.state_dict(),
                os.path.join(checkpoint_dir, 'controller_weights.pth'),
            )
            # This saves the read and write heads using Rodney's checkpoint naming.
            torch.save(self._model.ntm.heads.state_dict(), os.path.join(checkpoint_dir, 'heads.pth'))
            # This saves the external memory contents so live evaluation can restore them.
            self._model.memory.save_memory(os.path.join(checkpoint_dir, ''))
            # This metadata makes it easy to verify the RGB-D live settings later.
            metadata = {
                'image_size': int(self.image_size),
                'lookback': int(self.lookback),
                'channels': 4,
                'fear_threshold': float(self.fear_threshold),
                'trained_batches': int(self._trained_batches),
                'loaded': bool(self.loaded),
            }
            with open(os.path.join(checkpoint_dir, 'smann_metadata.json'), 'w', encoding='ascii') as handle:
                json.dump(metadata, handle, indent=2)
            logger.info(f'SMANN checkpoint exported to {checkpoint_dir}.')
            return True
        except Exception as exc:
            logger.warning(f'Failed to export SMANN checkpoint to {checkpoint_dir}: {exc}')
            return False

    def _ensure_model(self, logger) -> bool:
        # This returns early when the live SMANN model is already built.
        if self._model is not None:
            return True

        # This imports Rodney's CarRacingTesting model code before we build the network.
        if not self._import_repo_model(logger):
            return False

        # This creates the Jackal RGB-D model with the exact Sanchez hyperparameters used offline.
        self._model = self._model_class(
            [4, self.image_size, self.image_size],
            2,
            1,
            controller_size=250,
            controller_layers=7,
            num_read_heads=30,
            num_write_heads=30,
            N=128,
            M=60,
        )
        # This prepares the optimizer so offline retraining still works when requested.
        self._optimizer = optim.Adam(self._model.parameters(), lr=7e-5)
        return True

    def _import_repo_model(self, logger) -> bool:
        # This returns early when the repo model class is already cached.
        if self._model_class is not None:
            return True

        # This checks that the chosen Behavior-Intrinsic-Fear source directory exists.
        if not os.path.isdir(self.repo_path):
            self._repo_import_error = (
                f'Fear model repository path {self.repo_path} is not available inside the container.'
            )
            logger.warning(self._repo_import_error)
            return False

        # This puts the chosen Sanchez source directory first on the import path.
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        try:
            # This clears cached Sanchez modules so CarRacingTesting wins over any older root-level import.
            for module_name in ('aio_complex', 'ntm_complex', 'complexcontroller', 'head', 'memory'):
                # This removes the cached module before we import from the selected repo path again.
                sys.modules.pop(module_name, None)
            # This tells Python to rescan the repo path before importing the Sanchez model.
            importlib.invalidate_caches()
            # This imports the exact EncapsulatedNTM implementation used for the Jackal checkpoint.
            module = importlib.import_module('aio_complex')
            # This stores the model class so later calls do not re-import it.
            self._model_class = getattr(module, 'EncapsulatedNTM')
            # This log makes it obvious which Sanchez code path the live run is using.
            logger.info(f'Imported Behavior-Intrinsic-Fear model code from {self.repo_path}.')
            return True
        except Exception as exc:
            # This caches the import error for later debugging.
            self._repo_import_error = str(exc)
            # This log keeps the trainer alive while clearly reporting the import problem.
            logger.warning(f'Failed to import Behavior-Intrinsic-Fear modules: {exc}')
            return False

    def _can_infer(self) -> bool:
        return self._model is not None and (self.loaded or self._trained_batches > 0)

    def _ensure_optimizer(self, learning_rate: float) -> None:
        if optim is None or self._model is None:
            return
        if self._optimizer is None:
            self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
            return
        for group in self._optimizer.param_groups:
            group['lr'] = float(learning_rate)

    def _prepared_window_to_model_input(self, prepared_window: np.ndarray) -> torch.Tensor:
        # This allocates the fixed two-sample batch shape Rodney's forward pass expects.
        batch = np.zeros((2, self.lookback, 4, self.image_size, self.image_size), dtype=np.float32)
        # This places the live Jackal RGB-D sequence into the first batch slot.
        batch[0] = prepared_window.astype(np.float32) / 255.0
        # This reorders the batch into [look_back, batch, channels, H, W] for SMANN.
        return torch.from_numpy(batch).permute(1, 0, 2, 3, 4).contiguous()

    def _resolve_checkpoint_prefix(self, checkpoint_path: str) -> str | None:
        if os.path.isdir(checkpoint_path):
            return os.path.join(checkpoint_path, '')

        if os.path.isfile(checkpoint_path):
            filename = os.path.basename(checkpoint_path)
            if filename == 'decision_layer.pth':
                return os.path.join(os.path.dirname(checkpoint_path), '')

        return None

