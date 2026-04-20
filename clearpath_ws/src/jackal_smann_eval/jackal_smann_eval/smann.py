# This imports future annotations so forward type hints stay simple.
from __future__ import annotations

# This imports importlib so we can load Rodney's model files dynamically.
import importlib
# This imports os so we can check checkpoint and repo paths.
import os
# This imports sys so we can add the Behavior-Intrinsic-Fear repo to the import path.
import sys
# This imports Sequence for the observation window type hints.
from typing import Sequence

# This imports numpy for the lookback window assembly.
import numpy as np

# This imports the observation container used by the adapter.
from jackal_smann_eval.types import ObservationBundle
# This imports the RGB-D formatter used for live SMANN inputs.
from jackal_smann_eval.vision_utils import ImageDecodingError, format_rgbd_timestep

try:
    # This imports torch for checkpoint loading and inference.
    import torch
    # This imports softmax for turning logits into probabilities.
    import torch.nn.functional as functional
except Exception:
    # This stores a safe fallback when torch is unavailable.
    torch = None
    # This stores a safe fallback when torch is unavailable.
    functional = None


# This adapts Rodney's offline-trained SMANN model to the Jackal runtime.
class SMANNModel:
    """Load the trained SMANN checkpoint and score live Jackal RGB-D windows."""

    # This builds the adapter with the chosen checkpoint, repo path, and threshold.
    def __init__(self, checkpoint_path: str, repo_path: str, image_size: int, lookback: int, fear_threshold: float) -> None:
        """Store the offline checkpoint settings and prepare empty model state."""

        # This stores the checkpoint path.
        self.checkpoint_path = checkpoint_path
        # This stores the Behavior-Intrinsic-Fear source path.
        self.repo_path = repo_path
        # This stores the square image size used during training.
        self.image_size = int(image_size)
        # This stores the sequence length used during training.
        self.lookback = int(lookback)
        # This stores the active fear threshold.
        self.fear_threshold = float(fear_threshold)
        # This tracks whether the checkpoint loaded successfully.
        self.loaded = False
        # This stores the imported model class.
        self._model_class = None
        # This stores the live model instance.
        self._model = None

    # This imports Rodney's model code and loads the trained weights.
    def load(self, logger) -> None:
        """Import the upstream model, build it, and restore the trained checkpoint."""

        # This stops early when torch is not available.
        if torch is None or functional is None:
            # This reports the missing torch dependency.
            logger.warning('PyTorch is unavailable, so SMANN fear scoring is disabled.')
            # This returns without loading a model.
            return

        # This stops early when the repo path does not exist.
        if not os.path.isdir(self.repo_path):
            # This reports the missing repo path.
            logger.warning(f'SMANN repo path was not found: {self.repo_path}')
            # This returns without loading a model.
            return

        # This stops early when the checkpoint path does not exist.
        if not os.path.isdir(self.checkpoint_path):
            # This reports the missing checkpoint path.
            logger.warning(f'SMANN checkpoint path was not found: {self.checkpoint_path}')
            # This returns without loading a model.
            return

        # This adds Rodney's repo to the import path when needed.
        if self.repo_path not in sys.path:
            # This inserts the repo path first so the correct modules are imported.
            sys.path.insert(0, self.repo_path)

        try:
            # This clears cached copies of Rodney's model modules.
            for module_name in ('aio_complex', 'ntm_complex', 'complexcontroller', 'head', 'memory'):
                # This removes one cached module if it exists.
                sys.modules.pop(module_name, None)

            # This refreshes Python's import cache.
            importlib.invalidate_caches()
            # This imports Rodney's model wrapper.
            module = importlib.import_module('aio_complex')
            # This stores the model class used by the checkpoint.
            self._model_class = getattr(module, 'EncapsulatedNTM')
            # This builds the model with the Jackal RGB-D shape.
            self._model = self._model_class([4, self.image_size, self.image_size], 2, 1, controller_size=250, controller_layers=7, num_read_heads=30, num_write_heads=30, N=128, M=60)
            # This loads the decision layer weights.
            self._model.ntm.fc.load_state_dict(torch.load(os.path.join(self.checkpoint_path, 'decision_layer.pth'), map_location='cpu'))
            # This loads the controller weights.
            self._model.ntm.controller.complexlstm.load_state_dict(torch.load(os.path.join(self.checkpoint_path, 'controller_weights.pth'), map_location='cpu'))
            # This loads the read and write heads.
            self._model.ntm.heads.load_state_dict(torch.load(os.path.join(self.checkpoint_path, 'heads.pth'), map_location='cpu'))

            # This restores the memory state when it is available.
            if os.path.exists(os.path.join(self.checkpoint_path, 'memory.pth')):
                # This loads the stored external memory contents.
                self._model.memory.load_memory(os.path.join(self.checkpoint_path, ''))

            # This switches the model into inference mode.
            self._model.eval()
            # This marks the adapter as ready.
            self.loaded = True
            # This reports the successful checkpoint load.
            logger.info(f'SMANN checkpoint loaded from {self.checkpoint_path}.')
        except Exception as exc:
            # This reports the checkpoint load failure.
            logger.warning(f'Failed to load SMANN checkpoint: {exc}')

    # This updates the active threshold without reloading the whole model.
    def set_threshold(self, fear_threshold: float) -> None:
        """Store a new fear threshold so sweeps can change one parameter cleanly."""

        # This stores the new threshold value.
        self.fear_threshold = float(fear_threshold)

    # This converts the recent observation history into the SMANN input layout.
    def prepare_window(self, observation_window: Sequence[ObservationBundle], logger):
        """Convert the recent lookback window into one [lookback, 4, H, W] array."""

        # This returns early until enough frames exist.
        if len(observation_window) < self.lookback:
            # This returns no prepared window yet.
            return None

        # This keeps only the latest lookback frames.
        recent_window = observation_window[-self.lookback:]
        # This collects one RGB-D timestep at a time.
        frames = []

        # This converts each observation into a 4-channel timestep.
        for observation in recent_window:
            # This stops when a required image is missing.
            if observation.color_msg is None or observation.depth_msg is None:
                # This returns no prepared window yet.
                return None

            try:
                # This formats one timestep as [4, H, W].
                timestep = format_rgbd_timestep(observation.color_msg, observation.depth_msg, height=self.image_size, width=self.image_size)
            except ImageDecodingError as exc:
                # This reports the image decoding issue.
                logger.warning(str(exc))
                # This returns no prepared window for this step.
                return None

            # This appends the timestep to the sequence.
            frames.append(timestep)

        # This stacks the timesteps into [lookback, 4, H, W].
        return np.stack(frames, axis=0)

    # This scores one prepared Jackal window with the loaded SMANN model.
    def score_prepared(self, prepared_window, logger) -> float:
        """Run inference on one prepared lookback window and return the fear probability."""

        # This returns zero when the input window is not ready.
        if prepared_window is None:
            # This returns no fear signal yet.
            return 0.0

        # This returns zero when the checkpoint was not loaded.
        if not self.loaded or self._model is None or torch is None or functional is None:
            # This returns no fear signal.
            return 0.0

        try:
            # This disables gradients during live inference.
            with torch.no_grad():
                # This allocates the two-sample batch Rodney's forward pass expects.
                batch = np.zeros((2, self.lookback, 4, self.image_size, self.image_size), dtype=np.float32)
                # This inserts the live sequence into the first batch slot.
                batch[0] = prepared_window.astype(np.float32) / 255.0
                # This reorders the tensor into [lookback, batch, channels, H, W].
                inputs = torch.from_numpy(batch).permute(1, 0, 2, 3, 4).contiguous()
                # This builds the delimiter tensor expected by Rodney's model.
                delimiter = torch.zeros((2, 2), dtype=torch.float32)
                # This resets the recurrent sequence state.
                self._model.init_sequence(2)
                # This runs the model forward pass.
                logits, _ = self._model(x=inputs, delimeter=delimiter, previous_state=None, seq=self.lookback)
                # This converts logits into class probabilities.
                probabilities = functional.softmax(logits[0], dim=-1)
                # This reads the danger class probability.
                return float(probabilities[0].item())
        except Exception as exc:
            # This reports the live inference failure.
            logger.warning(f'SMANN inference failed: {exc}')
            # This returns a safe fallback score.
            return 0.0

    # This scores the latest live observation history directly.
    def score(self, observation_window: Sequence[ObservationBundle], logger) -> float:
        """Prepare the live window and return the raw danger probability."""

        # This prepares the live lookback window.
        prepared_window = self.prepare_window(observation_window, logger)
        # This returns the danger score for that prepared window.
        return self.score_prepared(prepared_window, logger)
