# This imports future annotations so forward type hints stay simple.
from __future__ import annotations

# This imports deque so we can maintain a short lookback window.
from collections import deque

# This imports numpy for simple scoring and clipping math.
import numpy as np

# This imports the SMANN adapter.
from jackal_smann_eval.smann import SMANNModel
# This imports the shared action, config, and observation containers.
from jackal_smann_eval.types import AgentAction, AgentConfig, ObservationBundle
# This imports the vision helpers used by the heuristic controller.
from jackal_smann_eval.vision_utils import ImageDecodingError, compute_green_goal_offset, depth_image_to_numpy, rgb_image_to_numpy, summarize_depth_sectors


# This owns the frozen SMANN model and the simple evaluation controller.
class Agent:
    """Choose actions, compute fear scores, and log external versus intrinsic reward totals."""

    # This builds the agent with the chosen config and SMANN model.
    def __init__(self, config: AgentConfig, logger) -> None:
        """Store the config, create the SMANN model, and prepare the short observation memory."""

        # This stores the config.
        self.config = config
        # This stores the logger.
        self.logger = logger
        # This creates the short observation window.
        self.observation_window = deque(maxlen=self.config.lookback)
        # This builds the SMANN adapter.
        self.smann = SMANNModel(self.config.smann_checkpoint, self.config.fear_repo_path, self.config.smann_image_size, self.config.lookback, self.config.smann_fear_threshold)
        # This stores the previous goal coverage for external reward shaping.
        self._previous_goal_coverage = 0.0
        # This stores the last fear score.
        self.last_fear_score = 0.0

    # This loads the frozen SMANN weights.
    def init(self) -> None:
        """Load the offline-trained SMANN checkpoint for frozen evaluation."""

        # This loads the model weights.
        self.smann.load(self.logger)

    # This resets the short memory and reward history between episodes.
    def reset(self) -> None:
        """Clear short-term state so a new episode starts with a fresh history window."""

        # This clears the recent observation window.
        self.observation_window.clear()
        # This resets the previous goal coverage.
        self._previous_goal_coverage = 0.0
        # This resets the last fear score.
        self.last_fear_score = 0.0

    # This updates the active fear threshold without reloading the model.
    def set_fear_threshold(self, fear_threshold: float) -> None:
        """Store a new fear threshold so threshold sweeps change one clear knob."""

        # This stores the new threshold in the config.
        self.config.smann_fear_threshold = float(fear_threshold)
        # This forwards the threshold to the SMANN adapter.
        self.smann.set_threshold(float(fear_threshold))

    # This appends one observation to the short lookback memory.
    def cache_observation(self, observation: ObservationBundle) -> None:
        """Append the latest observation so the SMANN model sees the recent sequence."""

        # This appends the latest observation.
        self.observation_window.append(observation)

    # This extracts lightweight control features from the current observation.
    def _analyze_observation(self, observation: ObservationBundle) -> dict[str, float]:
        """Turn RGB and depth into goal offset and coarse obstacle distance summaries."""

        # This builds the default analysis summary.
        analysis = {'goal_offset': 0.0, 'left_depth': 1.0, 'center_depth': 1.0, 'right_depth': 1.0, 'minimum_depth': 1.0}

        # This extracts the goal offset when a color frame exists.
        if observation.color_msg is not None:
            try:
                # This decodes the RGB image.
                rgb_image = rgb_image_to_numpy(observation.color_msg)
                # This computes the goal centroid offset.
                analysis['goal_offset'] = float(compute_green_goal_offset(rgb_image))
            except ImageDecodingError:
                # This ignores transient color decoding errors.
                pass

        # This extracts coarse depth sectors when a depth frame exists.
        if observation.depth_msg is not None:
            try:
                # This decodes the depth image.
                depth_image = depth_image_to_numpy(observation.depth_msg)
                # This summarizes the depth image into coarse sectors.
                depth_summary = summarize_depth_sectors(depth_image)
                # This stores the left depth summary.
                analysis['left_depth'] = float(depth_summary['left'])
                # This stores the center depth summary.
                analysis['center_depth'] = float(depth_summary['center'])
                # This stores the right depth summary.
                analysis['right_depth'] = float(depth_summary['right'])
                # This stores the minimum depth summary.
                analysis['minimum_depth'] = float(depth_summary['minimum'])
            except ImageDecodingError:
                # This ignores transient depth decoding errors.
                pass

        # This returns the final analysis summary.
        return analysis

    # This computes the fear score for the current short sequence.
    def score_fear(self, observation: ObservationBundle) -> float:
        """Score the candidate lookback window and return the raw SMANN fear probability."""

        # The evaluator already cached the current observation before act().
        # Score the actual lookback window without duplicating the newest frame.
        fear_score = float(self.smann.score(list(self.observation_window), self.logger))
        self.last_fear_score = fear_score
        return fear_score

    # This chooses the next action according to the selected evaluation mode.
    def act(self, observation: ObservationBundle) -> AgentAction:
        """Choose a simple action for external-only, intrinsic-only, or combined evaluation."""

        # This returns a stop action for terminal observations.
        if observation.goal_reached or observation.terminal or observation.truncated:
            # This returns a zero action.
            return AgentAction()

        # This analyzes the current RGB and depth readings.
        analysis = self._analyze_observation(observation)
        # This scores the candidate fear window.
        fear_score = self.score_fear(observation)
        # This checks whether the fear threshold is active.
        fear_active = fear_score >= self.config.smann_fear_threshold

        # This runs the pure external baseline when requested.
        if self.config.reward_mode == 'external_only':
            # This returns the external-only action.
            return self._goal_action(observation, analysis)

        # This runs the pure fear baseline when requested.
        if self.config.reward_mode == 'intrinsic_only':
            # This returns the intrinsic-only action.
            return self._fear_action(analysis, fear_active)

        # This gives fear priority in the combined mode.
        if fear_active:
            # This returns the fear-reactive action.
            return self._fear_action(analysis, True)

        # This returns the normal goal-seeking action when fear is not active.
        return self._goal_action(observation, analysis)

    # This computes the simple goal-seeking baseline action.
    def _goal_action(self, observation: ObservationBundle, analysis: dict[str, float]) -> AgentAction:
        """Drive toward the goal while lightly steering around obvious forward obstacles."""

        # This stops when the goal already fills enough of the image.
        if observation.goal_reached:
            # This returns a stop action.
            return AgentAction()

        # This turns away from very close obstacles.
        if analysis['center_depth'] < 0.18 or analysis['minimum_depth'] < 0.12:
            # This picks the safer turn direction.
            turn_sign = 1.0 if analysis['left_depth'] > analysis['right_depth'] else -1.0
            # This returns a cautious turn action.
            return AgentAction(linear_x=0.05, angular_z=turn_sign * self.config.action_turn_speed)

        # This steers toward the visible goal when it is in frame.
        if observation.goal_coverage > 0.01:
            # This clips the goal offset into the turn range.
            angular = float(np.clip(analysis['goal_offset'] * self.config.action_turn_speed, -self.config.action_turn_speed, self.config.action_turn_speed))
            # This returns a goal-following action.
            return AgentAction(linear_x=self.config.action_linear_speed, angular_z=angular)

        # This returns a forward exploration action when the goal is not visible yet.
        return AgentAction(linear_x=self.config.action_linear_speed, angular_z=0.0)

    # This computes the fear-reactive baseline action.
    def _fear_action(self, analysis: dict[str, float], fear_active: bool) -> AgentAction:
        """Choose a cautious action that turns away when fear or a close obstacle is active."""

        # This turns away from the more blocked side when fear is active.
        if fear_active or analysis['center_depth'] < 0.25:
            # This picks the safer turn direction.
            turn_sign = 1.0 if analysis['left_depth'] > analysis['right_depth'] else -1.0
            # This returns a cautious fear-driven action.
            return AgentAction(linear_x=self.config.fear_linear_speed, angular_z=turn_sign * self.config.fear_turn_speed)

        # This returns a slow forward action when fear is low.
        return AgentAction(linear_x=self.config.fear_linear_speed, angular_z=0.0)

    # This computes external, intrinsic, and combined rewards for logging.
    def compute_rewards(self, observation: ObservationBundle) -> tuple[float, float, float]:
        """Compute task reward, fear penalty, and their sum for evaluation summaries."""

        # This computes the change in goal coverage.
        goal_progress = float(np.clip(observation.goal_coverage - self._previous_goal_coverage, -0.25, 0.25))
        # This starts the external reward with a small step cost.
        external_reward = -0.01
        # This adds reward for goal progress.
        external_reward += 5.0 * goal_progress

        # This adds a success bonus when the goal is reached.
        if observation.goal_reached:
            # This adds the success bonus.
            external_reward += 2.0

        # This adds a terminal penalty for collisions.
        if observation.collision:
            # This adds the collision penalty.
            external_reward -= 1.0

        # This computes the intrinsic fear penalty.
        intrinsic_reward = -float(self.last_fear_score)
        # This computes the combined reward.
        combined_reward = float(external_reward + intrinsic_reward)
        # This stores the latest goal coverage for the next step.
        self._previous_goal_coverage = float(observation.goal_coverage)
        # This returns the three reward values.
        return float(external_reward), float(intrinsic_reward), float(combined_reward)
