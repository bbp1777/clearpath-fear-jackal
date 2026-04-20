"""
Policy, reward, and fear-integration logic for the Jackal trainer. This file adapts a
PPO rollout/update structure to the ROS2 Jackal state/action space while
keeping the offline fear hooks close to the rest of the learning loop.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from fear_jackal_sim.memory_fear import SequenceMemoryRewardModel
from fear_jackal_sim.replay_buffer import ExperienceReplayBuffer
from fear_jackal_sim.rl_types import AgentAction, AgentConfig, ObservationBundle, Transition
from fear_jackal_sim.smann import SMANNAdapter
from fear_jackal_sim.vision_utils import (
    ImageDecodingError,
    compute_green_goal_offset,
    depth_image_to_numpy,
    rgb_image_to_numpy,
    summarize_depth_sectors,
)

try:
    import torch
    from torch import nn, optim
    from torch.distributions import Categorical
    from torch.nn import functional as F
    from torch.nn.utils import clip_grad_norm_
except Exception:
    torch = None
    nn = None
    optim = None
    Categorical = None
    F = None
    clip_grad_norm_ = None


# These tiny networks are the online PPO policy/value backbone for the Jackal.
# They are not part of Rodney Sanchez's fear model; that logic lives in SMANNAdapter
# and SequenceMemoryRewardModel farther down in the agent.
if nn is not None:
    class ActorNetwork(nn.Module):
        """Small MLP actor head that maps Jackal features to action logits."""
        def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
            """Build the policy network used by PPO."""
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x):
            """Return action logits for the current feature tensor."""
            return self.net(x)


    class CriticNetwork(nn.Module):
        """Small MLP critic head that predicts state value from Jackal features."""
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            """Build the value network used by PPO."""
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            """Return the value estimate for the current feature tensor."""
            return self.net(x)
else:
    ActorNetwork = None
    CriticNetwork = None


if TYPE_CHECKING:
    from fear_jackal_sim.environment import FearEnvironment


@dataclass
class PendingPolicyStep:
    """
    Temporary container holding the tensors needed to finish a PPO rollout step after the
    environment responds.
    """
    state_tensor: Any
    action_tensor: Any
    log_prob: Any
    value: Any
    action_index: int
    action_name: str


@dataclass
class RolloutBuffer:
    """PPO rollout storage for Jackal feature states."""

    states: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    logprobs: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    state_values: list[Any] = field(default_factory=list)
    is_terminals: list[bool] = field(default_factory=list)
    action_names: list[str] = field(default_factory=list)

    def clear(self) -> None:
        """Clear the rollout so the next PPO batch starts fresh."""
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()
        self.action_names.clear()

    def __len__(self) -> int:
        """Return how many reward-bearing steps are currently buffered."""
        return len(self.rewards)


class FearAgent:
    """
    Owns the online policy, reward shaping, fear-model hooks, and short observation memory
    for one Jackal learner.
    """
    def __init__(
        self,
        environment: FearEnvironment,
        config: AgentConfig,
        logger,
        writer=None,
    ) -> None:
        """
        Initialize FearAgent and the state it needs during runtime.
        """
        self.environment = environment
        self.config = config
        self.logger = logger
        self.writer = writer
        self.fear_model_mode = str(self.config.fear_model_mode).strip().lower()
        if self.config.reward_mode == 'external_only':
            self.fear_model_mode = 'none'

        # Replay data and PPO rollout data serve different jobs: replay supports
        # archiving/inspection, while rollout storage is what PPO updates actually read.
        self.replay_buffer = ExperienceReplayBuffer(self.config.replay_buffer_capacity)
        self.rollout_buffer = RolloutBuffer()
        self.observation_window: deque[ObservationBundle] = deque(maxlen=self.config.lookback)
        self.episode_transitions: list[Transition] = []
        # The fear-model objects are separate from PPO: they only score short history
        # windows and contribute intrinsic reward or optional reactive overrides.
        self.smann = SMANNAdapter(
            checkpoint_path=self.config.smann_checkpoint,
            repo_path=self.config.fear_repo_path,
            image_size=self.config.smann_image_size,
            lookback=self.config.lookback,
            fear_threshold=self.config.smann_fear_threshold,
        )
        self.memory_fear = SequenceMemoryRewardModel(
            dataset_dir=self.config.manual_memory_dataset_dir,
            bank_path=self.config.manual_memory_bank_path,
            lookback=self.config.lookback,
            image_size=self.config.memory_similarity_image_size,
            depth_clip_m=self.config.memory_similarity_depth_clip_m,
        )
        self.actor = None
        self.critic = None
        self.policy_old_actor = None
        self.policy_old_critic = None
        self.policy_optimizer = None
        self.last_intrinsic_reward = 0.0
        self.last_combined_reward = 0.0
        self.last_fear_score = 0.0
        self.last_raw_fear_score = 0.0
        self.last_fear_active = False
        self.last_fear_override = False
        self.last_policy_action_name = 'none'
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_policy_entropy = 0.0
        self.last_policy_clip_fraction = 0.0
        self.policy_update_count = 0
        self._pending_policy_step: PendingPolicyStep | None = None
        self._reward_goal_offset = 0.0

    def init(self) -> None:
        """
        Initialize the policy networks and load whichever fear backend is currently
        configured.
        """
        self._load_networks()
        self._load_replay_buffer()

        if self.fear_model_mode == 'smann':
            self.smann.load(self.logger)
            self.logger.info('Fear model mode is smann; live intrinsic reward will come from the Sanchez sequence model.')
            return

        if self.fear_model_mode == 'none':
            self.logger.info('Fear model mode is none; this run will not load or score intrinsic fear.')
            return

        if self.fear_model_mode != 'memory_similarity':
            self.logger.warning(
                f'Unknown fear model mode {self.fear_model_mode}; defaulting to memory_similarity.'
            )
            self.fear_model_mode = 'memory_similarity'

        self.memory_fear.load(self.logger)
        self.logger.info(
            'Fear model mode is memory_similarity; live intrinsic reward will come from offline three-step memory matching.'
        )

    def _load_networks(self) -> None:
        """
        Load or initialize the resources needed by this agent subsystem.
        """
        # If we keep PPO enabled, actor and critic are required because PPO needs both
        # a policy distribution and a value baseline. Removing them would disable PPO.
        if self.config.use_policy_network and ActorNetwork is not None and CriticNetwork is not None and optim is not None:
            input_dim = self._policy_feature_dim()
            action_dim = len(self._action_library())
            self.actor = ActorNetwork(input_dim, self.config.policy_hidden_dim, action_dim)
            self.critic = CriticNetwork(input_dim, self.config.policy_hidden_dim)
            self.policy_old_actor = ActorNetwork(input_dim, self.config.policy_hidden_dim, action_dim)
            self.policy_old_critic = CriticNetwork(input_dim, self.config.policy_hidden_dim)
            self.policy_optimizer = optim.Adam(
                [
                    {'params': self.actor.parameters(), 'lr': self.config.policy_learning_rate},
                    {'params': self.critic.parameters(), 'lr': self.config.value_learning_rate},
                ]
            )
            final_linear = self.actor.net[-1]
            with torch.no_grad():
                final_linear.bias.zero_()
                if final_linear.bias.numel() >= 2:
                    final_linear.bias[1] = 1.2
            self.policy_old_actor.load_state_dict(self.actor.state_dict())
            self.policy_old_critic.load_state_dict(self.critic.state_dict())
            self.logger.info(
                'Online PPO policy was initialized using the Sanchez-style rollout update structure '
                'with a feature-based Jackal state encoder.'
            )
        elif self.config.use_policy_network:
            self.logger.warning('PyTorch is unavailable; using heuristic actions instead of a learned policy.')

    def _load_replay_buffer(self) -> None:
        """
        Load or initialize the resources needed by this agent subsystem.
        """
        self.logger.info(
            f'Replay buffer initialized with capacity {self.config.replay_buffer_capacity} transitions.'
        )

    def reward(self, observation: ObservationBundle) -> float:
        """
        Compute the sparse external task reward.

        The simulator task only pays out when the RGB goal-coverage threshold has
        been crossed. Collisions and timeouts are terminal conditions, not external
        reward events.
        """
        if observation.goal_reached:
            return float(self.config.goal_reward_bonus)
        return 0.0

    def cache_observation(self, observation: ObservationBundle) -> None:
        """
        Append the latest observation to the short temporal window used by fear models.
        """
        self.observation_window.append(observation)

    def step(
        self,
        action: AgentAction,
        environment: FearEnvironment,
    ) -> tuple[ObservationBundle, float, bool, bool]:
        """
        Publish the chosen action, gather the next observation, and compute its external
        reward.
        """
        environment.publish_action(action)
        environment.increment_step()
        current_representation = environment.build_observation()
        self.cache_observation(current_representation)
        reward = self.reward(current_representation)
        return (
            current_representation,
            reward,
            current_representation.goal_reached or current_representation.terminal,
            current_representation.truncated,
        )

    def reset(self) -> ObservationBundle:
        """
        Clear episode-local memory so a new rollout starts with a clean short history.
        """
        self.observation_window.clear()
        self.episode_transitions.clear()
        self.last_intrinsic_reward = 0.0
        self.last_combined_reward = 0.0
        self.last_fear_score = 0.0
        self.last_raw_fear_score = 0.0
        self.last_fear_active = False
        self.last_fear_override = False
        self.last_policy_action_name = 'none'
        self._pending_policy_step = None
        self._reward_goal_offset = 0.0
        return self.environment.reset()

    def act(self, state: ObservationBundle) -> AgentAction:
        """
        Choose the next action from the policy or heuristic fallback, then optionally apply
        reactive fear logic.
        """
        self.last_fear_score = 0.0
        self.last_raw_fear_score = 0.0
        self.last_fear_active = False
        self.last_fear_override = False
        self.last_policy_action_name = 'none'

        if state.goal_reached or state.terminal or state.truncated or state.collision:
            self._pending_policy_step = None
            return AgentAction()

        base_action = self._select_base_action(state)
        fear_score = self._score_state_for_action(state)
        self.last_fear_score = fear_score
        self.last_raw_fear_score = self._latest_raw_fear_score(fear_score)
        self.last_fear_active = self._latest_fear_active(fear_score)

        if self.config.fear_reactive_policy and fear_score >= self.config.smann_fear_threshold:
            turn_direction = 1.0 if ((state.step_index // 5) % 2 == 0) else -1.0
            # Keep a little forward motion during fear overrides so the robot can still
            # escape and explore instead of getting trapped in place turning left/right.
            reactive_action = AgentAction(
                linear_x=self.config.fear_reactive_linear_speed,
                angular_z=turn_direction * self.config.fear_reactive_turn_speed,
            )
            self.last_fear_override = True
            # We intentionally do not train PPO on the override step, because the action
            # executed in the world is no longer the one sampled by the policy.
            self._pending_policy_step = None
            if state.step_index % 10 == 0 or state.step_index < self.config.lookback + 2:
                self.logger.info(
                    'Fear-reactive override '
                    f'step={state.step_index} '
                f'fear={fear_score:.3f} '
                f'raw_fear={self.last_raw_fear_score:.3f} '
                f'fear_active={self.last_fear_active} '
                f'policy_action={self.last_policy_action_name} '
                    f'override_linear={reactive_action.linear_x:.3f} '
                    f'override_angular={reactive_action.angular_z:.3f}'
                )
            return reactive_action

        return base_action

    def remember(
        self,
        state: ObservationBundle,
        external_reward: float,
        action: AgentAction,
        terminal: bool,
        truncated: bool,
    ) -> float:
        """
        Store the completed transition, compute intrinsic fear reward, and update PPO
        rollout bookkeeping.
        """
        prepared_window = self._prepare_fear_window(list(self.observation_window))
        intrinsic_reward = self._compute_intrinsic_reward(prepared_window)
        combined_reward = self._combine_rewards(external_reward, intrinsic_reward)
        fear_reward_label = -1 if state.collision else 0

        transition = Transition(
            state_summary=state.summary(),
            action=action,
            external_reward=float(external_reward),
            intrinsic_reward=float(intrinsic_reward),
            combined_reward=float(combined_reward),
            terminal=bool(state.goal_reached or terminal),
            truncated=bool(truncated),
            vision_window=prepared_window,
            danger_label=1 if state.collision else 0,
            reward_label=fear_reward_label,
        )
        self.replay_buffer.append(transition)
        self.episode_transitions.append(transition)
        self.last_intrinsic_reward = float(intrinsic_reward)
        self.last_combined_reward = float(combined_reward)

        done = bool(state.goal_reached or terminal or truncated)
        self._commit_rollout_step(combined_reward, done)

        if state.step_index % 10 == 0 or done:
            self.logger.info(
                'Agent memory updated '
                f'step={state.step_index} '
                f'goal={state.goal_coverage:.3f} '
                f'collision={state.collision} '
                f'goal_reached={state.goal_reached} '
                f'fear={self.last_fear_score:.3f} '
                f'raw_fear={self.last_raw_fear_score:.3f} '
                f'fear_active={self.last_fear_active} '
                f'override={self.last_fear_override} '
                f'action={self.last_policy_action_name} '
                f'ext={external_reward:.3f} '
                f'int={intrinsic_reward:.3f} '
                f'combined={combined_reward:.3f} '
                f'actor_loss={self.last_actor_loss:.4f} '
                f'critic_loss={self.last_critic_loss:.4f}'
            )

        return intrinsic_reward

    def clear_memory(self) -> None:
        """
        Clear the agent's episode-local memory structures.
        """
        self.observation_window.clear()
        self.episode_transitions.clear()
        self.replay_buffer.clear()
        self.rollout_buffer.clear()
        self._pending_policy_step = None

    def _select_base_action(self, state: ObservationBundle) -> AgentAction:
        """
        Choose the internal value represented by this helper method.
        """
        if (
            self.actor is not None
            and self.critic is not None
            and self.policy_old_actor is not None
            and self.policy_old_critic is not None
            and Categorical is not None
        ):
            return self._sample_policy_action(state)

        self._pending_policy_step = None
        analysis = self._analyze_observation(state)
        self.last_policy_action_name = 'heuristic'

        if state.goal_reached:
            return AgentAction()

        if analysis['center_depth'] < 0.18 or analysis['minimum_depth'] < 0.12:
            turn_sign = 1.0 if analysis['left_depth'] > analysis['right_depth'] else -1.0
            return AgentAction(linear_x=0.05, angular_z=turn_sign * self.config.action_angular_speed)

        if state.goal_coverage > 0.01:
            angular = float(np.clip(
                analysis['goal_offset'] * self.config.action_angular_speed,
                -self.config.action_angular_speed,
                self.config.action_angular_speed,
            ))
            linear = self.config.action_linear_speed * (0.45 if abs(analysis['goal_offset']) > 0.35 else 0.75)
            return AgentAction(linear_x=linear, angular_z=angular)

        return AgentAction(linear_x=self.config.action_linear_speed, angular_z=0.0)

    def _sample_policy_action(self, state: ObservationBundle) -> AgentAction:
        """
        Sample the value needed by the surrounding PPO logic.
        """
        state_tensor = torch.tensor(self._policy_features(state), dtype=torch.float32)

        with torch.no_grad():
            logits = self.policy_old_actor(state_tensor)
            # Temperature and epsilon follow the same intent as before, but now the
            # sampled action is stored in a Sanchez-style PPO rollout buffer.
            temperature = max(float(self.config.policy_temperature), 1.0e-3)
            logits = logits / temperature
            dist = Categorical(logits=logits)
            if float(np.random.rand()) < float(self.config.exploration_epsilon):
                action_tensor = torch.tensor(
                    int(np.random.randint(len(self._action_library()))),
                    dtype=torch.int64,
                )
            else:
                action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            value = self.policy_old_critic(state_tensor).squeeze(-1)

        action_index = int(action_tensor.item())
        action_name, action = self._action_library()[action_index]
        self.last_policy_action_name = action_name
        self._pending_policy_step = PendingPolicyStep(
            state_tensor=state_tensor.detach(),
            action_tensor=action_tensor.detach(),
            log_prob=log_prob.detach(),
            value=value.detach(),
            action_index=action_index,
            action_name=action_name,
        )
        return action
    def _commit_rollout_step(self, reward: float, done: bool) -> None:
        """
        Commit the pending step into rollout storage once reward information is known.
        """
        if self._pending_policy_step is None or torch is None:
            return

        step = self._pending_policy_step
        self._pending_policy_step = None
        self.rollout_buffer.states.append(step.state_tensor)
        self.rollout_buffer.actions.append(step.action_tensor)
        self.rollout_buffer.logprobs.append(step.log_prob)
        self.rollout_buffer.state_values.append(step.value)
        self.rollout_buffer.rewards.append(float(reward))
        self.rollout_buffer.is_terminals.append(bool(done))
        self.rollout_buffer.action_names.append(step.action_name)

    def train_policy_from_rollout(self, min_steps: int = 1, force: bool = False) -> bool:
        """
        Run a clipped PPO update over the currently buffered rollout and sync the frozen old
        policy.
        """
        if (
            self.actor is None
            or self.critic is None
            or self.policy_old_actor is None
            or self.policy_old_critic is None
            or self.policy_optimizer is None
            or torch is None
            or Categorical is None
            or F is None
            or clip_grad_norm_ is None
        ):
            self.last_actor_loss = 0.0
            self.last_critic_loss = 0.0
            self.last_policy_entropy = 0.0
            self.last_policy_clip_fraction = 0.0
            return False

        rollout_size = len(self.rollout_buffer)
        if rollout_size == 0:
            self.last_actor_loss = 0.0
            self.last_critic_loss = 0.0
            self.last_policy_entropy = 0.0
            self.last_policy_clip_fraction = 0.0
            return False

        if not force and rollout_size < max(int(min_steps), 1):
            return False

        action_name = self.rollout_buffer.action_names[-1] if self.rollout_buffer.action_names else 'n/a'
        try:
            rewards = []
            discounted_reward = 0.0
            for reward, is_terminal in zip(reversed(self.rollout_buffer.rewards), reversed(self.rollout_buffer.is_terminals)):
                if is_terminal:
                    discounted_reward = 0.0
                discounted_reward = float(reward) + (self.config.discount_factor * discounted_reward)
                rewards.insert(0, discounted_reward)

            returns = torch.tensor(rewards, dtype=torch.float32)
            if returns.numel() > 1:
                returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1.0e-7)

            old_states = torch.stack(self.rollout_buffer.states, dim=0).detach()
            old_actions = torch.stack(self.rollout_buffer.actions, dim=0).long().view(-1).detach()
            old_logprobs = torch.stack(self.rollout_buffer.logprobs, dim=0).view(-1).detach()
            old_state_values = torch.stack(self.rollout_buffer.state_values, dim=0).view(-1).detach()
            advantages = returns.detach() - old_state_values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-7)

            actor_loss_value = 0.0
            critic_loss_value = 0.0
            entropy_value = 0.0
            ratios_mean = 1.0
            clip_fraction_value = 0.0
            for _ in range(max(int(self.config.ppo_update_epochs), 1)):
                logits = self.actor(old_states)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(old_actions)
                entropy = dist.entropy().mean()
                state_values = self.critic(old_states).squeeze(-1)

                ratios = torch.exp(logprobs - old_logprobs)
                clipped_mask = (ratios > 1.0 + float(self.config.ppo_clip_epsilon)) | (
                    ratios < 1.0 - float(self.config.ppo_clip_epsilon)
                )
                surr1 = ratios * advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - float(self.config.ppo_clip_epsilon),
                    1.0 + float(self.config.ppo_clip_epsilon),
                ) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = float(self.config.critic_loss_coefficient) * F.mse_loss(state_values, returns)
                total_loss = actor_loss + critic_loss - (float(self.config.entropy_coefficient) * entropy)

                self.policy_optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.gradient_clip_norm,
                )
                self.policy_optimizer.step()

                actor_loss_value = float(actor_loss.item())
                critic_loss_value = float(critic_loss.item())
                entropy_value = float(entropy.item())
                ratios_mean = float(ratios.mean().item())
                clip_fraction_value = float(clipped_mask.float().mean().item())

            self.policy_old_actor.load_state_dict(self.actor.state_dict())
            self.policy_old_critic.load_state_dict(self.critic.state_dict())
            self.rollout_buffer.clear()

            self.last_actor_loss = actor_loss_value
            self.last_critic_loss = critic_loss_value
            self.last_policy_entropy = entropy_value
            self.last_policy_clip_fraction = clip_fraction_value
            self.policy_update_count += 1

            if self.writer is not None:
                self.writer.add_scalar('ppo/actor_loss', self.last_actor_loss, self.policy_update_count)
                self.writer.add_scalar('ppo/critic_loss', self.last_critic_loss, self.policy_update_count)
                self.writer.add_scalar('ppo/entropy', entropy_value, self.policy_update_count)
                self.writer.add_scalar('ppo/clip_fraction', clip_fraction_value, self.policy_update_count)
                self.writer.add_scalar('ppo/ratio_mean', ratios_mean, self.policy_update_count)
                self.writer.add_scalar('ppo/rollout_size', float(rollout_size), self.policy_update_count)

            if self.policy_update_count % 5 == 0 or force:
                self.logger.info(
                    'Online PPO update '
                    f'updates={self.policy_update_count} '
                    f'batch={rollout_size} '
                    f'action={action_name} '
                    f'actor_loss={self.last_actor_loss:.4f} '
                    f'critic_loss={self.last_critic_loss:.4f} '
                    f'entropy={entropy_value:.4f}'
                )
            return True
        except Exception as exc:
            self.last_actor_loss = 0.0
            self.last_critic_loss = 0.0
            self.last_policy_entropy = 0.0
            self.last_policy_clip_fraction = 0.0
            self.rollout_buffer.clear()
            self.logger.warning(f'Online PPO update failed; clearing the current rollout and keeping the trainer alive: {exc}')
            return False

    def _action_library(self) -> list[tuple[str, AgentAction]]:
        """
        Return the action metadata used by the policy.
        """
        linear = float(self.config.action_linear_speed)
        angular = float(self.config.action_angular_speed)
        return [
            ('stop', AgentAction()),
            ('forward', AgentAction(linear_x=linear, angular_z=0.0)),
            ('arc_left', AgentAction(linear_x=linear * 0.70, angular_z=angular * 0.55)),
            ('arc_right', AgentAction(linear_x=linear * 0.70, angular_z=-angular * 0.55)),
            ('turn_left', AgentAction(linear_x=linear * 0.25, angular_z=angular)),
            ('turn_right', AgentAction(linear_x=linear * 0.25, angular_z=-angular)),
        ]

    def _policy_feature_dim(self) -> int:
        """
        Policy helper used while building PPO inputs or outputs.
        """
        return 9

    def _policy_features(self, state: ObservationBundle) -> np.ndarray:
        """
        Compress the full observation into the low-dimensional feature vector consumed by
        the PPO networks.
        """
        analysis = self._analyze_observation(state)
        step_fraction = 0.0
        if self.environment.config.max_episode_steps > 0:
            step_fraction = min(float(state.step_index) / float(self.environment.config.max_episode_steps), 1.0)
        goal_visible = 1.0 if state.goal_coverage > 0.005 else 0.0
        return np.asarray(
            [
                float(state.goal_coverage),
                goal_visible,
                float(analysis['goal_offset']),
                float(analysis['left_depth']),
                float(analysis['center_depth']),
                float(analysis['right_depth']),
                float(analysis['minimum_depth']),
                float(state.collision),
                float(step_fraction),
            ],
            dtype=np.float32,
        )

    def _analyze_observation(self, observation: ObservationBundle) -> dict[str, float]:
        """
        Extract goal-centering and depth-sector helper measurements from the current
        observation.
        """
        analysis = {
            'goal_offset': 0.0,
            'left_depth': 1.0,
            'center_depth': 1.0,
            'right_depth': 1.0,
            'minimum_depth': 1.0,
        }

        if observation.color_msg is not None:
            try:
                rgb_image = rgb_image_to_numpy(observation.color_msg)
                analysis['goal_offset'] = float(compute_green_goal_offset(rgb_image))
            except ImageDecodingError:
                pass

        if observation.depth_msg is not None:
            try:
                depth_image = depth_image_to_numpy(observation.depth_msg)
                depth_summary = summarize_depth_sectors(depth_image)
                analysis['left_depth'] = float(depth_summary['left'])
                analysis['center_depth'] = float(depth_summary['center'])
                analysis['right_depth'] = float(depth_summary['right'])
                analysis['minimum_depth'] = float(depth_summary['minimum'])
            except ImageDecodingError:
                pass

        return analysis
    def _prepare_fear_window(self, observation_window: list[ObservationBundle]):
        """
        Prepare the intermediate representation needed by the fear model.
        """
        if self.fear_model_mode == 'none':
            return None
        if self.fear_model_mode == 'smann':
            return self.smann.prepare_window(observation_window, self.logger)
        return self.memory_fear.prepare_window(observation_window, self.logger)

    def _score_state_for_action(self, state: ObservationBundle) -> float:
        """
        Score the current state according to the configured fear backend.
        """
        if self.fear_model_mode == 'none':
            self.last_raw_fear_score = 0.0
            self.last_fear_active = False
            return 0.0
        candidate_window = list(self.observation_window)
        candidate_window.append(state)
        prepared_window = self._prepare_fear_window(candidate_window)
        if self.fear_model_mode == 'smann':
            return float(self.smann.score_prepared(prepared_window, self.logger))
        return float(self.memory_fear.score_prepared(prepared_window, self.logger))

    def _latest_raw_fear_score(self, thresholded_score: float) -> float:
        if self.fear_model_mode == 'smann':
            return float(self.smann.last_raw_score)
        return float(thresholded_score)

    def _latest_fear_active(self, thresholded_score: float) -> bool:
        if self.fear_model_mode == 'smann':
            return bool(self.smann.last_fear_active)
        return bool(abs(thresholded_score) > 0.0)

    def _compute_intrinsic_reward(self, prepared_window) -> float:
        """
        Turn the current short observation history into an intrinsic fear penalty.
        """
        if self.fear_model_mode == 'none':
            self.last_raw_fear_score = 0.0
            self.last_fear_score = 0.0
            self.last_fear_active = False
            return 0.0
        if self.fear_model_mode == 'smann':
            intrinsic_reward = float(self.smann.compute_thresholded_intrinsic_reward(prepared_window, self.logger))
            thresholded_score = abs(intrinsic_reward)
            self.last_raw_fear_score = float(self.smann.last_raw_score)
            self.last_fear_score = thresholded_score
            self.last_fear_active = bool(self.smann.last_fear_active)
            return intrinsic_reward
        reward = float(self.memory_fear.predict_reward_prepared(prepared_window, self.logger))
        self.last_raw_fear_score = abs(reward)
        self.last_fear_score = abs(reward)
        self.last_fear_active = bool(abs(reward) > 0.0)
        return reward

    def _combine_rewards(self, external_reward: float, intrinsic_reward: float) -> float:
        """
        Blend external and intrinsic reward according to the configured training mode and
        scales.
        """
        scaled_external = self.config.external_reward_scale * float(external_reward)
        scaled_intrinsic = self.config.intrinsic_reward_scale * float(intrinsic_reward)

        if self.config.reward_mode == 'intrinsic_only':
            return scaled_intrinsic
        if self.config.reward_mode == 'external_only':
            return scaled_external
        return scaled_external + scaled_intrinsic
