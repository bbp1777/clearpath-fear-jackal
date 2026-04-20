"""
Top-level ROS training node that owns the episode loop, simulator reset/relaunch flow, PPO
update cadence, logging, and episode archiving.
"""
from __future__ import annotations

import math
import json
import os
import shlex
import signal
import subprocess
import time
from typing import Optional

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node

from fear_jackal_sim.agent import FearAgent
from fear_jackal_sim.dataset_tools import archive_episode_transitions
from fear_jackal_sim.environment import FearEnvironment
from fear_jackal_sim.rl_types import AgentAction, AgentConfig, EnvironmentConfig, TrainerConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class FearTrainer(Node):
    """
    Central coordinator for training. It bridges ROS parameters, the environment wrapper,
    the policy agent, TensorBoard logging, and simulator lifecycle management.
    """
    def __init__(self) -> None:
        """
        Build the environment, agent, logging tools, and wall-clock timer that drive
        training forward.
        """
        super().__init__('fear_trainer')
        self._declare_parameters()
        self.environment_config = self._read_environment_config()
        self.agent_config = self._read_agent_config()
        self.trainer_config = self._read_trainer_config()

        self.writer = self._create_writer()
        self.environment = FearEnvironment(self, self.environment_config)
        self.environment.register_reset_callback(self._reset_simulation)
        self.agent = FearAgent(self.environment, self.agent_config, self.get_logger(), self.writer)
        self.agent.init()
        # This log call prints the frozen-evaluation settings right when the trainer boots.
        self._log_runtime_mode()

        self.global_step = 0
        self.episode_index = 0
        self.episode_active = False
        self.episode_external_reward = 0.0
        self.episode_intrinsic_reward = 0.0
        self.episode_combined_reward = 0.0
        self.episode_fear_sum = 0.0
        self.episode_fear_max = 0.0
        self.episode_fear_active_count = 0
        self.episode_fear_count = 0
        self._waiting_logged = False
        self._reset_pending = False
        self._reset_wait_logged = False
        self._reset_requested_monotonic = 0.0
        self._reset_retry_count = 0
        self._max_reset_retries = 3
        self._sim_process: Optional[subprocess.Popen[str]] = None

        # Use steady wall-clock time when the trainer owns Gazebo so the reset loop can
        # bootstrap the simulator before /clock exists again after a full relaunch.
        timer_clock = Clock(clock_type=ClockType.STEADY_TIME) if self.environment_config.manage_sim_process else None
        # This timer drives the slower policy or heuristic decision loop.
        self.timer = self.create_timer(self.trainer_config.control_period_s, self._training_tick, clock=timer_clock)
        # This timer refreshes the last action stamp so ros2_control does not time out between decisions.
        self.action_keepalive_timer = self.create_timer(0.05, self._action_keepalive_tick, clock=timer_clock)
        # This startup log confirms the trainer finished building all runtime pieces.
        self.get_logger().info('Fear trainer scaffold is running.')

    def _declare_parameters(self) -> None:
        """
        Declare every ROS parameter consumed by the trainer, environment, policy, and reset
        logic.
        """
        self.declare_parameter('namespace', 'jackal_sidewalk')
        self.declare_parameter('color_topic', '/jackal_sidewalk/sensors/camera_0/color/image')
        self.declare_parameter('depth_topic', '/jackal_sidewalk/sensors/camera_0/depth/image')
        self.declare_parameter('enable_audio', False)
        self.declare_parameter('audio_topic', '')
        self.declare_parameter('goal_coverage_topic', '/jackal_sidewalk/goal/coverage')
        self.declare_parameter('collision_topic', '/jackal_sidewalk/collision')
        self.declare_parameter('cmd_vel_topic', '/jackal_sidewalk/cmd_vel')
        self.declare_parameter('world_name', 'mini_sidewalk')
        self.declare_parameter('model_name', 'jackal_sidewalk/robot')
        self.declare_parameter('setup_path', '/workspaces/clearpath_docker/sim_setup')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        self.declare_parameter('rviz', False)
        self.declare_parameter('manage_sim_process', True)
        self.declare_parameter('ros_setup_script', '/opt/ros/jazzy/setup.bash')
        self.declare_parameter('workspace_setup_script', '/workspaces/clearpath_docker/clearpath_ws/install/setup.bash')
        self.declare_parameter('spawn_x', -14.0)
        self.declare_parameter('spawn_y', 0.0)
        self.declare_parameter('spawn_z', 0.20)
        self.declare_parameter('spawn_yaw', 0.0)
        self.declare_parameter('max_episode_steps', 400)
        self.declare_parameter('goal_completion_threshold', 0.30)
        self.declare_parameter('reset_timeout_s', 5.0)
        self.declare_parameter('reset_settle_s', 1.0)
        self.declare_parameter('sim_relaunch_timeout_s', 45.0)
        self.declare_parameter('sim_shutdown_timeout_s', 10.0)
        self.declare_parameter('sim_time_rewind_threshold_s', 2.0)
        self.declare_parameter('post_reset_discard_collision_messages', 2)
        self.declare_parameter('lookback', 3)
        self.declare_parameter('reward_mode', 'combined')
        self.declare_parameter('external_reward_scale', 1.0)
        self.declare_parameter('intrinsic_reward_scale', 1.0)
        self.declare_parameter('replay_buffer_capacity', 5000)
        self.declare_parameter('policy_hidden_dim', 64)
        self.declare_parameter('action_linear_speed', 0.40)
        self.declare_parameter('action_angular_speed', 0.75)
        # This keeps the default live evaluator on the heuristic action source until a frozen PPO checkpoint exists.
        self.declare_parameter('use_policy_network', False)
        self.declare_parameter('policy_learning_rate', 3.0e-4)
        self.declare_parameter('value_learning_rate', 1.0e-3)
        self.declare_parameter('policy_temperature', 1.35)
        self.declare_parameter('exploration_epsilon', 0.10)
        self.declare_parameter('discount_factor', 0.99)
        self.declare_parameter('ppo_update_epochs', 4)
        self.declare_parameter('ppo_clip_epsilon', 0.20)
        self.declare_parameter('entropy_coefficient', 0.03)
        self.declare_parameter('critic_loss_coefficient', 0.5)
        self.declare_parameter('gradient_clip_norm', 1.0)
        self.declare_parameter('goal_progress_scale', 0.0)
        self.declare_parameter('goal_alignment_scale', 0.0)
        self.declare_parameter('step_penalty', 0.0)
        self.declare_parameter('collision_penalty', 0.0)
        self.declare_parameter('goal_reward_bonus', 1.0)
        # This makes SMANN the default fear backend for the live evaluator.
        self.declare_parameter('fear_model_mode', 'smann')
        self.declare_parameter('manual_memory_dataset_dir', '')
        self.declare_parameter('manual_memory_bank_path', '')
        self.declare_parameter('memory_similarity_image_size', 84)
        self.declare_parameter('memory_similarity_depth_clip_m', 5.0)
        # This points the live evaluator at the offline Jackal SMANN checkpoint.
        self.declare_parameter('smann_checkpoint', '/workspaces/clearpath_docker/clearpath_ws/logs/rodney_training/jackal_mann_independent/weights')
        self.declare_parameter('smann_dataset_dir', '/workspaces/clearpath_docker/clearpath_ws/logs/rodney_dataset')
        # This points the adapter at the CarRacingTesting source tree used during training.
        self.declare_parameter('fear_repo_path', '/workspaces/Behavior-Intrinsic-Fear-main/CarRacingTesting')
        self.declare_parameter('sanchez_upstream_repo', 'https://github.com/ras8047/Behavior-Intrinsic-Fear')
        self.declare_parameter('sanchez_upstream_commit', '')
        self.declare_parameter('smann_image_size', 84)
        self.declare_parameter('smann_fear_threshold', 0.50)
        self.declare_parameter('fear_reactive_policy', False)
        self.declare_parameter('fear_reactive_linear_speed', 0.12)
        self.declare_parameter('fear_reactive_turn_speed', 0.90)
        # This shorter default keeps stamped velocity commands fresh during frozen evaluation.
        self.declare_parameter('control_period_s', 0.1)
        self.declare_parameter('train_every_n_steps', 64)
        self.declare_parameter('vicarious_update_every_n_episodes', 1)
        self.declare_parameter('tensorboard_log_dir', '/workspaces/clearpath_docker/clearpath_ws/logs/tensorboard')
        self.declare_parameter('clear_buffer_on_reset', False)
        self.declare_parameter('enable_online_smann_updates', False)
        # This flag disables PPO learning so live runs stay frozen for evaluation.
        self.declare_parameter('evaluation_only', False)
        self.declare_parameter('run_name', 'fear_trainer')
        self.declare_parameter('archive_episodes', True)
        self.declare_parameter('episode_archive_dir', '/workspaces/clearpath_docker/clearpath_ws/logs/episode_archives')

    def _read_environment_config(self) -> EnvironmentConfig:
        """
        Read environment-related parameters into an EnvironmentConfig dataclass.
        """
        return EnvironmentConfig(
            namespace=self.get_parameter('namespace').value,
            color_topic=self.get_parameter('color_topic').value,
            depth_topic=self.get_parameter('depth_topic').value,
            enable_audio=bool(self.get_parameter('enable_audio').value),
            audio_topic=self.get_parameter('audio_topic').value,
            goal_coverage_topic=self.get_parameter('goal_coverage_topic').value,
            collision_topic=self.get_parameter('collision_topic').value,
            cmd_vel_topic=self.get_parameter('cmd_vel_topic').value,
            world_name=self.get_parameter('world_name').value,
            model_name=self.get_parameter('model_name').value,
            setup_path=str(self.get_parameter('setup_path').value),
            use_sim_time=bool(self.get_parameter('use_sim_time').value),
            rviz=bool(self.get_parameter('rviz').value),
            manage_sim_process=bool(self.get_parameter('manage_sim_process').value),
            ros_setup_script=str(self.get_parameter('ros_setup_script').value),
            workspace_setup_script=str(self.get_parameter('workspace_setup_script').value),
            spawn_x=float(self.get_parameter('spawn_x').value),
            spawn_y=float(self.get_parameter('spawn_y').value),
            spawn_z=float(self.get_parameter('spawn_z').value),
            spawn_yaw=float(self.get_parameter('spawn_yaw').value),
            max_episode_steps=int(self.get_parameter('max_episode_steps').value),
            goal_completion_threshold=float(self.get_parameter('goal_completion_threshold').value),
            reset_timeout_s=float(self.get_parameter('reset_timeout_s').value),
            reset_settle_s=float(self.get_parameter('reset_settle_s').value),
            sim_relaunch_timeout_s=float(self.get_parameter('sim_relaunch_timeout_s').value),
            sim_shutdown_timeout_s=float(self.get_parameter('sim_shutdown_timeout_s').value),
            sim_time_rewind_threshold_s=float(self.get_parameter('sim_time_rewind_threshold_s').value),
            post_reset_discard_collision_messages=int(self.get_parameter('post_reset_discard_collision_messages').value),
        )

    def _read_agent_config(self) -> AgentConfig:
        """
        Read policy, reward, and fear-model parameters into an AgentConfig dataclass.
        """
        return AgentConfig(
            lookback=int(self.get_parameter('lookback').value),
            reward_mode=str(self.get_parameter('reward_mode').value),
            external_reward_scale=float(self.get_parameter('external_reward_scale').value),
            intrinsic_reward_scale=float(self.get_parameter('intrinsic_reward_scale').value),
            replay_buffer_capacity=int(self.get_parameter('replay_buffer_capacity').value),
            policy_hidden_dim=int(self.get_parameter('policy_hidden_dim').value),
            action_linear_speed=float(self.get_parameter('action_linear_speed').value),
            action_angular_speed=float(self.get_parameter('action_angular_speed').value),
            use_policy_network=bool(self.get_parameter('use_policy_network').value),
            policy_learning_rate=float(self.get_parameter('policy_learning_rate').value),
            value_learning_rate=float(self.get_parameter('value_learning_rate').value),
            policy_temperature=float(self.get_parameter('policy_temperature').value),
            exploration_epsilon=float(self.get_parameter('exploration_epsilon').value),
            discount_factor=float(self.get_parameter('discount_factor').value),
            ppo_update_epochs=int(self.get_parameter('ppo_update_epochs').value),
            ppo_clip_epsilon=float(self.get_parameter('ppo_clip_epsilon').value),
            entropy_coefficient=float(self.get_parameter('entropy_coefficient').value),
            critic_loss_coefficient=float(self.get_parameter('critic_loss_coefficient').value),
            gradient_clip_norm=float(self.get_parameter('gradient_clip_norm').value),
            goal_progress_scale=float(self.get_parameter('goal_progress_scale').value),
            goal_alignment_scale=float(self.get_parameter('goal_alignment_scale').value),
            step_penalty=float(self.get_parameter('step_penalty').value),
            collision_penalty=float(self.get_parameter('collision_penalty').value),
            goal_reward_bonus=float(self.get_parameter('goal_reward_bonus').value),
            fear_model_mode=str(self.get_parameter('fear_model_mode').value),
            manual_memory_dataset_dir=str(self.get_parameter('manual_memory_dataset_dir').value),
            manual_memory_bank_path=str(self.get_parameter('manual_memory_bank_path').value),
            memory_similarity_image_size=int(self.get_parameter('memory_similarity_image_size').value),
            memory_similarity_depth_clip_m=float(self.get_parameter('memory_similarity_depth_clip_m').value),
            smann_checkpoint=str(self.get_parameter('smann_checkpoint').value),
            smann_dataset_dir=str(self.get_parameter('smann_dataset_dir').value),
            fear_repo_path=str(self.get_parameter('fear_repo_path').value),
            sanchez_upstream_repo=str(self.get_parameter('sanchez_upstream_repo').value),
            sanchez_upstream_commit=str(self.get_parameter('sanchez_upstream_commit').value),
            smann_image_size=int(self.get_parameter('smann_image_size').value),
            smann_fear_threshold=float(self.get_parameter('smann_fear_threshold').value),
            fear_reactive_policy=bool(self.get_parameter('fear_reactive_policy').value),
            fear_reactive_linear_speed=float(self.get_parameter('fear_reactive_linear_speed').value),
            fear_reactive_turn_speed=float(self.get_parameter('fear_reactive_turn_speed').value),
        )

    def _read_trainer_config(self) -> TrainerConfig:
        """
        Read cadence, logging, and archive parameters into a TrainerConfig dataclass.
        """
        return TrainerConfig(
            control_period_s=float(self.get_parameter('control_period_s').value),
            train_every_n_steps=int(self.get_parameter('train_every_n_steps').value),
            vicarious_update_every_n_episodes=int(self.get_parameter('vicarious_update_every_n_episodes').value),
            tensorboard_log_dir=str(self.get_parameter('tensorboard_log_dir').value),
            clear_buffer_on_reset=bool(self.get_parameter('clear_buffer_on_reset').value),
            enable_online_smann_updates=bool(self.get_parameter('enable_online_smann_updates').value),
            # This reads the frozen-evaluation toggle from the ROS parameter set.
            evaluation_only=bool(self.get_parameter('evaluation_only').value),
            run_name=str(self.get_parameter('run_name').value),
            archive_episodes=bool(self.get_parameter('archive_episodes').value),
            episode_archive_dir=str(self.get_parameter('episode_archive_dir').value),
        )

    def _create_writer(self) -> Optional[SummaryWriter]:
        """
        Create the optional TensorBoard writer if the dependency is installed.
        """
        if SummaryWriter is None:
            self.get_logger().warning('TensorBoard writer is unavailable; scalar logging is disabled.')
            return None

        os.makedirs(self.trainer_config.tensorboard_log_dir, exist_ok=True)
        run_dir = os.path.join(self.trainer_config.tensorboard_log_dir, self.trainer_config.run_name)
        writer = SummaryWriter(log_dir=run_dir)
        self._write_run_metadata(writer)
        return writer

    def _write_run_metadata(self, writer: SummaryWriter) -> None:
        """
        Write the run configuration into TensorBoard so paper plots retain provenance.
        """
        effective_fear_model_mode = (
            'none' if self.agent_config.reward_mode == 'external_only' else self.agent_config.fear_model_mode
        )
        metadata = {
            'reward_mode': self.agent_config.reward_mode,
            'external_reward_scale': self.agent_config.external_reward_scale,
            'intrinsic_reward_scale': self.agent_config.intrinsic_reward_scale,
            'smann_fear_threshold': self.agent_config.smann_fear_threshold,
            'smann_dataset_dir': self.agent_config.smann_dataset_dir,
            'smann_checkpoint': self.agent_config.smann_checkpoint,
            'fear_model_mode': self.agent_config.fear_model_mode,
            'effective_fear_model_mode': effective_fear_model_mode,
            'fear_repo_path': self.agent_config.fear_repo_path,
            'sanchez_upstream_repo': self.agent_config.sanchez_upstream_repo,
            'sanchez_upstream_commit': self.agent_config.sanchez_upstream_commit,
            'max_episode_steps': self.environment_config.max_episode_steps,
            'goal_completion_threshold': self.environment_config.goal_completion_threshold,
            'evaluation_only': self.trainer_config.evaluation_only,
            'enable_online_smann_updates': self.trainer_config.enable_online_smann_updates,
            'run_name': self.trainer_config.run_name,
        }
        writer.add_text('run/metadata', f'```json\n{json.dumps(metadata, indent=2)}\n```', 0)

    def _log_runtime_mode(self) -> None:
        """
        Log the active evaluation settings so reward-mode and threshold sweeps are obvious
        in the ROS output before an episode begins.
        """
        # This branch clearly reports when the trainer is running as a frozen evaluator.
        if self.trainer_config.evaluation_only:
            effective_fear_model_mode = (
                'none' if self.agent_config.reward_mode == 'external_only' else self.agent_config.fear_model_mode
            )
            self.get_logger().info(
                'Frozen evaluation mode is enabled; '
                f'reward_mode={self.agent_config.reward_mode} '
                f'fear_model_mode={effective_fear_model_mode} '
                f'fear_threshold={self.agent_config.smann_fear_threshold:.3f} '
                f'checkpoint={self.agent_config.smann_checkpoint or "none"} '
                f'use_policy_network={self.agent_config.use_policy_network}'
            )
            return

        # This branch makes it obvious when the trainer is still allowed to learn online.
        smann_update_text = 'enabled' if self.trainer_config.enable_online_smann_updates else 'disabled'
        self.get_logger().info(
            'Online PPO learning mode is enabled; '
            f'SMANN online updates are {smann_update_text}.'
        )

    def _action_keepalive_tick(self) -> None:
        """
        Re-send the last action often enough to keep the diff-drive controller alive
        while the frozen evaluator waits for the next policy tick.
        """
        # This guard skips keepalive messages while a reset is still in progress.
        if self._reset_pending:
            return

        # This guard skips keepalive messages before an episode has actually started.
        if not self.episode_active:
            return

        # This guard waits until fresh post-reset frames are available again.
        if not self.environment.is_ready():
            return

        # This guard only runs the keepalive during frozen evaluation sweeps.
        if not self.trainer_config.evaluation_only:
            return

        # This republishes the cached action with a fresh timestamp.
        self.environment.republish_last_action()

    def _training_tick(self) -> None:
        """
        Run one control-loop tick: handle resets, step the policy once, record reward, and
        trigger PPO updates when enough rollout data has been gathered.
        """
        # The trainer behaves like a small state machine: wait for reset readiness,
        # start an episode, take one step, then repeat until a terminal event occurs.
        if self._reset_pending:
            self._advance_episode_reset()
            return

        if not self.episode_active:
            self._request_episode_reset(retrying=False)
            return

        if not self.environment.is_ready():
            if not self._waiting_logged:
                self.get_logger().info('Waiting for color and depth topics before starting trainer steps.')
                self._waiting_logged = True
            return

        self._waiting_logged = False

        state = self.environment.build_observation()
        if state.goal_reached or state.terminal or state.truncated:
            self._handle_terminal_observation(state)
            return

        action = self.agent.act(state)
        next_state, external_reward, terminal, truncated = self.agent.step(action, self.environment)
        intrinsic_reward = self.agent.remember(next_state, external_reward, action, terminal, truncated)

        self._record_reward_step(next_state, external_reward, intrinsic_reward, advance_global_step=True)

        # PPO is updated only after a full rollout chunk has been collected so the
        # learning pattern stays close to the rollout-then-optimize structure.
        if (not self.trainer_config.evaluation_only) and self.global_step % self.trainer_config.train_every_n_steps == 0:
            if self.agent.actor is not None:
                updated = self.agent.train_policy_from_rollout(
                    min_steps=self.trainer_config.train_every_n_steps,
                    force=False,
                )
                if updated:
                    self.get_logger().info(
                        'Online PPO status '
                        f'global_step={self.global_step} '
                        f'policy_updates={self.agent.policy_update_count} '
                        f'actor_loss={self.agent.last_actor_loss:.4f} '
                        f'critic_loss={self.agent.last_critic_loss:.4f}'
                    )
            else:
                self.get_logger().info(
                    'Heuristic control is active; enable the policy network to let the Jackal learn online.'
                )
        # This branch reminds us that evaluation sweeps are intentionally frozen.
        elif self.trainer_config.evaluation_only and self.global_step % self.trainer_config.train_every_n_steps == 0:
            self.get_logger().info('Frozen evaluation mode is active; skipping online PPO updates for this run.')

        if terminal or truncated:
            self._publish_stop_burst(repeats=2, interval_s=0.05)
            self._finish_episode(next_state)

    def _request_episode_reset(self, retrying: bool) -> None:
        """
        Enter the reset state and start tracking timeouts/retries for the next episode
        start.
        """
        if retrying:
            self._reset_retry_count += 1
        else:
            self._reset_retry_count = 0
            self.episode_external_reward = 0.0
            self.episode_intrinsic_reward = 0.0
            self.episode_combined_reward = 0.0
            self.episode_fear_sum = 0.0
            self.episode_fear_max = 0.0
            self.episode_fear_active_count = 0
            self.episode_fear_count = 0

        self.agent.reset()
        self._reset_pending = True
        self._reset_requested_monotonic = time.monotonic()
        self._reset_wait_logged = False
        self._waiting_logged = False

        target_episode = self.episode_index + 1
        if retrying:
            self.get_logger().warning(
                f'Retrying simulator reset for episode {target_episode} (attempt {self._reset_retry_count + 1}).'
            )
        else:
            self.get_logger().info(
                f'Requested simulator reset for episode {target_episode}; waiting for fresh post-reset observations and collision-clear confirmation.'
            )

    def _advance_episode_reset(self) -> None:
        """
        Advance the reset state machine until the simulator and sensors are ready again.
        """
        elapsed = time.monotonic() - self._reset_requested_monotonic
        reset_timeout_s = (
            self.environment_config.sim_relaunch_timeout_s
            if self.environment_config.manage_sim_process
            else self.environment_config.reset_timeout_s
        )

        if not self.environment.is_ready():
            if elapsed >= reset_timeout_s and self._reset_retry_count + 1 < self._max_reset_retries:
                self.get_logger().warning(
                    'Fresh post-reset sensor frames and collision-clear confirmation did not arrive before timeout; issuing another reset request.'
                )
                self._request_episode_reset(retrying=True)
                return

            if not self._reset_wait_logged:
                self.get_logger().info('Waiting for fresh post-reset color/depth frames and a cleared collision signal before starting the episode.')
                self._reset_wait_logged = True
            return

        state = self.environment.build_observation()
        if state.collision or state.terminal:
            if elapsed >= reset_timeout_s and self._reset_retry_count + 1 < self._max_reset_retries:
                self.get_logger().warning(
                    'Post-reset observation is still terminal/colliding; retrying simulator reset.'
                )
                self._request_episode_reset(retrying=True)
                return

            if not self._reset_wait_logged:
                self.get_logger().warning(
                    'Post-reset observation is still colliding; waiting for Gazebo to return the robot to a safe start pose.'
                )
                self._reset_wait_logged = True
            return

        self.episode_index += 1
        self.episode_active = True
        self._reset_pending = False
        self._reset_wait_logged = False
        self.get_logger().info(f'Started episode {self.episode_index}.')

    def _record_reward_step(
        self,
        state,
        external_reward: float,
        intrinsic_reward: float,
        advance_global_step: bool,
    ) -> None:
        """
        Accumulate per-episode reward totals and update trainer-side scalar logging state.
        """
        if advance_global_step:
            self.global_step += 1

        self.episode_external_reward += float(external_reward)
        self.episode_intrinsic_reward += float(intrinsic_reward)
        self.episode_combined_reward += float(self.agent.last_combined_reward)
        fear_score = float(self.agent.last_raw_fear_score)
        self.episode_fear_sum += fear_score
        self.episode_fear_max = max(self.episode_fear_max, fear_score)
        self.episode_fear_active_count += 1 if bool(self.agent.last_fear_active) else 0
        self.episode_fear_count += 1

        if self.writer is not None:
            self.writer.add_scalar('step/external_reward', float(external_reward), self.global_step)
            self.writer.add_scalar('step/intrinsic_reward', float(intrinsic_reward), self.global_step)
            self.writer.add_scalar('step/combined_reward', float(self.agent.last_combined_reward), self.global_step)
            self.writer.add_scalar('step/goal_coverage', float(state.goal_coverage), self.global_step)
            self.writer.add_scalar('step/collision_flag', float(state.collision), self.global_step)
            self.writer.add_scalar('step/goal_reached', float(state.goal_reached), self.global_step)
            if self.agent.fear_model_mode != 'none':
                self.writer.add_scalar('fear/raw_unsafe_probability', float(self.agent.last_raw_fear_score), self.global_step)
                self.writer.add_scalar('fear/thresholded_score', float(self.agent.last_fear_score), self.global_step)
                self.writer.add_scalar('fear/active', float(self.agent.last_fear_active), self.global_step)
                self.writer.add_scalar('fear/threshold', float(self.agent_config.smann_fear_threshold), self.global_step)
                self.writer.add_scalar('fear/override', float(self.agent.last_fear_override), self.global_step)

    def _handle_terminal_observation(self, state) -> None:
        """
        Finish an episode immediately when the environment reports terminal or truncated
        state before a new action is issued.
        """
        self._publish_stop_burst(repeats=2, interval_s=0.05)
        self.agent.cache_observation(state)

        external_reward = self.agent.reward(state)
        intrinsic_reward = self.agent.remember(
            state,
            external_reward,
            AgentAction(),
            state.goal_reached or state.terminal,
            state.truncated,
        )
        self._record_reward_step(state, external_reward, intrinsic_reward, advance_global_step=False)

        if state.collision:
            reason = 'collision'
        elif state.goal_reached:
            reason = 'goal'
        else:
            reason = 'truncation'

        self.get_logger().info(
            'Terminal observation latched before issuing a new action; '
            f'ending episode immediately (reason={reason}, step={state.step_index}).'
        )
        self._finish_episode(state)

    def _finish_episode(self, final_state) -> None:
        """
        Close out an episode, archive labeled data, optionally update fear memory, and queue
        the next reset.
        """
        mean_fear_score = self.episode_fear_sum / max(self.episode_fear_count, 1)
        fear_active_fraction = self.episode_fear_active_count / max(self.episode_fear_count, 1)
        self.get_logger().info(
            'Episode complete '
            f'episode={self.episode_index} '
            f'external={self.episode_external_reward:.3f} '
            f'intrinsic={self.episode_intrinsic_reward:.3f} '
            f'combined={self.episode_combined_reward:.3f} '
            f'mean_fear={mean_fear_score:.3f} '
            f'max_fear={self.episode_fear_max:.3f} '
            f'fear_active_fraction={fear_active_fraction:.3f} '
            f'goal_reached={final_state.goal_reached} '
            f'terminal={final_state.terminal} '
            f'truncated={final_state.truncated}'
        )

        if self.writer is not None:
            self.writer.add_scalar('episode/external_return', self.episode_external_reward, self.episode_index)
            self.writer.add_scalar('episode/intrinsic_return', self.episode_intrinsic_reward, self.episode_index)
            self.writer.add_scalar('episode/combined_return', self.episode_combined_reward, self.episode_index)
            self.writer.add_scalar('episode/length', final_state.step_index, self.episode_index)
            self.writer.add_scalar('episode/goal_reached', float(final_state.goal_reached), self.episode_index)
            self.writer.add_scalar('episode/terminal_collision', float(final_state.collision), self.episode_index)
            self.writer.add_scalar('episode/goal_coverage_final', float(final_state.goal_coverage), self.episode_index)
            if self.agent.fear_model_mode != 'none':
                self.writer.add_scalar('fear/raw_unsafe_probability_mean', mean_fear_score, self.episode_index)
                self.writer.add_scalar('fear/raw_unsafe_probability_max', self.episode_fear_max, self.episode_index)
                self.writer.add_scalar('fear/active_fraction', fear_active_fraction, self.episode_index)
                self.writer.add_scalar('fear/threshold', float(self.agent_config.smann_fear_threshold), self.episode_index)
            self.writer.add_scalar('ppo/actor_loss_latest', float(self.agent.last_actor_loss), self.episode_index)
            self.writer.add_scalar('ppo/critic_loss_latest', float(self.agent.last_critic_loss), self.episode_index)
            self.writer.add_scalar('ppo/entropy_latest', float(self.agent.last_policy_entropy), self.episode_index)
            self.writer.add_scalar('ppo/clip_fraction_latest', float(self.agent.last_policy_clip_fraction), self.episode_index)

        # This guard keeps the PPO policy fixed during frozen evaluation runs.
        if (not self.trainer_config.evaluation_only) and self.agent.actor is not None:
            self.agent.train_policy_from_rollout(min_steps=1, force=True)

        if self.trainer_config.archive_episodes:
            archive_summary = archive_episode_transitions(
                self.trainer_config.episode_archive_dir,
                self.episode_index,
                self.agent.episode_transitions,
            )
            if bool(archive_summary['saved']):
                self.get_logger().info(
                    'Archived episode windows '
                    f"episode={self.episode_index} "
                    f"windows={archive_summary['windows']} "
                    f"danger={archive_summary['danger_windows']} "
                    f"safe={archive_summary['safe_windows']} "
                    f"path={archive_summary['path']}"
                )

        # Keep the base PPO condition clean: external_only should not update SMANN.
        if (
            not self.trainer_config.evaluation_only
            and self.trainer_config.enable_online_smann_updates
            and self.agent_config.reward_mode != 'external_only'
            and self.episode_index % self.trainer_config.vicarious_update_every_n_episodes == 0
        ):
            self.train_external_memory_with_vicarious_conditioning()

        if self.trainer_config.clear_buffer_on_reset:
            self.clear_memory_buffers()

        self.episode_active = False

    def clear_memory_buffers(self) -> None:
        """
        Clear replay and short-term memory if the experiment configuration requests it.
        """
        self.agent.clear_memory()
        self.get_logger().info('Cleared agent replay, PPO rollout, and episode buffers.')

    def train_external_memory_with_vicarious_conditioning(self) -> None:
        # The offline memory-similarity path is trained ahead of time from manual data.
        # Only the explicit smann mode should run Sanchez-style episode-end updates.
        """
        Run one SMANN online-conditioning pass using freshly collected replay windows when
        that mode is active.
        """
        if self.agent.fear_model_mode != 'smann':
            return

        try:
            metrics = self.agent.smann.train_with_vicarious_conditioning(
                self.agent.replay_buffer.as_list(),
                self.get_logger(),
            )
        except Exception as exc:
            self.get_logger().warning(
                f'SMANN vicarious-conditioning failed; keeping the current trainer alive: {exc}'
            )
            return

        if bool(metrics.get('trained', False)):
            self.get_logger().info(
                'SMANN online update '
                f"samples={metrics['samples']} loss={float(metrics['loss']):.4f}"
            )

        if self.writer is not None:
            self.writer.add_scalar('smann/vicarious_loss', float(metrics['loss']), self.episode_index)
            self.writer.add_scalar('smann/vicarious_samples', float(metrics['samples']), self.episode_index)

    def _reset_simulation(self) -> None:
        """
        Choose the configured simulator reset strategy for the next episode.
        """
        if self.environment_config.manage_sim_process:
            # The current preferred reset path is a full relaunch so stale controller,
            # sensor, and collision state cannot leak into the next episode.
            self._relaunch_managed_simulation()
            return

        self._pose_teleport_reset()

    def _relaunch_managed_simulation(self) -> None:
        """
        Stop and restart the simulator child process so each episode begins from a clean
        world state.
        """
        self._publish_stop_burst(repeats=2, interval_s=0.05)
        self._stop_managed_sim_process()

        self._start_managed_sim_process()
        self.get_logger().info(
            'Reset simulator for the next episode by fully relaunching '
            'sidewalk_sim.launch.py as a managed child process.'
        )

    def _pose_teleport_reset(self) -> None:
        """
        Fallback reset path that teleports the robot instead of relaunching Gazebo.
        """
        world_control_service = f'/world/{self.environment_config.world_name}/control'
        pose_services = [
            f'/world/{self.environment_config.world_name}/set_pose/blocking',
            f'/world/{self.environment_config.world_name}/set_pose',
        ]

        # Avoid Gazebo world resets here. In this Clearpath Harmonic setup they either
        # are unsupported or rewind sim time, which destabilizes tf / ros2_control.
        self._publish_stop_burst(repeats=3, interval_s=0.05)
        self._call_world_control(world_control_service, 'pause: true', log_failure=False)
        time.sleep(0.10)

        pose_success, target_name, pose_service = self._set_robot_pose(pose_services)

        time.sleep(0.10)
        self._publish_stop_burst(repeats=3, interval_s=0.05)
        self._call_world_control(world_control_service, 'pause: false', log_failure=False)
        time.sleep(0.15)
        self._publish_stop_burst(repeats=3, interval_s=0.05)
        time.sleep(self.environment_config.reset_settle_s)

        if pose_success:
            self.get_logger().info(
                'Reset simulator for the next episode using pose teleport only '
                f"for entity '{target_name}' via {pose_service}."
            )
            return

        self.get_logger().warning(
            'Gazebo pose teleport failed during episode reset; '
            'the robot may remain at the previous collision location.'
        )

    def _managed_sim_launch_command(self) -> str:
        """
        Construct the shell command used to launch the simulator as a managed child process.
        """
        use_sim_time = 'true' if self.environment_config.use_sim_time else 'false'
        rviz = 'true' if self.environment_config.rviz else 'false'
        enable_audio = 'true' if self.environment_config.enable_audio else 'false'
        launch_args = [
            'ros2',
            'launch',
            'fear_jackal_sim',
            'sidewalk_sim.launch.py',
            f'namespace:={self.environment_config.namespace}',
            f'setup_path:={self.environment_config.setup_path}',
            f'use_sim_time:={use_sim_time}',
            f'rviz:={rviz}',
            f'enable_audio:={enable_audio}',
            f'x:={self.environment_config.spawn_x}',
            f'y:={self.environment_config.spawn_y}',
            f'z:={self.environment_config.spawn_z}',
            f'yaw:={self.environment_config.spawn_yaw}',
            f'color_topic:={self.environment_config.color_topic}',
            f'depth_topic:={self.environment_config.depth_topic}',
            f'collision_topic:={self.environment_config.collision_topic}',
        ]
        ros_setup = shlex.quote(self.environment_config.ros_setup_script)
        workspace_setup = shlex.quote(self.environment_config.workspace_setup_script)
        quoted_launch = ' '.join(shlex.quote(part) for part in launch_args)
        return f'source {ros_setup} && source {workspace_setup} && {quoted_launch}'

    def _start_managed_sim_process(self) -> None:
        """
        Spawn the managed simulator child process and cache its handle.
        """
        command = self._managed_sim_launch_command()
        try:
            process = subprocess.Popen(
                ['bash', '-lc', command],
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            self._sim_process = None
            message = 'Unable to start the managed simulation process because bash is unavailable.'
            self.get_logger().error(message)
            raise RuntimeError(message) from exc
        except OSError as exc:
            self._sim_process = None
            message = f'Unable to start the managed simulation process: {exc}'
            self.get_logger().error(message)
            raise RuntimeError(message) from exc

        self._sim_process = process
        time.sleep(1.0)
        if process.poll() is not None:
            self._sim_process = None
            message = f'Managed simulation process exited immediately with code {process.returncode}.'
            self.get_logger().error(message)
            raise RuntimeError(message)

    def _wait_for_process_exit(self, process: subprocess.Popen[str], timeout_s: float) -> bool:
        """
        Wait for a child process to exit while respecting the configured timeout.
        """
        try:
            process.wait(timeout=max(float(timeout_s), 0.0))
            return True
        except subprocess.TimeoutExpired:
            return False

    def _stop_managed_sim_process(self) -> None:
        """
        Stop the current simulator child process as cleanly as possible.
        """
        process = self._sim_process
        self._sim_process = None
        if process is None:
            return

        if process.poll() is not None:
            return

        shutdown_steps = [
            (signal.SIGINT, self.environment_config.sim_shutdown_timeout_s),
            (signal.SIGTERM, max(self.environment_config.sim_shutdown_timeout_s * 0.5, 1.0)),
            (signal.SIGKILL, 1.0),
        ]
        for sig, timeout_s in shutdown_steps:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            except Exception as exc:
                self.get_logger().warning(
                    f'Unable to signal the managed simulation process with {sig.name}: {exc}'
                )
                return

            if self._wait_for_process_exit(process, timeout_s):
                return

        self.get_logger().warning('Managed simulation process did not exit cleanly after SIGKILL.')

    def _publish_stop_burst(self, repeats: int, interval_s: float) -> None:
        """
        Publish several zero-velocity commands so the robot settles before reset or
        shutdown.
        """
        repeats = max(int(repeats), 1)
        interval_s = max(float(interval_s), 0.0)
        for _ in range(repeats):
            self.environment.publish_action(AgentAction())
            if interval_s > 0.0:
                time.sleep(interval_s)

    def _candidate_model_names(self) -> list[str]:
        """
        Return the possible Gazebo entity names the reset logic should try.
        """
        candidates = [
            self.environment_config.model_name,
            f'{self.environment_config.namespace}/robot',
            self.environment_config.namespace,
            'j100-0000',
        ]
        deduplicated = []
        for candidate in candidates:
            if candidate and candidate not in deduplicated:
                deduplicated.append(candidate)
        return deduplicated

    def _set_robot_pose(self, service_names: list[str]) -> tuple[bool, str | None, str | None]:
        """
        Try the known model-name candidates until the pose-set call succeeds.
        """
        yaw = float(self.environment_config.spawn_yaw)
        half_yaw = yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        for target_name in self._candidate_model_names():
            request = (
                f'name: "{target_name}" '
                f'position {{ x: {self.environment_config.spawn_x} y: {self.environment_config.spawn_y} z: {self.environment_config.spawn_z} }} '
                f'orientation {{ x: 0.0 y: 0.0 z: {qz} w: {qw} }}'
            )
            for service_name in service_names:
                if self._call_pose_service(service_name, request, log_failure=False):
                    return True, target_name, service_name
        return False, None, None

    def _call_pose_service(self, service_name: str, request: str, log_failure: bool = True) -> bool:
        """
        Issue the Gazebo pose service request used by teleport resets.
        """
        timeout_ms = max(int(self.environment_config.reset_timeout_s * 1000.0), 1000)
        command = [
            'gz',
            'service',
            '-s',
            service_name,
            '--reqtype',
            'gz.msgs.Pose',
            '--reptype',
            'gz.msgs.Boolean',
            '--timeout',
            str(timeout_ms),
            '--req',
            request,
        ]
        return self._run_gz_service(command, log_label='Gazebo set-pose request', log_failure=log_failure)

    def _call_world_control(self, service_name: str, request: str, log_failure: bool = True) -> bool:
        """
        Issue the Gazebo world-control request used by broader reset helpers.
        """
        timeout_ms = max(int(self.environment_config.reset_timeout_s * 1000.0), 1000)
        command = [
            'gz',
            'service',
            '-s',
            service_name,
            '--reqtype',
            'gz.msgs.WorldControl',
            '--reptype',
            'gz.msgs.Boolean',
            '--timeout',
            str(timeout_ms),
            '--req',
            request,
        ]
        return self._run_gz_service(command, log_label='Gazebo world-control request', log_failure=log_failure)

    def _run_gz_service(self, command: list[str], log_label: str, log_failure: bool = True) -> bool:
        """
        Run the lower-level Gazebo service command and report whether it succeeded.
        """
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            if log_failure:
                self.get_logger().warning('The gz CLI is unavailable, so Gazebo reset could not be requested.')
            return False
        except Exception as exc:
            if log_failure:
                self.get_logger().warning(f'{log_label} failed unexpectedly: {exc}')
            return False

        output = ' '.join(part for part in (result.stdout, result.stderr) if part).lower()
        success = (
            result.returncode == 0
            and 'data: true' in output
            and 'data: false' not in output
            and 'error parsing text-format' not in output
            and 'unable to create request' not in output
            and 'service not available' not in output
            and 'host unreachable' not in output
            and 'not supported' not in output
            and 'timed out' not in output
        )

        if not success and log_failure:
            self.get_logger().warning(
                f'{log_label} failed with code {result.returncode}: {result.stdout}{result.stderr}'
            )

        return success

    def destroy_node(self) -> bool:
        """
        Shut down the trainer and stop any managed simulator child process before releasing
        ROS resources.
        """
        try:
            self._stop_managed_sim_process()
        except KeyboardInterrupt:
            pass
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        return super().destroy_node()


def main(args=None) -> None:
    """
    ROS entrypoint that constructs and spins the trainer node until shutdown.
    """
    rclpy.init(args=args)
    node = FearTrainer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass




