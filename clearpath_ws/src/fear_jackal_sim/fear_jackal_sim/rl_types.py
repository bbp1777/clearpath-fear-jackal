"""
Shared dataclasses that keep actions, observations, transitions, and configuration payloads
explicit across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentAction:
    """
    Dataclass describing one planar velocity command for the Jackal.
    """
    linear_x: float = 0.0
    angular_z: float = 0.0


@dataclass
class ObservationBundle:
    """
    Dataclass describing the latest cached observation and episode flags.
    """
    color_msg: Any = None
    depth_msg: Any = None
    audio_level: float = 0.0
    goal_coverage: float = 0.0
    collision: bool = False
    goal_reached: bool = False
    step_index: int = 0
    terminal: bool = False
    truncated: bool = False

    def feature_vector(self, max_steps: int) -> list[float]:
        """
        Return a tiny fallback feature vector derived from the observation.
        """
        step_fraction = 0.0 if max_steps <= 0 else min(float(self.step_index) / float(max_steps), 1.0)
        return [
            float(self.goal_coverage),
            float(self.collision),
            float(step_fraction),
            1.0 if self.depth_msg is not None else 0.0,
        ]

    def summary(self) -> dict[str, float | bool | int]:
        """
        Serialize the observation into simple scalar fields for archives and logs.
        """
        return {
            'audio_level': float(self.audio_level),
            'goal_coverage': float(self.goal_coverage),
            'collision': bool(self.collision),
            'goal_reached': bool(self.goal_reached),
            'step_index': int(self.step_index),
            'terminal': bool(self.terminal),
            'truncated': bool(self.truncated),
        }


@dataclass
class Transition:
    """
    Dataclass for one RL transition, including optional vision-window labeling data.
    """
    state_summary: dict[str, float | bool | int]
    action: AgentAction
    external_reward: float
    intrinsic_reward: float
    combined_reward: float
    terminal: bool
    truncated: bool
    vision_window: Any = None
    danger_label: int | None = None
    reward_label: int | None = None


@dataclass
class EnvironmentConfig:
    """
    Configuration bundle for topics, resets, and episode limits.
    """
    namespace: str = 'jackal_sidewalk'
    color_topic: str = '/jackal_sidewalk/sensors/camera_0/color/image'
    depth_topic: str = '/jackal_sidewalk/sensors/camera_0/depth/image'
    enable_audio: bool = False
    audio_topic: str = ''
    goal_coverage_topic: str = '/jackal_sidewalk/goal/coverage'
    collision_topic: str = '/jackal_sidewalk/collision'
    cmd_vel_topic: str = '/jackal_sidewalk/cmd_vel'
    world_name: str = 'mini_sidewalk'
    model_name: str = 'jackal_sidewalk/robot'
    setup_path: str = '/workspaces/clearpath_docker/sim_setup'
    use_sim_time: bool = True
    rviz: bool = False
    manage_sim_process: bool = True
    ros_setup_script: str = '/opt/ros/jazzy/setup.bash'
    workspace_setup_script: str = '/workspaces/clearpath_docker/clearpath_ws/install/setup.bash'
    spawn_x: float = -14.0
    spawn_y: float = 0.0
    spawn_z: float = 0.20
    spawn_yaw: float = 0.0
    max_episode_steps: int = 400
    goal_completion_threshold: float = 0.30
    reset_timeout_s: float = 5.0
    reset_settle_s: float = 1.0
    sim_relaunch_timeout_s: float = 45.0
    sim_shutdown_timeout_s: float = 10.0
    sim_time_rewind_threshold_s: float = 2.0
    post_reset_discard_collision_messages: int = 2


@dataclass
class AgentConfig:
    """
    Configuration bundle for reward shaping, policy learning, and fear-model selection.
    """
    lookback: int = 3
    reward_mode: str = 'combined'
    external_reward_scale: float = 1.0
    intrinsic_reward_scale: float = 1.0
    replay_buffer_capacity: int = 5000
    policy_hidden_dim: int = 64
    action_linear_speed: float = 0.40
    action_angular_speed: float = 0.75
    use_policy_network: bool = True
    policy_learning_rate: float = 3.0e-4
    value_learning_rate: float = 1.0e-3
    policy_temperature: float = 1.35
    exploration_epsilon: float = 0.10
    ppo_update_epochs: int = 4
    ppo_clip_epsilon: float = 0.20
    discount_factor: float = 0.99
    entropy_coefficient: float = 0.03
    critic_loss_coefficient: float = 0.5
    gradient_clip_norm: float = 1.0
    goal_progress_scale: float = 0.0
    goal_alignment_scale: float = 0.0
    step_penalty: float = 0.0
    collision_penalty: float = 0.0
    goal_reward_bonus: float = 1.0
    fear_model_mode: str = 'smann'
    manual_memory_dataset_dir: str = ''
    manual_memory_bank_path: str = ''
    memory_similarity_image_size: int = 84
    memory_similarity_depth_clip_m: float = 5.0
    smann_checkpoint: str = ''
    smann_dataset_dir: str = '/workspaces/clearpath_docker/clearpath_ws/logs/rodney_dataset'
    fear_repo_path: str = '/workspaces/Behavior-Intrinsic-Fear-main/CarRacingTesting'
    sanchez_upstream_repo: str = 'https://github.com/ras8047/Behavior-Intrinsic-Fear'
    sanchez_upstream_commit: str = ''
    smann_image_size: int = 84
    smann_fear_threshold: float = 0.50
    fear_reactive_policy: bool = False
    fear_reactive_linear_speed: float = 0.12
    fear_reactive_turn_speed: float = 0.90


@dataclass
class TrainerConfig:
    """
    Configuration bundle for trainer cadence, logging, and archive behavior.
    """
    control_period_s: float = 0.25
    train_every_n_steps: int = 64
    vicarious_update_every_n_episodes: int = 1
    tensorboard_log_dir: str = '/workspaces/clearpath_docker/clearpath_ws/logs/tensorboard'
    clear_buffer_on_reset: bool = False
    enable_online_smann_updates: bool = False
    # This flag disables PPO weight updates during live evaluation runs.
    evaluation_only: bool = False
    run_name: str = 'fear_trainer'
    archive_episodes: bool = True
    episode_archive_dir: str = '/workspaces/clearpath_docker/clearpath_ws/logs/episode_archives'
