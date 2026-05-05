# This imports future annotations so forward type hints stay simple.
from __future__ import annotations

# This imports dataclass so config and message containers stay simple.
from dataclasses import dataclass
# This imports Any so ROS message fields can stay lightweight in the dataclasses.
from typing import Any


# This describes one planar motion command for the Jackal.
@dataclass
class AgentAction:
    """Store one simple forward and turn command for the robot."""

    # This stores the forward speed.
    linear_x: float = 0.0
    # This stores the turn rate.
    angular_z: float = 0.0


# This describes the latest sensor snapshot and episode flags.
@dataclass
class ObservationBundle:
    """Store the latest RGB-D messages, task scalars, and terminal flags."""

    # This stores the latest color image message.
    color_msg: Any = None
    # This stores the latest depth image message.
    depth_msg: Any = None
    # This stores the latest green goal coverage value.
    goal_coverage: float = 0.0
    # This stores the latest collision flag.
    collision: bool = False
    # This stores whether the goal has been reached.
    goal_reached: bool = False
    # This stores the current episode step index.
    step_index: int = 0
    # This stores whether the episode is terminal.
    terminal: bool = False
    # This stores whether the episode was truncated.
    truncated: bool = False


# This groups the ROS topics and simulator settings used by the environment.
@dataclass
class EnvironmentConfig:
    """Store the topic names, simulator settings, and episode limits."""

    # This stores the robot namespace.
    namespace: str = 'jackal_sidewalk'
    # This stores the color topic name.
    color_topic: str = '/jackal_sidewalk/sensors/camera_0/color/image'
    # This stores the depth topic name.
    depth_topic: str = '/jackal_sidewalk/sensors/camera_0/depth/image'
    # This stores the goal coverage topic name.
    goal_coverage_topic: str = '/jackal_sidewalk/goal/coverage'
    # This stores the collision topic name.
    collision_topic: str = '/jackal_sidewalk/collision'
    # This stores the command velocity topic name.
    cmd_vel_topic: str = '/jackal_sidewalk/cmd_vel'
    # This stores the world name.
    world_name: str = 'mini_sidewalk'
    # This stores the Clearpath setup path.
    setup_path: str = '/workspaces/clearpath_docker/sim_setup'
    # This stores whether sim time is enabled.
    use_sim_time: bool = True
    # This stores the spawn x position.
    spawn_x: float = -14.0
    # This stores the spawn y position.
    spawn_y: float = 0.0
    # This stores the spawn z position.
    spawn_z: float = 0.2
    # This stores the spawn yaw angle.
    spawn_yaw: float = 0.0
    # This stores the episode length cap.
    max_episode_steps: int = 300
    # This stores the success threshold for green goal coverage.
    goal_completion_threshold: float = 0.3


# This groups the SMANN and controller settings used by the agent.
@dataclass
class AgentConfig:
    """Store the frozen SMANN settings, behavior mode, and speed values."""

    # This stores the reward comparison mode.
    reward_mode: str = 'combined'
    # This stores the checkpoint directory for the offline-trained SMANN weights.
    smann_checkpoint: str = ''
    # This stores the Behavior-Intrinsic-Fear source directory.
    fear_repo_path: str = '/workspaces/clearpath_docker/Behavior-Intrinsic-Fear-main/CarRacingTesting'
    # This stores the square image size used by the SMANN model.
    smann_image_size: int = 84
    # This stores the SMANN lookback value.
    lookback: int = 3
    # This stores the fear threshold used for evaluation sweeps.
    smann_fear_threshold: float = 0.50
    # This stores the normal forward speed.
    action_linear_speed: float = 0.35
    # This stores the normal turn speed.
    action_turn_speed: float = 0.75
    # This stores the cautious fear forward speed.
    fear_linear_speed: float = 0.10
    # This stores the cautious fear turn speed.
    fear_turn_speed: float = 0.90


# This groups the evaluator loop and logging settings.
@dataclass
class EvaluatorConfig:
    """Store the main loop period and log output path for the evaluator node."""

    # This stores the main evaluation loop period.
    control_period_s: float = 0.1
    # This stores the directory used for episode summaries.
    episode_log_dir: str = '/workspaces/clearpath_docker/clearpath_ws/logs/jackal_smann_eval'
