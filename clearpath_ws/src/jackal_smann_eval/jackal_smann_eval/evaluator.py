# This imports future annotations so forward type hints stay simple.
from __future__ import annotations

# This imports json so episode summaries can be saved to disk.
import json
# This imports os so log directories can be created and checked.
import os
# This imports time so startup readiness can settle before the first command is sent.
import time

# This imports rclpy for the ROS node entrypoint.
import rclpy
# This imports Node for the ROS node base class.
from rclpy.node import Node

# This imports the frozen evaluation agent.
from jackal_smann_eval.agent import Agent
# This imports the environment wrapper.
from jackal_smann_eval.environment import Environment
# This imports the shared types.
from jackal_smann_eval.types import AgentAction, AgentConfig, EnvironmentConfig, EvaluatorConfig


# This owns the frozen single-episode evaluation loop.
class JackalSMANNEvaluator(Node):
    """Run one frozen SMANN evaluation episode per launch and let the user reset manually."""

    # This reads parameters, builds the environment and agent, and starts the main loop timer.
    def __init__(self) -> None:
        """Declare parameters inline, build the runtime objects, and start the evaluation loop."""

        # This names the ROS node.
        super().__init__('jackal_smann_evaluator')

        # This declares all evaluator parameters except use_sim_time in one call.
        self.declare_parameters('', [
            ('namespace', 'jackal_sidewalk'),
            ('color_topic', '/jackal_sidewalk/sensors/camera_0/color/image'),
            ('depth_topic', '/jackal_sidewalk/sensors/camera_0/depth/image'),
            ('goal_coverage_topic', '/jackal_sidewalk/goal/coverage'),
            ('collision_topic', '/jackal_sidewalk/collision'),
            ('cmd_vel_topic', '/jackal_sidewalk/cmd_vel'),
            ('setup_path', '/workspaces/clearpath_docker/sim_setup'),
            ('world_name', 'mini_sidewalk'),
            ('spawn_x', -14.0),
            ('spawn_y', 0.0),
            ('spawn_z', 0.2),
            ('spawn_yaw', 0.0),
            ('max_episode_steps', 1000),
            ('goal_completion_threshold', 0.3),
            ('control_period_s', 1.0),
            ('reward_mode', 'combined'),
            ('smann_checkpoint', '/workspaces/clearpath_docker/clearpath_ws/logs/smann_training/smann_grid/final_selected/weights'),
            ('fear_repo_path', '/workspaces/clearpath_docker/Behavior-Intrinsic-Fear-main/CarRacingTesting'),
            ('smann_image_size', 84),
            ('lookback', 3),
            ('smann_fear_threshold', 0.50),
            ('action_linear_speed', 0.35),
            ('action_turn_speed', 0.75),
            ('fear_linear_speed', 0.20),
            ('fear_turn_speed', 0.70),
            ('episode_log_dir', '/workspaces/clearpath_docker/clearpath_ws/logs/jackal_smann_eval'),
        ])

        # This declares use_sim_time only when ROS has not already done it.
        if not self.has_parameter('use_sim_time'):
            # This declares the sim-time parameter.
            self.declare_parameter('use_sim_time', True)

        # This packs the environment parameters into one config object.
        self.environment_config = EnvironmentConfig(
            namespace=str(self.get_parameter('namespace').value),
            color_topic=str(self.get_parameter('color_topic').value),
            depth_topic=str(self.get_parameter('depth_topic').value),
            goal_coverage_topic=str(self.get_parameter('goal_coverage_topic').value),
            collision_topic=str(self.get_parameter('collision_topic').value),
            cmd_vel_topic=str(self.get_parameter('cmd_vel_topic').value),
            world_name=str(self.get_parameter('world_name').value),
            setup_path=str(self.get_parameter('setup_path').value),
            use_sim_time=bool(self.get_parameter('use_sim_time').value),
            spawn_x=float(self.get_parameter('spawn_x').value),
            spawn_y=float(self.get_parameter('spawn_y').value),
            spawn_z=float(self.get_parameter('spawn_z').value),
            spawn_yaw=float(self.get_parameter('spawn_yaw').value),
            max_episode_steps=int(self.get_parameter('max_episode_steps').value),
            goal_completion_threshold=float(self.get_parameter('goal_completion_threshold').value),
        )

        # This packs the agent parameters into one config object.
        self.agent_config = AgentConfig(
            reward_mode=str(self.get_parameter('reward_mode').value),
            smann_checkpoint=str(self.get_parameter('smann_checkpoint').value),
            fear_repo_path=str(self.get_parameter('fear_repo_path').value),
            smann_image_size=int(self.get_parameter('smann_image_size').value),
            lookback=int(self.get_parameter('lookback').value),
            smann_fear_threshold=float(self.get_parameter('smann_fear_threshold').value),
            action_linear_speed=float(self.get_parameter('action_linear_speed').value),
            action_turn_speed=float(self.get_parameter('action_turn_speed').value),
            fear_linear_speed=float(self.get_parameter('fear_linear_speed').value),
            fear_turn_speed=float(self.get_parameter('fear_turn_speed').value),
        )

        # This packs the evaluator loop parameters into one config object.
        self.evaluator_config = EvaluatorConfig(
            control_period_s=float(self.get_parameter('control_period_s').value),
            episode_log_dir=str(self.get_parameter('episode_log_dir').value),
        )

        # This creates the log directory when needed.
        os.makedirs(self.evaluator_config.episode_log_dir, exist_ok=True)
        # This builds the environment wrapper.
        self.environment = Environment(self, self.environment_config)
        # This builds the frozen evaluation agent.
        self.agent = Agent(self.agent_config, self.get_logger())
        # This loads the offline SMANN checkpoint.
        self.agent.init()
        # This stores the global step counter.
        self.global_step = 0
        # This stores the episode counter.
        self.episode_index = 1
        # This tracks whether an episode is currently active.
        self.episode_active = False
        # This tracks whether the episode has finished.
        self.episode_finished = False
        # This stores when the environment first became ready.
        self.ready_since = None
        # This stores the current episode external reward sum.
        self.episode_external_reward = 0.0
        # This stores the current episode intrinsic reward sum.
        self.episode_intrinsic_reward = 0.0
        # This stores the current episode combined reward sum.
        self.episode_combined_reward = 0.0
        self.episode_fear_sum = 0.0
        self.episode_fear_max = 0.0
        self.episode_fear_count = 0
        # This creates the main evaluation loop timer.
        self.timer = self.create_timer(self.evaluator_config.control_period_s, self._tick)
        # This reports the frozen evaluation settings.
        self.get_logger().info(
            f'Frozen evaluation mode is enabled with reward_mode={self.agent_config.reward_mode} '
            f'fear_threshold={self.agent_config.smann_fear_threshold:.3f} '
            f'checkpoint={self.agent_config.smann_checkpoint}'
        )

    # This runs one evaluator state-machine tick.
    def _tick(self) -> None:
        """Start one episode after the sensors settle, step the frozen controller, then stop and wait for a manual reset."""

        # This returns after the episode is finished.
        if self.episode_finished:
            # This keeps the node idle after one completed trial.
            return

        # This waits for fresh RGB-D data before starting the episode.
        if not self.episode_active:
            # This clears the ready timer when the environment is not ready.
            if not self.environment.is_ready():
                # This forgets any earlier ready timestamp.
                self.ready_since = None
                # This waits for ready data.
                return

            # This stores when the environment first looked ready.
            if self.ready_since is None:
                # This records the first ready time.
                self.ready_since = time.monotonic()
                # This waits one more cycle before starting.
                return

            # This waits for the startup state to settle before sending the first command.
            if time.monotonic() - self.ready_since < 2.0:
                # This waits for the controller stack to finish activating.
                return

            # This resets the agent memory.
            self.agent.reset()
            # This clears the episode external reward.
            self.episode_external_reward = 0.0
            # This clears the episode intrinsic reward.
            self.episode_intrinsic_reward = 0.0
            # This clears the episode combined reward.
            self.episode_combined_reward = 0.0
            self.episode_fear_sum = 0.0
            self.episode_fear_max = 0.0
            self.episode_fear_count = 0
            # This marks the episode as active.
            self.episode_active = True
            # This logs the episode start.
            self.get_logger().info(f'Started evaluation episode {self.episode_index}.')
            # This stops the tick after starting the episode.
            return

        # This builds the latest observation.
        observation = self.environment.build_observation()
        # This caches the observation for lookback scoring.
        self.agent.cache_observation(observation)
        # This chooses the next frozen evaluation action.
        action = self.agent.act(observation)
        # This publishes the chosen action once for this control step.
        self.environment.publish_action(action)
        # This advances the step count.
        self.environment.increment_step()
        # This computes the reward totals for logging.
        external_reward, intrinsic_reward, combined_reward = self.agent.compute_rewards(observation)
        # This accumulates external reward.
        self.episode_external_reward += float(external_reward)
        # This accumulates intrinsic reward.
        self.episode_intrinsic_reward += float(intrinsic_reward)
        # This accumulates combined reward.
        self.episode_combined_reward += float(combined_reward)
        fear_score = float(self.agent.last_fear_score)
        self.episode_fear_sum += fear_score
        self.episode_fear_max = max(self.episode_fear_max, fear_score)
        self.episode_fear_count += 1
        # This advances the global step counter.
        self.global_step += 1

        # This prints a compact periodic status line.
        if observation.step_index % 10 == 0:
            # This logs the current evaluation status.
            self.get_logger().info(
                f'step={observation.step_index} '
                f'goal={observation.goal_coverage:.3f} '
                f'collision={observation.collision} '
                f'fear={self.agent.last_fear_score:.6f} '
                f'threshold={self.agent_config.smann_fear_threshold:.3f} '
                f'mode={self.agent_config.reward_mode}'
            )

        # This finishes the episode on success, terminal failure, or truncation.
        if observation.goal_reached or observation.terminal or observation.truncated:
            # This stops the robot at the episode boundary.
            self.environment.publish_action(AgentAction())
            # This marks the episode as inactive.
            self.episode_active = False
            # This marks the episode as finished.
            self.episode_finished = True

            mean_fear_score = self.episode_fear_sum / max(self.episode_fear_count, 1)
            # This builds the episode summary dictionary.
            summary = {
                'episode': int(self.episode_index),
                'reward_mode': str(self.agent_config.reward_mode),
                'fear_threshold': float(self.agent_config.smann_fear_threshold),
                'external_reward': float(self.episode_external_reward),
                'intrinsic_reward': float(self.episode_intrinsic_reward),
                'combined_reward': float(self.episode_combined_reward),
                'fear_score_mean': float(mean_fear_score),
                'fear_score_max': float(self.episode_fear_max),
                'goal_coverage': float(observation.goal_coverage),
                'goal_reached': bool(observation.goal_reached),
                'collision': bool(observation.collision),
                'terminal': bool(observation.terminal),
                'truncated': bool(observation.truncated),
                'steps': int(observation.step_index),
            }

            # This builds the summary file path.
            summary_path = os.path.join(self.evaluator_config.episode_log_dir, f'episode_{self.episode_index:06d}.json')

            # This writes the summary JSON file.
            with open(summary_path, 'w', encoding='ascii') as handle:
                # This saves the summary contents.
                json.dump(summary, handle, indent=2)

            # This logs the episode summary.
            self.get_logger().info(
                f'Episode complete episode={self.episode_index} '
                f'ext={self.episode_external_reward:.3f} '
                f'int={self.episode_intrinsic_reward:.3f} '
                f'combined={self.episode_combined_reward:.3f} '
                f'mean_fear={mean_fear_score:.3f} '
                f'max_fear={self.episode_fear_max:.3f} '
                f'path={summary_path}'
            )
            # This logs the manual reset note.
            self.get_logger().info('Evaluation run is complete. Reset the sim manually before starting the next run.')

    # This releases the ROS node resources.
    def destroy_node(self) -> bool:
        """Release the ROS node resources for the evaluator."""

        # This releases the ROS node resources.
        return super().destroy_node()


# This starts and spins the evaluator node.
def main(args=None) -> None:
    """Create the evaluator node and spin it with the default ROS executor."""

    # This starts the ROS client library.
    rclpy.init(args=args)
    # This creates the evaluator node.
    node = JackalSMANNEvaluator()

    try:
        # This spins the evaluator until shutdown.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This ignores Ctrl-C shutdowns.
        pass
    finally:
        try:
            # This destroys the evaluator node cleanly.
            node.destroy_node()
        except Exception:
            # This ignores shutdown cleanup errors.
            pass
        try:
            # This shuts ROS down cleanly.
            rclpy.shutdown()
        except Exception:
            # This ignores shutdown cleanup errors.
            pass
