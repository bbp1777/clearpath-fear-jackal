"""
ROS-facing environment wrapper that turns topic traffic into clean observations, actions,
and reset gating for the trainer.
"""
from __future__ import annotations

from collections.abc import Callable
from math import hypot
from typing import Optional

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float64

from fear_jackal_sim.rl_types import AgentAction, EnvironmentConfig, ObservationBundle


class FearEnvironment:
    """
    Caches the latest sensor state and exposes the small environment API used by the trainer
    and agent.
    """
    def __init__(self, node: Node, config: EnvironmentConfig) -> None:
        """
        Initialize subscriptions, publishers, and all episode/reset bookkeeping.
        """
        self._node = node
        self.config = config
        self._episode_steps = 0
        self._latest_color: Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        self._latest_audio = 0.0
        self._latest_goal_coverage = 0.0
        self._latest_collision = False
        self._latest_odom_x = float(self.config.spawn_x)
        self._latest_odom_y = float(self.config.spawn_y)
        self._latest_goal_distance = self._compute_goal_distance(
            self._latest_odom_x,
            self._latest_odom_y,
        )
        self._closest_goal_distance = float(self._latest_goal_distance)
        self._collision_latched = False
        self._freeze_sensor_updates = False
        self._awaiting_post_reset_frames = False
        self._awaiting_post_reset_collision_clear = False
        self._post_reset_color_ready = False
        self._post_reset_depth_ready = False
        self._post_reset_discard_collision_messages_remaining = 0
        self._minimum_sensor_stamp: Optional[Time] = None
        self._post_reset_requires_time_rewind = False
        self._reset_callback: Optional[Callable[[], None]] = None
        # This stores the most recent command so we can refresh it between slower policy ticks.
        self._last_action = AgentAction()

        # All robot motion commands flow through this one publisher so resets and
        # safety-stop bursts can be issued from a single place.
        self._cmd_pub = self._node.create_publisher(TwistStamped, self.config.cmd_vel_topic, 10)
        self._node.create_subscription(Image, self.config.color_topic, self._on_color, 10)
        self._node.create_subscription(Image, self.config.depth_topic, self._on_depth, 10)
        self._node.create_subscription(Float32, self.config.goal_coverage_topic, self._on_goal_coverage, 10)
        self._node.create_subscription(Bool, self.config.collision_topic, self._on_collision, 10)
        self._node.create_subscription(Odometry, self.config.odom_topic, self._on_odom, 10)

        if self.config.enable_audio and self.config.audio_topic:
            self._node.create_subscription(Float64, self.config.audio_topic, self._on_audio, 10)

    def register_reset_callback(self, callback: Callable[[], None]) -> None:
        """
        Register the simulator reset hook owned by the trainer.
        """
        self._reset_callback = callback

    def _is_stale_stamp(self, msg_stamp: Time) -> bool:
        """
        Reject messages that still belong to the pre-reset world.
        """
        if self._minimum_sensor_stamp is None:
            return False

        if self._post_reset_requires_time_rewind:
            rewind_delta_ns = self._minimum_sensor_stamp.nanoseconds - msg_stamp.nanoseconds
            rewind_threshold_ns = int(self.config.sim_time_rewind_threshold_s * 1e9)
            return rewind_delta_ns <= rewind_threshold_ns

        if self.config.manage_sim_process:
            rewind_delta_ns = self._minimum_sensor_stamp.nanoseconds - msg_stamp.nanoseconds
            rewind_threshold_ns = int(self.config.sim_time_rewind_threshold_s * 1e9)
            if rewind_delta_ns > rewind_threshold_ns:
                return False

        return msg_stamp <= self._minimum_sensor_stamp

    def _is_stale_sensor_message(self, msg: Image) -> bool:
        """
        Reject camera frames that still belong to the pre-reset world.
        """
        return self._is_stale_stamp(Time.from_msg(msg.header.stamp))

    def _finalize_post_reset_readiness_if_ready(self) -> None:
        """
        Mark the environment ready once fresh frames and a cleared collision signal have
        arrived.
        """
        if not self._awaiting_post_reset_frames:
            return
        if not self._post_reset_color_ready or not self._post_reset_depth_ready:
            return
        if self._awaiting_post_reset_collision_clear:
            return

        self._awaiting_post_reset_frames = False
        self._minimum_sensor_stamp = None
        self._post_reset_requires_time_rewind = False
        self._node.get_logger().info(
            'Fresh post-reset color/depth frames and a cleared collision signal were received; '
            'environment is ready.'
        )

    def _mark_post_reset_frame(self, sensor_name: str) -> None:
        """
        Record that one of the required post-reset sensor streams is now fresh.
        """
        if not self._awaiting_post_reset_frames:
            return

        if sensor_name == 'color':
            self._post_reset_color_ready = True
        elif sensor_name == 'depth':
            self._post_reset_depth_ready = True

        self._finalize_post_reset_readiness_if_ready()

    def _on_color(self, msg: Image) -> None:
        """
        Cache the newest usable color frame.
        """
        if self._freeze_sensor_updates:
            return
        if self._is_stale_sensor_message(msg):
            return
        self._latest_color = msg
        self._mark_post_reset_frame('color')

    def _on_depth(self, msg: Image) -> None:
        """
        Cache the newest usable depth frame.
        """
        if self._freeze_sensor_updates:
            return
        if self._is_stale_sensor_message(msg):
            return
        self._latest_depth = msg
        self._mark_post_reset_frame('depth')

    def _on_audio(self, msg: Float64) -> None:
        """
        Cache the newest optional audio reading.
        """
        if self._freeze_sensor_updates:
            return
        self._latest_audio = float(msg.data)

    def _compute_goal_distance(self, x_position: float, y_position: float) -> float:
        """
        Return the planar Euclidean distance from the robot to the fixed goal marker.
        """
        return float(hypot(
            x_position - float(self.config.goal_position_x),
            y_position - float(self.config.goal_position_y),
        ))

    def _on_odom(self, msg: Odometry) -> None:
        """
        Cache the newest usable odometry reading and update paper-only goal-distance metrics.
        """
        if self._freeze_sensor_updates:
            return
        stamp = Time.from_msg(msg.header.stamp)
        if self._is_stale_stamp(stamp):
            return
        self._latest_odom_x = float(msg.pose.pose.position.x)
        self._latest_odom_y = float(msg.pose.pose.position.y)
        self._latest_goal_distance = self._compute_goal_distance(self._latest_odom_x, self._latest_odom_y)
        self._closest_goal_distance = min(self._closest_goal_distance, self._latest_goal_distance)

    def _on_goal_coverage(self, msg: Float32) -> None:
        """
        Cache the newest goal-coverage scalar.
        """
        if self._freeze_sensor_updates:
            return
        if self._awaiting_post_reset_frames:
            return
        self._latest_goal_coverage = float(msg.data)

    def _on_collision(self, msg: Bool) -> None:
        """
        Handle collision latching and post-reset collision-message filtering.
        """
        collision = bool(msg.data)

        # Right after a reset, collision booleans from the old episode can still be
        # in flight, so we wait for a clean false before trusting new frames.
        if self._awaiting_post_reset_frames:
            if self._post_reset_discard_collision_messages_remaining > 0:
                self._post_reset_discard_collision_messages_remaining -= 1
                return

            self._latest_collision = collision
            if not collision and self._post_reset_color_ready and self._post_reset_depth_ready:
                self._awaiting_post_reset_collision_clear = False
                self._finalize_post_reset_readiness_if_ready()
            return

        if collision:
            if not self._collision_latched:
                self._node.get_logger().info(
                    'Environment latched a terminal collision event; freezing sensor updates.'
                )
                self.publish_action(AgentAction())
            self._latest_collision = True
            self._collision_latched = True
            self._freeze_sensor_updates = True
            return

        if not self._collision_latched:
            self._latest_collision = False

    def is_ready(self) -> bool:
        """
        Return whether the trainer can safely step the current episode.
        """
        return (
            self._latest_color is not None
            and self._latest_depth is not None
            and not self._awaiting_post_reset_frames
        )

    def _publish_twist(self, action: AgentAction) -> None:
        """
        Build and publish one stamped Twist command from the supplied planar action.
        """
        # This creates the stamped velocity message expected by ros2_control.
        cmd = TwistStamped()
        # This stamps the command with the node's current clock time.
        cmd.header.stamp = self._node.get_clock().now().to_msg()
        # This keeps the command frame tied to the robot base.
        cmd.header.frame_id = 'base_link'
        # This fills in the commanded forward speed.
        cmd.twist.linear.x = float(action.linear_x)
        # This fills in the commanded turn rate.
        cmd.twist.angular.z = float(action.angular_z)
        # This publishes the final message to the robot command topic.
        self._cmd_pub.publish(cmd)

    def publish_action(self, action: AgentAction) -> None:
        """
        Store the latest action choice and publish it immediately.
        """
        # This caches the latest action so the keepalive timer can resend it if needed.
        self._last_action = AgentAction(linear_x=float(action.linear_x), angular_z=float(action.angular_z))
        # This sends the newest command right away.
        self._publish_twist(self._last_action)

    def republish_last_action(self) -> None:
        """
        Re-send the most recent command so the diff-drive controller does not time out
        while the evaluation loop is waiting for the next decision tick.
        """
        # This refreshes the last command using a fresh timestamp.
        self._publish_twist(self._last_action)

    def increment_step(self) -> None:
        """
        Advance the current episode step counter.
        """
        self._episode_steps += 1

    def build_observation(self) -> ObservationBundle:
        """
        Assemble the latest cached state into one ObservationBundle.
        """
        goal_reached = self._latest_goal_coverage >= self.config.goal_completion_threshold
        terminal = bool(self._collision_latched)
        truncated = bool(self._episode_steps >= self.config.max_episode_steps)
        return ObservationBundle(
            color_msg=self._latest_color,
            depth_msg=self._latest_depth,
            audio_level=float(self._latest_audio),
            goal_coverage=float(self._latest_goal_coverage),
            collision=bool(self._collision_latched),
            goal_reached=bool(goal_reached),
            step_index=int(self._episode_steps),
            terminal=terminal,
            truncated=truncated,
        )

    def current_goal_distance(self) -> float:
        """
        Return the most recent robot-to-goal distance in meters for logging only.
        """
        return float(self._latest_goal_distance)

    def closest_goal_distance(self) -> float:
        """
        Return the closest robot-to-goal distance reached in the current episode.
        """
        return float(self._closest_goal_distance)

    def reset(self) -> ObservationBundle:
        """
        Reset episode-local state and trigger the registered simulator reset path.
        """
        reset_stamp = self._node.get_clock().now()

        # Reset both the semantic episode state and the freshness gates so the next
        # episode cannot start until new sensor frames arrive from the reset world.
        self._episode_steps = 0
        self._latest_color = None
        self._latest_depth = None
        self._latest_collision = False
        self._collision_latched = False
        self._freeze_sensor_updates = False
        self._latest_goal_coverage = 0.0
        self._latest_audio = 0.0
        self._latest_odom_x = float(self.config.spawn_x)
        self._latest_odom_y = float(self.config.spawn_y)
        self._latest_goal_distance = self._compute_goal_distance(self._latest_odom_x, self._latest_odom_y)
        self._closest_goal_distance = float(self._latest_goal_distance)
        # This clears the cached action so resets start from a stopped robot command.
        self._last_action = AgentAction()
        self._awaiting_post_reset_frames = True
        self._awaiting_post_reset_collision_clear = True
        self._post_reset_color_ready = False
        self._post_reset_depth_ready = False
        self._post_reset_discard_collision_messages_remaining = (
            max(int(self.config.post_reset_discard_collision_messages), 0)
            if self.config.manage_sim_process
            else 0
        )
        self._minimum_sensor_stamp = reset_stamp
        self._post_reset_requires_time_rewind = (
            bool(self.config.manage_sim_process)
            and reset_stamp.nanoseconds > int(self.config.sim_time_rewind_threshold_s * 1e9)
        )

        if self._reset_callback is not None:
            self._reset_callback()
        else:
            self._node.get_logger().warning(
                'No simulator reset callback is registered yet. Resetting internal episode state only.'
            )

        self._finalize_post_reset_readiness_if_ready()

        return self.build_observation()
