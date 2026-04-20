# This imports future annotations so forward type hints stay simple.
from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from ros_gz_interfaces.msg import Contacts
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

from fear_jackal_sim.vision_utils import ImageDecodingError, compute_green_goal_coverage, rgb_image_to_numpy


class FearGoalMonitor(Node):
    # Publish goal coverage and simulator-grounded terminal collision signals.

    def __init__(self) -> None:
        super().__init__('fear_goal_monitor')

        self.declare_parameter('color_topic', '/jackal_sidewalk/sensors/camera_0/color/image')
        self.declare_parameter('depth_topic', '/jackal_sidewalk/sensors/camera_0/depth/image')
        self.declare_parameter('cmd_vel_topic', '/jackal_sidewalk/cmd_vel')
        self.declare_parameter('odom_topic', '/jackal_sidewalk/platform/odom')
        self.declare_parameter('goal_coverage_topic', '/jackal_sidewalk/goal/coverage')
        self.declare_parameter('collision_topic', '/jackal_sidewalk/collision')
        self.declare_parameter('goal_min_green', 200)
        self.declare_parameter('goal_max_red', 90)
        self.declare_parameter('goal_max_blue', 90)
        self.declare_parameter('goal_min_green_minus_red', 100)
        self.declare_parameter('goal_min_green_minus_blue', 100)
        self.declare_parameter(
            'terminal_contact_topics',
            [
                '/jackal_sidewalk/sim/contacts/grass_left',
                '/jackal_sidewalk/sim/contacts/grass_right',
                '/jackal_sidewalk/sim/contacts/box_01',
                '/jackal_sidewalk/sim/contacts/box_02',
                '/jackal_sidewalk/sim/contacts/box_03',
                '/jackal_sidewalk/sim/contacts/box_04',
            ],
        )
        self.declare_parameter(
            'terminal_touch_topics',
            [
                '/jackal_sidewalk/sim/touched/grass_left/touched',
                '/jackal_sidewalk/sim/touched/grass_right/touched',
                '/jackal_sidewalk/sim/touched/box_01/touched',
                '/jackal_sidewalk/sim/touched/box_02/touched',
                '/jackal_sidewalk/sim/touched/box_03/touched',
                '/jackal_sidewalk/sim/touched/box_04/touched',
            ],
        )
        self.declare_parameter('use_odometry_collision_fallback', False)
        self.declare_parameter('sidewalk_center_y', 0.0)
        self.declare_parameter('sidewalk_half_width_m', 1.30)
        self.declare_parameter('off_sidewalk_margin_m', 0.02)
        self.declare_parameter('off_sidewalk_hold_s', 0.10)
        self.declare_parameter('robot_half_width_m', 0.24)
        self.declare_parameter('robot_half_length_m', 0.32)
        self.declare_parameter('obstacle_contact_half_width_m', 0.12)
        self.declare_parameter('obstacle_contact_half_length_m', 0.12)
        self.declare_parameter('motion_state_timeout_s', 1.0)
        self.declare_parameter('sim_time_jump_tolerance_s', 1.0)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        goal_topic = self.get_parameter('goal_coverage_topic').value
        collision_topic = self.get_parameter('collision_topic').value

        self._goal_min_green = int(self.get_parameter('goal_min_green').value)
        self._goal_max_red = int(self.get_parameter('goal_max_red').value)
        self._goal_max_blue = int(self.get_parameter('goal_max_blue').value)
        self._goal_min_green_minus_red = int(self.get_parameter('goal_min_green_minus_red').value)
        self._goal_min_green_minus_blue = int(self.get_parameter('goal_min_green_minus_blue').value)
        self._terminal_contact_topics = [
            str(topic) for topic in self.get_parameter('terminal_contact_topics').value
        ]
        self._terminal_touch_topics = [
            str(topic) for topic in self.get_parameter('terminal_touch_topics').value
        ]
        self._use_odometry_collision_fallback = bool(self.get_parameter('use_odometry_collision_fallback').value)
        self._sidewalk_center_y = float(self.get_parameter('sidewalk_center_y').value)
        self._sidewalk_half_width = float(self.get_parameter('sidewalk_half_width_m').value)
        self._off_sidewalk_margin = float(self.get_parameter('off_sidewalk_margin_m').value)
        self._off_sidewalk_hold_duration = Duration(seconds=float(self.get_parameter('off_sidewalk_hold_s').value))
        self._robot_half_width = float(self.get_parameter('robot_half_width_m').value)
        self._robot_half_length = float(self.get_parameter('robot_half_length_m').value)
        self._obstacle_contact_half_width = float(self.get_parameter('obstacle_contact_half_width_m').value)
        self._obstacle_contact_half_length = float(self.get_parameter('obstacle_contact_half_length_m').value)
        self._motion_state_timeout = Duration(seconds=float(self.get_parameter('motion_state_timeout_s').value))
        self._sim_time_jump_tolerance = Duration(seconds=float(self.get_parameter('sim_time_jump_tolerance_s').value))

        self._obstacle_boxes = [
            (-8.0, 0.55, 0.7, 0.7),
            (-2.5, -0.45, 0.8, 0.8),
            (3.5, 0.0, 0.7, 0.7),
            (8.5, -0.55, 0.7, 0.7),
        ]
        self._latest_odom_x = 0.0
        self._latest_odom_y = self._sidewalk_center_y
        self._latest_odom_stamp: Time | None = None
        self._last_sim_stamp: Time | None = None
        self._off_sidewalk_since: Time | None = None
        self._collision_active = False

        self._coverage_pub = self.create_publisher(Float32, goal_topic, 10)
        self._collision_pub = self.create_publisher(Bool, collision_topic, 10)
        self.create_subscription(Image, color_topic, self._on_color, 10)
        self.create_subscription(Image, depth_topic, self._on_depth, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self._terminal_contact_subscriptions = [
            self.create_subscription(
                Contacts,
                contact_topic,
                lambda msg, topic=contact_topic: self._on_terminal_contact(msg, topic),
                10,
            )
            for contact_topic in self._terminal_contact_topics
        ]
        self._terminal_touch_subscriptions = [
            self.create_subscription(
                Bool,
                touch_topic,
                lambda msg, topic=touch_topic: self._on_terminal_touch(msg, topic),
                10,
            )
            for touch_topic in self._terminal_touch_topics
        ]

        self.get_logger().info(
            f'Goal monitor listening on color={color_topic} depth={depth_topic} and publishing '
            f'coverage={goal_topic} collision={collision_topic}; terminal contacts={self._terminal_contact_topics}; '
            f'terminal touches={self._terminal_touch_topics}'
        )

    def _on_color(self, msg: Image) -> None:
        self._handle_sim_time_stamp(Time.from_msg(msg.header.stamp))
        try:
            image = rgb_image_to_numpy(msg)
        except ImageDecodingError as exc:
            self.get_logger().warning(str(exc))
            return
        coverage = compute_green_goal_coverage(
            image,
            min_green=self._goal_min_green,
            max_red=self._goal_max_red,
            max_blue=self._goal_max_blue,
            min_green_minus_red=self._goal_min_green_minus_red,
            min_green_minus_blue=self._goal_min_green_minus_blue,
        )
        self._coverage_pub.publish(Float32(data=float(coverage)))

    def _on_odom(self, msg: Odometry) -> None:
        stamp = Time.from_msg(msg.header.stamp)
        self._handle_sim_time_stamp(stamp)
        self._latest_odom_x = float(msg.pose.pose.position.x)
        self._latest_odom_y = float(msg.pose.pose.position.y)
        self._latest_odom_stamp = stamp
        self._publish_collision_state(stamp)

    def _on_depth(self, msg: Image) -> None:
        stamp = Time.from_msg(msg.header.stamp)
        self._handle_sim_time_stamp(stamp)
        self._publish_collision_state(stamp)

    def _on_terminal_contact(self, msg: Contacts, topic: str) -> None:
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            self._handle_sim_time_stamp(Time.from_msg(msg.header.stamp))
        if not msg.contacts:
            return
        if not self._collision_active:
            self.get_logger().info(
                f'Publishing terminal collision event reason=sim_contact topic={topic} contacts={len(msg.contacts)}'
            )
        self._collision_active = True
        self._collision_pub.publish(Bool(data=True))

    def _on_terminal_touch(self, msg: Bool, topic: str) -> None:
        if not msg.data:
            return
        if not self._collision_active:
            self.get_logger().info(f'Publishing terminal collision event reason=sim_touch topic={topic}')
        self._collision_active = True
        self._collision_pub.publish(Bool(data=True))

    def _publish_collision_state(self, stamp: Time) -> None:
        # Terminal collisions stay latched for the episode. Managed relaunches
        # clear this latch through the sim-time rewind path below.
        if self._collision_active:
            self._collision_pub.publish(Bool(data=True))
            return

        if not self._use_odometry_collision_fallback:
            self._collision_pub.publish(Bool(data=False))
            return

        obstacle_collision = self._compute_obstacle_collision(stamp)
        off_sidewalk_collision = self._compute_off_sidewalk_collision(stamp)
        collision = bool(obstacle_collision or off_sidewalk_collision)

        if collision and not self._collision_active:
            reasons = []
            if obstacle_collision:
                reasons.append('obstacle')
            if off_sidewalk_collision:
                reasons.append('off_sidewalk')
            reason_text = '+'.join(reasons)
            self.get_logger().info(
                f'Publishing terminal collision event reasons={reason_text} '
                f'odom_x={self._latest_odom_x:.3f} odom_y={self._latest_odom_y:.3f}'
            )
        self._collision_active = collision
        self._collision_pub.publish(Bool(data=collision))

    def _handle_sim_time_stamp(self, stamp: Time) -> None:
        if self._last_sim_stamp is not None:
            tolerance_ns = self._sim_time_jump_tolerance.nanoseconds
            if stamp.nanoseconds + tolerance_ns < self._last_sim_stamp.nanoseconds:
                self._latest_odom_x = 0.0
                self._latest_odom_y = self._sidewalk_center_y
                self._latest_odom_stamp = None
                self._off_sidewalk_since = None
                if self._collision_active:
                    self._collision_active = False
                    self._collision_pub.publish(Bool(data=False))
                self.get_logger().debug(
                    'Detected simulation clock jump backwards; cleared cached motion state for the relaunched environment.'
                )
        self._last_sim_stamp = stamp

    def _motion_state_is_fresh(self, stamp: Time | None, now: Time) -> bool:
        if stamp is None:
            return False
        return bool(now.nanoseconds - stamp.nanoseconds <= self._motion_state_timeout.nanoseconds)

    def _compute_obstacle_collision(self, now: Time) -> bool:
        if not self._motion_state_is_fresh(self._latest_odom_stamp, now):
            return False
        for center_x, center_y, size_x, size_y in self._obstacle_boxes:
            overlaps_x = abs(self._latest_odom_x - center_x) <= ((size_x / 2.0) + self._obstacle_contact_half_length)
            overlaps_y = abs(self._latest_odom_y - center_y) <= ((size_y / 2.0) + self._obstacle_contact_half_width)
            if overlaps_x and overlaps_y:
                return True
        return False

    def _compute_off_sidewalk_collision(self, now: Time) -> bool:
        if not self._motion_state_is_fresh(self._latest_odom_stamp, now):
            self._off_sidewalk_since = None
            return False
        lateral_offset = abs(self._latest_odom_y - self._sidewalk_center_y)
        touching_non_sidewalk = (
            lateral_offset + self._robot_half_width
            >= (self._sidewalk_half_width - self._off_sidewalk_margin)
        )
        if not touching_non_sidewalk:
            self._off_sidewalk_since = None
            return False
        if self._off_sidewalk_since is None:
            self._off_sidewalk_since = now
            return False
        return bool(now - self._off_sidewalk_since >= self._off_sidewalk_hold_duration)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FearGoalMonitor()
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
