"""
Diagnostics node that reports whether RGB, depth, and optional audio streams are arriving
from the simulator.
"""
from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64


class FearSensorSubscriber(Node):
    """Collects simulated RealSense data and optional fear-audio inputs."""

    def __init__(self) -> None:
        """
        Declare topics and timers, then subscribe to the requested sensor streams.
        """
        super().__init__('fear_sensor_subscriber')

        self.declare_parameter('color_topic', '/jackal_sidewalk/sensors/camera_0/color/image')
        self.declare_parameter('depth_topic', '/jackal_sidewalk/sensors/camera_0/depth/image')
        self.declare_parameter('collision_topic', '/jackal_sidewalk/collision')
        self.declare_parameter('audio_enabled', False)
        self.declare_parameter('audio_topic', '')
        self.declare_parameter('report_period_s', 2.0)
        self.declare_parameter('danger_audio_threshold', 0.35)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        collision_topic = self.get_parameter('collision_topic').value
        self._audio_enabled = bool(self.get_parameter('audio_enabled').value)
        audio_topic = self.get_parameter('audio_topic').value
        report_period_s = float(self.get_parameter('report_period_s').value)

        self._danger_audio_threshold = float(self.get_parameter('danger_audio_threshold').value)
        self._report_period = Duration(seconds=report_period_s)
        self._last_report_time = self.get_clock().now()
        self._terminal_latched = False

        self._latest_color: Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        self._latest_audio: Optional[Float64] = None

        self._color_arrival_time = None
        self._depth_arrival_time = None
        self._audio_arrival_time = None

        self.create_subscription(Image, color_topic, self._on_color, 10)
        self.create_subscription(Image, depth_topic, self._on_depth, 10)
        self.create_subscription(Bool, collision_topic, self._on_collision, 10)
        if self._audio_enabled and audio_topic:
            self.create_subscription(Float64, audio_topic, self._on_audio, 10)

        self.create_timer(0.25, self._report_status)

        if self._audio_enabled and audio_topic:
            self.get_logger().info(
                'Listening for RealSense and audio inputs on '
                f'color={color_topic}, depth={depth_topic}, audio={audio_topic}'
            )
        else:
            self.get_logger().info(
                'Listening for RealSense inputs on '
                f'color={color_topic}, depth={depth_topic}'
            )

    def _on_collision(self, msg: Bool) -> None:
        """
        Pause or resume status reporting at episode boundaries.
        """
        collision = bool(msg.data)
        if collision and not self._terminal_latched:
            self._terminal_latched = True
            self.get_logger().info('Terminal collision latched; stopping sensor snapshot updates for this episode.')
        elif not collision and self._terminal_latched:
            self._terminal_latched = False
            self.get_logger().info('Collision latch cleared; resuming sensor snapshot updates.')

    def _on_color(self, msg: Image) -> None:
        """
        Cache the latest color frame arrival.
        """
        if self._terminal_latched:
            return
        self._latest_color = msg
        self._color_arrival_time = self.get_clock().now()

    def _on_depth(self, msg: Image) -> None:
        """
        Cache the latest depth frame arrival.
        """
        if self._terminal_latched:
            return
        self._latest_depth = msg
        self._depth_arrival_time = self.get_clock().now()

    def _on_audio(self, msg: Float64) -> None:
        """
        Cache the latest audio value and arrival.
        """
        if self._terminal_latched:
            return
        self._latest_audio = msg
        self._audio_arrival_time = self.get_clock().now()

    def _age_seconds(self, arrival_time) -> Optional[float]:
        """
        Convert a stored arrival time into an age in seconds.
        """
        if arrival_time is None:
            return None
        now = self.get_clock().now()
        return (now.nanoseconds - arrival_time.nanoseconds) / 1e9

    def _report_status(self) -> None:
        """
        Emit a periodic summary of sensor freshness.
        """
        if self._terminal_latched:
            return

        now = self.get_clock().now()
        if now - self._last_report_time < self._report_period:
            return

        self._last_report_time = now
        missing = []

        if self._latest_color is None:
            missing.append('color')
        if self._latest_depth is None:
            missing.append('depth')
        if self._audio_enabled and self._latest_audio is None:
            missing.append('audio')

        if missing:
            self.get_logger().warn('Waiting for sensor inputs: ' + ', '.join(missing))
            return

        if self._audio_enabled and self._latest_audio is not None:
            audio_value = float(self._latest_audio.data)
            danger_state = 'DANGER' if audio_value >= self._danger_audio_threshold else 'safe'
            self.get_logger().info(
                'Snapshot '
                f'color={self._latest_color.width}x{self._latest_color.height} '
                f'depth={self._latest_depth.width}x{self._latest_depth.height} '
                f'audio={audio_value:.3f} '
                f'state={danger_state} '
                f'ages(s)='
                f'[{self._age_seconds(self._color_arrival_time):.2f}, '
                f'{self._age_seconds(self._depth_arrival_time):.2f}, '
                f'{self._age_seconds(self._audio_arrival_time):.2f}]'
            )
            return

        self.get_logger().info(
            'Snapshot '
            f'color={self._latest_color.width}x{self._latest_color.height} '
            f'depth={self._latest_depth.width}x{self._latest_depth.height} '
            f'ages(s)='
            f'[{self._age_seconds(self._color_arrival_time):.2f}, '
            f'{self._age_seconds(self._depth_arrival_time):.2f}]'
        )


def main(args=None) -> None:
    """
    ROS entrypoint that spins the diagnostics node.
    """
    rclpy.init(args=args)
    node = FearSensorSubscriber()

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


if __name__ == '__main__':
    main()

