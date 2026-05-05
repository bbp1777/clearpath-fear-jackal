"""
Apply runtime timeout overrides to the generated Clearpath control nodes.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient


def _qualify_name(namespace: str, node_name: str) -> str:
    if node_name.startswith('/'):
        return node_name
    namespace = namespace.strip('/')
    if not namespace:
        return f'/{node_name}'
    return f'/{namespace}/{node_name}'


class CommandTimeoutOverrideNode(Node):
    """
    Wait for the generated Clearpath nodes, then raise the stamped-command timeout.
    """

    def __init__(self) -> None:
        super().__init__('command_timeout_overrides')

        self.declare_parameter('namespace', 'jackal_sidewalk')
        self.declare_parameter('platform_velocity_controller_node', 'platform_velocity_controller')
        self.declare_parameter('twist_mux_node', 'twist_mux')
        self.declare_parameter('platform_cmd_vel_timeout_s', 2.0)
        self.declare_parameter('twist_mux_external_timeout_s', 2.0)
        self.declare_parameter('retry_period_s', 1.0)
        self.declare_parameter('retry_log_period', 10)

        namespace = str(self.get_parameter('namespace').value)
        controller_node = str(self.get_parameter('platform_velocity_controller_node').value)
        twist_mux_node = str(self.get_parameter('twist_mux_node').value)
        self._platform_timeout = float(self.get_parameter('platform_cmd_vel_timeout_s').value)
        self._external_timeout = float(self.get_parameter('twist_mux_external_timeout_s').value)
        retry_period_s = max(float(self.get_parameter('retry_period_s').value), 0.1)
        self._retry_log_period = max(int(self.get_parameter('retry_log_period').value), 1)

        self._controller_target = _qualify_name(namespace, controller_node)
        self._twist_mux_target = _qualify_name(namespace, twist_mux_node)
        self._controller_client = AsyncParameterClient(self, self._controller_target)
        self._twist_mux_client = AsyncParameterClient(self, self._twist_mux_target)

        self._attempt_count = 0
        self._controller_request_pending = False
        self._twist_mux_request_pending = False
        self._controller_applied = False
        self._twist_mux_applied = False
        self._shutdown_requested = False

        self._timer = self.create_timer(retry_period_s, self._apply_overrides)
        self.get_logger().info(
            'Waiting to override command timeouts '
            f'controller={self._controller_target} cmd_vel_timeout={self._platform_timeout:.2f}s '
            f'twist_mux={self._twist_mux_target} external_timeout={self._external_timeout:.2f}s'
        )

    def _apply_overrides(self) -> None:
        if self._controller_applied and self._twist_mux_applied:
            if not self._shutdown_requested:
                self.get_logger().info('Command timeout overrides are active; shutting down helper node.')
                self._shutdown_requested = True
            self._timer.cancel()
            return

        self._attempt_count += 1
        if not self._controller_applied and not self._controller_request_pending:
            self._try_set_controller_timeout()
        if not self._twist_mux_applied and not self._twist_mux_request_pending:
            self._try_set_twist_mux_timeout()

        if self._attempt_count % self._retry_log_period == 0:
            waiting_for = []
            if not self._controller_applied:
                waiting_for.append(self._controller_target)
            if not self._twist_mux_applied:
                waiting_for.append(self._twist_mux_target)
            self.get_logger().info(
                f'Command timeout override still waiting for nodes: {", ".join(waiting_for)}'
            )

    def _try_set_controller_timeout(self) -> None:
        if not self._controller_client.wait_for_services(timeout_sec=0.0):
            return
        future = self._controller_client.set_parameters(
            [Parameter('cmd_vel_timeout', value=self._platform_timeout)]
        )
        self._controller_request_pending = True
        future.add_done_callback(self._on_controller_result)

    def _try_set_twist_mux_timeout(self) -> None:
        if not self._twist_mux_client.wait_for_services(timeout_sec=0.0):
            return
        future = self._twist_mux_client.set_parameters(
            [Parameter('topics.external.timeout', value=self._external_timeout)]
        )
        self._twist_mux_request_pending = True
        future.add_done_callback(self._on_twist_mux_result)

    def _on_controller_result(self, future) -> None:
        self._controller_request_pending = False
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - runtime ROS failure path
            self.get_logger().warning(f'Unable to set controller cmd_vel_timeout yet: {exc}')
            return

        results = self._parameter_results(response)
        if results and all(item.successful for item in results):
            self._controller_applied = True
            self.get_logger().info(
                f'Applied {self._platform_timeout:.2f}s cmd_vel_timeout to {self._controller_target}.'
            )
            return

        reasons = ', '.join(item.reason for item in results if item.reason) if results else 'unknown failure'
        self.get_logger().warning(
            f'Controller timeout override was rejected by {self._controller_target}: {reasons}'
        )

    def _on_twist_mux_result(self, future) -> None:
        self._twist_mux_request_pending = False
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - runtime ROS failure path
            self.get_logger().warning(f'Unable to set twist_mux external timeout yet: {exc}')
            return

        results = self._parameter_results(response)
        if results and all(item.successful for item in results):
            self._twist_mux_applied = True
            self.get_logger().info(
                f'Applied {self._external_timeout:.2f}s topics.external.timeout to {self._twist_mux_target}.'
            )
            return

        reasons = ', '.join(item.reason for item in results if item.reason) if results else 'unknown failure'
        self.get_logger().warning(
            f'Twist mux timeout override was rejected by {self._twist_mux_target}: {reasons}'
        )

    def _parameter_results(self, response) -> list:
        if response is None:
            return []
        results = getattr(response, 'results', None)
        if results is not None:
            return list(results)
        if isinstance(response, (list, tuple)):
            return list(response)
        return []


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CommandTimeoutOverrideNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
