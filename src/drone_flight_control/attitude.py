"""Attitude control and quadcopter dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuadcopterDynamics:
    """Minimal quadcopter rigid-body dynamics."""

    mass: float = 1.0
    inertia: np.ndarray = None  # type: ignore
    gravity: float = 9.81
    drag_coeff: float = 0.1

    def __post_init__(self) -> None:
        if self.inertia is None:
            self.inertia = np.diag([0.01, 0.01, 0.02])

    def equations_of_motion(
        self,
        state: np.ndarray,
        thrust: float,
        torques: np.ndarray,
    ) -> np.ndarray:
        """Compute state derivative."""
        # State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        if state.shape[0] != 13:
            raise ValueError(f"State must be 13 elements, got {state.shape[0]}")

        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]

        # Normalize quaternion
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            quat = quat / quat_norm

        # Rotation matrix from quaternion
        R = self._quaternion_to_rotation(quat)

        # Linear acceleration
        thrust_vec = thrust * R[:, 2]
        drag = -self.drag_coeff * vel
        acc = (thrust_vec + np.array([0, 0, -self.mass * self.gravity])) / self.mass + drag

        # Quaternion derivative
        q_dot = 0.5 * self._quaternion_multiply(quat, np.array([0, *omega]))

        # Angular acceleration
        inertia_inv = np.linalg.inv(self.inertia)
        angular_acc = inertia_inv @ (torques - np.cross(omega, self.inertia @ omega))

        return np.concatenate([vel, acc, q_dot, angular_acc])

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)],
        ])


class AttitudeController:
    """Simple PID attitude controller."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.1) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def reset(self) -> None:
        """Reset controller state."""
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute(
        self,
        desired_attitude: np.ndarray,
        current_attitude: np.ndarray,
        current_omega: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute torque command."""
        error = desired_attitude - current_attitude
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 1e-6 else np.zeros(3)
        self.prev_error = error.copy()

        torque = self.kp * error + self.ki * self.integral + self.kd * derivative
        return torque
