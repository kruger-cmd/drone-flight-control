"""Position control and waypoint navigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Waypoint:
    """Single waypoint."""

    position: np.ndarray
    velocity: float = 1.0
    hold_time: float = 0.0


class PositionController:
    """PID position controller."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.5) -> None:
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
        desired_position: np.ndarray,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute desired acceleration and attitude setpoint."""
        error = desired_position - current_position
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 1e-6 else np.zeros(3)
        self.prev_error = error.copy()

        # PID acceleration
        desired_acc = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Convert acceleration to attitude setpoint
        # For small angles, roll/pitch = accel_x/g, accel_y/g
        g = 9.81
        desired_attitude = np.array([
            -desired_acc[1] / g,  # roll
            desired_acc[0] / g,  # pitch
            0.0,  # yaw
        ])
        desired_thrust = max(0.0, g + desired_acc[2])

        return desired_attitude, desired_thrust
