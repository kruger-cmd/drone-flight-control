"""Waypoint navigation, altitude hold, and obstacle avoidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Waypoint:
    """Single waypoint with optional hold time."""

    position: np.ndarray
    velocity: float = 1.0
    hold_time: float = 0.0


class WaypointNavigator:
    """Sequential waypoint navigation with arrival detection."""

    def __init__(self, waypoints: Sequence[Waypoint] | None = None, arrival_threshold: float = 0.5) -> None:
        self.waypoints = list(waypoints or [])
        self.current_index = 0
        self.arrival_threshold = arrival_threshold
        self.hold_timer = 0.0
        self.reached = False

    def reset(self) -> None:
        """Reset to first waypoint."""
        self.current_index = 0
        self.hold_timer = 0.0
        self.reached = False

    def update(self, current_position: np.ndarray, dt: float) -> np.ndarray | None:
        """Return current target position, or None if finished."""
        if self.current_index >= len(self.waypoints):
            return None

        target = self.waypoints[self.current_index].position
        error = np.linalg.norm(current_position - target)

        if error < self.arrival_threshold:
            self.hold_timer += dt
            if self.hold_timer >= self.waypoints[self.current_index].hold_time:
                self.current_index += 1
                self.hold_timer = 0.0
                if self.current_index >= len(self.waypoints):
                    self.reached = True
                    return None

        return target


class AltitudeHold:
    """Altitude hold with simple PID."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.5) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, desired_altitude: float, current_altitude: float, current_vel_z: float, dt: float) -> float:
        """Return desired vertical acceleration."""
        error = desired_altitude - current_altitude
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 1e-6 else 0.0
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative


class ObstacleAvoidance:
    """Simple repulsive potential field obstacle avoidance."""

    def __init__(self, safe_distance: float = 2.0, strength: float = 5.0) -> None:
        self.safe_distance = safe_distance
        self.strength = strength

    def compute(self, current_position: np.ndarray, obstacles: Sequence[np.ndarray]) -> np.ndarray:
        """Return avoidance acceleration vector."""
        avoidance = np.zeros(3)
        for obs in obstacles:
            diff = current_position - obs
            dist = np.linalg.norm(diff)
            if dist < self.safe_distance and dist > 1e-6:
                # Repulsive force inversely proportional to distance
                force = self.strength * (1.0 / dist - 1.0 / self.safe_distance) / (dist ** 2)
                avoidance += force * diff / dist

        return avoidance
