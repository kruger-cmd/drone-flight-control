"""Tests for drone flight control."""

import numpy as np

from drone_flight_control.attitude import AttitudeController, QuadcopterDynamics
from drone_flight_control.position import PositionController, Waypoint


def test_quadcopter_dynamics():
    dynamics = QuadcopterDynamics(mass=1.0)
    state = np.zeros(13)
    state[6] = 1.0  # quaternion w=1
    thrust = 9.81  # hover
    torques = np.zeros(3)
    derivative = dynamics.equations_of_motion(state, thrust, torques)
    assert derivative.shape == (13,)
    # Linear acceleration should be approximately zero for hover
    assert np.linalg.norm(derivative[3:6]) < 1.0


def test_attitude_controller():
    ctrl = AttitudeController(kp=1.0, ki=0.0, kd=0.1)
    desired = np.array([0.0, 0.0, 0.0])
    current = np.array([0.1, -0.1, 0.05])
    omega = np.zeros(3)
    torque = ctrl.compute(desired, current, omega, dt=0.01)
    assert torque.shape == (3,)
    assert np.linalg.norm(torque) > 0


def test_position_controller():
    ctrl = PositionController(kp=1.0, ki=0.0, kd=0.5)
    desired_pos = np.array([1.0, 2.0, 3.0])
    current_pos = np.array([0.0, 0.0, 0.0])
    current_vel = np.array([0.0, 0.0, 0.0])
    attitude, thrust = ctrl.compute(desired_pos, current_pos, current_vel, dt=0.01)
    assert attitude.shape == (3,)
    assert thrust >= 0.0
