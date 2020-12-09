import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np
from scipy.spatial.transform import Rotation
import math
from ..utilities import is_translation_vector, is_rotation_matrix


def plot_t_distance_to_ground_truth(
    target_translation: np.ndarray,
    translation_log: List[np.ndarray],
    x_line_position: Optional[float] = None,
    file_name: str = None,
):
    """Create a plot of the distance between a given t value and the ground truth."""

    t_delta = np.array(translation_log) - target_translation
    fig, ax = plt.subplots()
    plt.title("Distance to Ground Truth (t)")
    plt.grid(True)
    ax.plot(t_delta[:, 0], "r")
    ax.plot(t_delta[:, 1], "g")
    ax.plot(t_delta[:, 2], "b")
    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Distance to Ground Truth Translation")
    ax.legend(["X", "Y", "Z"])

    # Add an x line.
    if x_line_position:
        plt.axvline(x=x_line_position, color="k")

    # Save and close the plot.
    plt.savefig(file_name)
    plt.close(fig)


def plot_R_distance_to_ground_truth(
    target_rotation: np.ndarray,
    rotation_log: List[np.ndarray],
    x_line_position: Optional[float] = None,
    file_name: str = None,
):
    """Create a plot of the distance between a given R value and the ground truth."""
    assert is_rotation_matrix(target_rotation)

    to_euler = lambda x: Rotation.from_matrix(x).as_euler("xyz", True)

    fig, ax = plt.subplots()
    plt.title("Distance to Ground Truth (R)")
    plt.grid(True)
    euler_target = to_euler(target_rotation)
    euler_log = np.stack([to_euler(R) - euler_target for R in rotation_log])

    ax.plot(euler_log[:, 0], "r")
    ax.plot(euler_log[:, 1], "g")
    ax.plot(euler_log[:, 2], "b")

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Difference to Ground Truth Rotation (Degrees)")
    ax.legend(["X", "Y", "Z"])

    if x_line_position:
        plt.axvline(x=x_line_position, color="k")

    plt.savefig(file_name)
    plt.close(fig)


def plot_t_convergence(
    target_rotation: np.ndarray,
    translation_log: List[np.ndarray],
    s_log: Optional[List[float]] = None,
    x_line_position: Optional[float] = None,
    file_name: Optional[str] = None,
):
    fig, ax = plt.subplots()
    plt.title("Convergence of t")
    plt.grid(True)

    # Compute the offset of each of the translations in the log w.r.t the target
    t_offset = np.array(translation_log)
    t_offset = t_offset[1:] - t_offset[:-1]

    ax.plot(t_offset[:, 0], "r")
    ax.plot(t_offset[:, 1], "g")
    ax.plot(t_offset[:, 2], "b")
    if s_log is not None:
        ax.plot(s_log, "k")
        ax.plot([-s for s in s_log], "k")

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Magnitude of Translation")
    ax.legend(["X", "Y", "Z", "Step size s"])

    if x_line_position:
        plt.axvline(x=x_line_position, color="k")

    plt.savefig("tmp_t_plot.png" if file_name is None else file_name)
    plt.close(fig)


def plot_r_convergence(
    target_rotation: np.ndarray,
    rotation_log: List[np.ndarray],
    x_line_position: Optional[float] = None,
    file_name: Optional[str] = None,
):
    to_euler = lambda x: Rotation.from_matrix(x).as_euler("xyz")

    fig, ax = plt.subplots()
    plt.title("Convergence of R")
    plt.grid(True)
    euler_log = []

    for rotation in rotation_log:
        euler_log.append(to_euler(rotation))

    euler_log = np.stack(euler_log)
    r_offset = euler_log[1:] - euler_log[:-1]
    r_offset_matrix = [Rotation.from_euler("xyz", r).as_matrix() for r in r_offset]

    theta_log = [
        (math.acos((np.trace(R) - 1) / 2) * 180 / 3.141) for R in r_offset_matrix
    ]
    r_offset *= 180 / 3.141
    neg_theta_log = [-theta for theta in theta_log]

    # Plot the rotation curves
    ax.plot(r_offset[:, 0], "r")
    ax.plot(r_offset[:, 1], "g")
    ax.plot(r_offset[:, 2], "b")
    ax.plot(neg_theta_log, "k")
    ax.plot(theta_log, "k")

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Magnitude of Rotation (degrees)")
    ax.legend(["X", "Y", "Z", "\u03B8"])

    if x_line_position:
        plt.axvline(x=x_line_position, color="k")

    plt.savefig("tmp_r_plot.png" if file_name is None else file_name)
    plt.close(fig)
