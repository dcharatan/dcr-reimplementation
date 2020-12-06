import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np
from scipy.spatial.transform import Rotation
import math


def plot_t_convergence(
    target_translation: np.ndarray,
    translation_log: List[np.ndarray],
    s_log: Optional[List[float]] = None,
    x_line_position: Optional[float] = None,
    file_name: Optional[str] = None,
):
    fig, ax = plt.subplots()
    plt.title("Convergence of t")
    plt.grid(True)

    # Compute the offset of each of the translations in the log w.r.t the target
    # t_offset = np.stack(translation_log) - target_translation
    t_offset = np.array(translation_log)

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
        plt.axvline(x=x_line_position)

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
    # euler_target = Rotation.from_matrix(target_rotation).as_euler("xyz")
    # r_offset = (np.stack(euler_log) - euler_target) * 180 / 3.141
    r_offset = euler_log * 180 / 3.141

    theta_log = [
        (math.acos((np.trace(R_a) - 1) / 2) * 180 / 3.141) for R_a in rotation_log
    ]
    neg_theta_log = [-theta for theta in theta_log]

    # Plot the rotation curves
    ax.plot(r_offset[:, 0], "r")
    ax.plot(r_offset[:, 1], "g")
    ax.plot(r_offset[:, 2], "b")
    ax.plot(neg_theta_log, "k")
    ax.plot(theta_log, "k")

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Magnitude of Rotation (degrees)")
    ax.legend(["X_rot", "Y_rot", "Z_rot", "\u03B8"])

    if x_line_position:
        plt.axvline(x=x_line_position)

    plt.savefig("tmp_r_plot.png" if file_name is None else file_name)
    plt.close(fig)
