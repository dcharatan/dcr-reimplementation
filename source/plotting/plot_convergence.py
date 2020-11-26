import matplotlib.pyplot as plt
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation


def plot_t_convergence(
    target_translation: np.ndarray, translation_log: List[np.ndarray]
):
    fig, ax = plt.subplots()
    plt.title("Convergence of t")
    plt.grid(True)

    t_offset = np.stack(translation_log) - target_translation

    for axis in range(3):
        ax.plot(t_offset[:, axis])

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Distance to Ground Truth")
    ax.legend(["X", "Y", "Z"])

    plt.savefig("tmp_t_plot.png")
    plt.close(fig)


def plot_r_convergence(target_rotation: np.ndarray, rotation_log: List[np.ndarray]):
    to_euler = lambda x: Rotation.from_matrix(x).as_euler("xyz")

    fig, ax = plt.subplots()
    plt.title("Convergence of R")
    plt.grid(True)
    euler_log = []
    for rotation in rotation_log:
        euler_log.append(to_euler(rotation))
    euler_log = np.stack(euler_log)

    euler_target = Rotation.from_matrix(target_rotation).as_euler("xyz")

    r_offset = (np.stack(euler_log) - euler_target) * 180 / 3.141
    for axis in range(3):
        ax.plot(r_offset[:, axis])

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Angular Difference to Ground Truth (degrees)")
    ax.legend(["X_rot", "Y_rot", "Z_rot"])

    plt.savefig("tmp_r_plot.png")
    plt.close(fig)
