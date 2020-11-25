import matplotlib.pyplot as plt
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation


def plot_t_convergence(
    target_translation: np.ndarray, translation_log: List[np.ndarray]
):
    fig, ax = plt.subplots()
    plt.title("Convergence of t")

    t_offset = np.stack(translation_log) - target_translation

    for axis in range(3):
        ax.plot(t_offset[:, axis])

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Distance to Ground Truth")
    ax.legend(["X", "Y", "Z"])

    plt.savefig("tmp_t_plot.png")
    plt.close(fig)


def plot_r_convergence(target_rotation: np.ndarray, rotation_log: List[np.ndarray]):
    fig, ax = plt.subplots()
    plt.title("Convergence of R")
    euler_rot_log = []
    for r_mat in rotation_log:
        euler_rot_log.append(Rotation.from_matrix(r_mat).as_euler("xyz"))
    euler_rot_log = np.ndarray(euler_rot_log)

    euler_target_location = Rotation.from_matrix(target_rotation).as_euler("xyz")

    r_offset = np.stack(translation_log) - target_translation

    for axis in range(3):
        ax.plot(r_offset[:, axis])

    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Angular Difference to Ground Truth")
    ax.legend(["X_rot", "Y_rot", "Z_rot"])

    plt.savefig("tmp_r_plot.png")
    plt.close(fig)
