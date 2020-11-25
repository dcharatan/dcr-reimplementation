import matplotlib.pyplot as plt
from typing import List
import numpy as np


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
