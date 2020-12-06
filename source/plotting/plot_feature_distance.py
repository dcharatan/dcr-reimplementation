import numpy as np
import matplotlib.pyplot as plt
from ..utilities import compute_feature_distance


def plot_feature_distance(
    image_reference: np.ndarray,
    image_current: np.ndarray,
    file_name: str,
):
    # Calculate AFD.
    afd, points_reference, points_current = compute_feature_distance(
        image_reference, image_current
    )

    # Plot feature distance vectors.
    fig, ax = plt.subplots()
    afd_string = "{:.3f}".format(afd)
    plt.title(f"Feature Distance (Average: {afd_string})")
    plt.quiver(
        points_reference[:, 0],
        points_reference[:, 1],
        points_current[:, 0] - points_reference[:, 0],
        points_current[:, 1] - points_reference[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.xlim(0, image_reference.shape[1])
    plt.ylim(0, image_reference.shape[0])
    plt.savefig(file_name)
    plt.close(fig)