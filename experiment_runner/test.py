import sys
import os
from os import path
import glob
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
# Add root directory to sys path
sys.path.append(path.realpath(path.dirname(path.dirname(__file__))))

from neural_sdf import core, models, datalog

if __name__ == "__main__":
    # Name of this specific model
    model_name = "resolution_5_5_5_feature_size_15_width_size_40"

    # Model folder
    model_dir = path.join(path.dirname(__file__), "..", "experiment_outputs", model_name)

    # Load existing model
    config_path = path.join(model_dir, f"{model_name}_config.yaml")
    config = core.load_config(config_path, config_path)

    # Folder where the meshes can be found
    data_dir = path.join(path.dirname(__file__), "..", "data")

    # Load the latest model
    model_files = glob.glob(path.join(model_dir, f"{model_name}_epoch_*.eqx"))
    model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    model = models.GridNet3D.load(model_files[-1])

    # Get points to evaluate positional encodings at
    domain_bounds = jnp.array(config["model"]["domain_bounds"])
    n_pts = 300
    x = jnp.linspace(domain_bounds[0][0], domain_bounds[1][0], n_pts)
    y = jnp.linspace(domain_bounds[0][1], domain_bounds[1][1], n_pts)
    z = jnp.linspace(domain_bounds[0][2], domain_bounds[1][2], n_pts)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    points = jnp.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
    pos_enc = jax.vmap(model.get_pos_encoding, in_axes=1)(points)
    # Reshape to have shape (n_pts, n_pts, n_pts, 2, 3)
    pos_enc = pos_enc.reshape(n_pts, n_pts, n_pts, 2, 3)

    # Plot positional encodings
    plt.figure()
    # Figure with 3 rows and 2 columns
    # Each column shows the positional encodings for a different axis
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(pos_enc[:, :, :, 0, i])
        plt.colorbar()
        plt.title(f"Positional encodings for X")
        plt.subplot(3, 2, 2*i+2)
        plt.imshow(pos_enc[:, :, :, 1, i])
        plt.colorbar()
        plt.title(f"Positional encodings for Y")
    plt.show()

    print("Done!")