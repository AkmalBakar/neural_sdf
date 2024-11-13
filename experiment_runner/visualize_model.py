import sys
import os
from os import path
import glob
import jax
import matplotlib.pyplot as plt

# Add root directory to sys path
sys.path.append(path.realpath(path.dirname(path.dirname(__file__))))

from neural_sdf import models, plot, core

if __name__ == "__main__":
    # Find all resolution_5_5_5 models
    root_output_dir = path.realpath(
        path.join(path.dirname(__file__), "..", "experiment_outputs")
    )
    model_paths = glob.glob(path.join(root_output_dir, "*resolution_5_5_5*", "resolution_5_5_5_*_epoch_2000.eqx"))
    
    if not model_paths:
        raise ValueError("No models found with resolution 5x5x5!")

    # Get the config for the model by modifying model path
    model_path = model_paths[0]
    config_path = model_path.split("_epoch")[0] + "_config.yaml"

    # Load the config and initialize mesh sampler
    config = core.load_config(config_path)
    _, _, sampler = core.create_data_loaders(config["data"], data_dir=path.join(path.dirname(__file__), "..", "data"))
    
    # Load the first model found
    model_path = model_paths[0]
    print(f"Loading model from {model_path}")
    model = models.GridNet3D.load(model_path)
    
    # Create output directory for plots
    plot_dir = path.join(path.dirname(model_path), "visualization_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Fixed parameters for visualization
    slice_axis = 2  # XY plane
    location = 0.5  # Halfway through the domain
    n_pts = 300
    
    # Plot positional encodings
    fig = plot.plot_positional_encodings(
        model, 
        model.domain_bounds, 
        slice_axis=slice_axis, 
        location=location,
        n_samples=n_pts
    )
    
    # Plot feature vectors
    fig = plot.plot_feature_vectors(
        model, 
        model.domain_bounds, 
        slice_axis=slice_axis, 
        location=location,
        n_samples=n_pts
    )

    # Plot comparison of model SDF and true mesh SDF
    fig = plot.plot_comparison_sdf_slice(
        model, 
        sampler.sdf,
        model.domain_bounds, 
        slice_axis=slice_axis, 
        location=location,
        n_pts=n_pts
    )

    plt.show(block=True)
    print("Done!")