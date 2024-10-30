import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import gaussian_kde

def generate_model_sdf_slice(model, domain_bounds, slice_axis=2, location=0, n_samples=100):
    # Check if the location is within domain bounds
    if location < domain_bounds[0, slice_axis] or location > domain_bounds[1, slice_axis]:
        raise ValueError(f"Location {location} is outside the domain bounds for axis {slice_axis}")

    # Create meshgrid for the other two axes
    axes = [0, 1, 2]
    axes.remove(slice_axis)
    x_axis, y_axis = axes

    xs = jnp.linspace(domain_bounds[0, x_axis], domain_bounds[1, x_axis], n_samples)
    ys = jnp.linspace(domain_bounds[0, y_axis], domain_bounds[1, y_axis], n_samples)
    plot_xs, plot_ys = jnp.meshgrid(xs, ys)

    # Create the points array
    plot_pts = jnp.zeros((n_samples, n_samples, 3))
    plot_pts = plot_pts.at[:, :, x_axis].set(plot_xs)
    plot_pts = plot_pts.at[:, :, y_axis].set(plot_ys)
    plot_pts = plot_pts.at[:, :, slice_axis].set(location)

    # Compute SDF values
    sdf_values = vmap(vmap(model))(plot_pts).squeeze()

    return plot_pts, sdf_values

def plot_sdf_slice_contour_comparison(plot_xs, plot_ys, pred_sdf, true_sdf, axis_names, figure_size=(10,10)):
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)

    n_levels = 10
    levels = np.linspace(true_sdf.min(), true_sdf.max(), n_levels)
    
    # Plot true SDF
    contour_true = ax.contour(plot_xs, plot_ys, true_sdf, levels=levels, cmap=mpl.colormaps['plasma'], alpha=0.5)
    contour_true_zero = ax.contour(plot_xs, plot_ys, true_sdf, levels=[0.0], colors='black')
    ax.clabel(contour_true, inline=True, fontsize=8)

    # Plot predicted SDF
    contour_pred = ax.contour(plot_xs, plot_ys, pred_sdf, levels=levels, cmap=mpl.colormaps['plasma'], alpha=0.5, linestyles="--")
    contour_pred_zero = ax.contour(plot_xs, plot_ys, pred_sdf, levels=[0.0], colors='black', linestyles="--")
    ax.clabel(contour_pred, inline=True, fontsize=8)

    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_title("SDF Slice - Predicted (Dashed) vs True (Solid)")
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    # Adjust the plot to fill the figure while maintaining the aspect ratio
    plt.tight_layout()

    return fig

def plot_comparison_sdf_slice(model, mesh_sdf_func, domain_bounds, slice_axis=2, location=0, n_pts=100):
    plot_pts, pred_sdf = generate_model_sdf_slice(
        model, domain_bounds, slice_axis, location, n_pts
    )
    true_sdf = mesh_sdf_func(np.asarray(plot_pts.reshape(-1,3))).reshape(pred_sdf.shape)

    axis_names = ["x", "y", "z"]
    axis_names.pop(slice_axis)
    axes = [0, 1, 2]
    axes.remove(slice_axis)
    axis_a, axis_b = axes

    # Domain bounds of slice
    slice_bounds = domain_bounds[:,axes]
    slice_sizes = np.abs(slice_bounds[1] - slice_bounds[0])

    # Figure size based on the slice sizes
    figure_size = 15 * slice_sizes / slice_sizes.max()
    figure_size = tuple(np.squeeze(figure_size))

    fig = plot_sdf_slice_contour_comparison(
        plot_pts[:, :, axis_a],
        plot_pts[:, :, axis_b],
        pred_sdf,
        true_sdf,
        axis_names,
        figure_size
    )
    return fig

def plot_sampling_density_heatmap(points, domain_bounds, slice_axis=2, slice_location=None, n_pts=100):
    """
    Generate a density heatmap plot of points projected onto a slice of the geometry.
    
    Parameters:
    - points: numpy array of shape (n_samples, 3) containing the 3D coordinates of sampled points
    - domain_bounds: Bounds of the domain, shape (2, 3)
    - slice_axis: Axis perpendicular to the slice (0 for YZ, 1 for XZ, 2 for XY)
    - slice_location: Location of the slice along the slice_axis. If None, use the center of the domain.
    - n_pts: Number of points for the plot grid
    
    Returns:
    - fig: matplotlib figure object
    """
    # Determine the slice location if not provided
    if slice_location is None:
        slice_location = (domain_bounds[0, slice_axis] + domain_bounds[1, slice_axis]) / 2
    
    # Determine the plotting axes
    plot_axes = [0, 1, 2]
    plot_axes.remove(slice_axis)
    x_axis, y_axis = plot_axes
    
    # Create a grid for the density estimation
    x = np.linspace(domain_bounds[0, x_axis], domain_bounds[1, x_axis], n_pts)
    y = np.linspace(domain_bounds[0, y_axis], domain_bounds[1, y_axis], n_pts)
    xx, yy = np.meshgrid(x, y)
    
    # Perform kernel density estimation using all points
    kde = gaussian_kde(points.T)
    
    # Prepare the grid points for evaluation
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    slice_coord = np.full(grid_points.shape[1], slice_location)
    eval_points = np.vstack([grid_points[0], grid_points[1], slice_coord])
    eval_points[[slice_axis, 2]] = eval_points[[2, slice_axis]]  # Swap slice_axis and Z if necessary
    
    # Evaluate KDE on the grid
    z = kde(eval_points)
    z = z.reshape(xx.shape)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()], 
                   origin='lower', cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Sampling Density')
    
    # Set labels and title
    ax.set_xlabel(['X', 'Y', 'Z'][x_axis])
    ax.set_ylabel(['X', 'Y', 'Z'][y_axis])
    slice_axis_name = ['X', 'Y', 'Z'][slice_axis]
    ax.set_title(f'Sampling Density Heatmap (Slice at {slice_axis_name}={slice_location:.2f})')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig