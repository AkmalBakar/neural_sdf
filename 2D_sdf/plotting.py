import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

def plot_comparison(net, true_sdf_func, domain_bounds, grid_pts_x, grid_pts_y, n_samples_x, n_samples_y):
    xs = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], n_samples_x)
    ys = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], n_samples_y)
    plot_xs, plot_ys = jnp.meshgrid(xs, ys)
    plot_pts = jnp.stack([plot_xs, plot_ys], axis=-1)
    final_pred = vmap(vmap(net))(plot_pts).squeeze()
    final_true = vmap(vmap(true_sdf_func))(plot_pts).squeeze()

    # plt.figure(figsize=(20, 10))
    # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # fig.suptitle('Comparison')

    data = [final_pred, final_true]
    label = ["Final Prediction", "Ground Truth"]

    # for i in range(2):
    #     ax = axs[i]
    #     im = ax.contour(plot_xs, plot_ys, data[i], levels=[-0.5,0.0,0.5])
    #     ax.set_title(f'{label[i]}')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel("Y")
    #     fig.colorbar(im, ax=ax)

    #     # Add vertical lines
    #     for x in grid_pts_x:
    #         ax.axvline(x=x, color="white", linestyle="--", linewidth=0.5)

    #     # Add horizontal lines
    #     for y in grid_pts_y:
    #         ax.axhline(y=y, color="white", linestyle="--", linewidth=0.5)

    plt.figure()
    plt.title('Comparison')
    ct1 = plt.contour(plot_xs, plot_ys, data[0], label=label[0], levels=[-0.5,0.0,0.5])
    ct2 = plt.contour(plot_xs, plot_ys, data[1], label=label[1], levels=[-0.5,0.0,0.5], linestyle="--")
    plt.colorbar(ct1)
    plt.legend()
    
    plt.tight_layout()
    # plt.show()

def plot_pos_encoding(net, domain_bounds, grid_pts_x, grid_pts_y, n_samples_x, n_samples_y):
    xs = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], n_samples_x)
    ys = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], n_samples_y)
    plot_xs, plot_ys = jnp.meshgrid(xs, ys)
    plot_pts = jnp.stack([plot_xs, plot_ys], axis=-1)
    feat_interp = vmap(vmap(net.get_pos_encoding))(plot_pts)

    plt.figure(figsize=(12, 18))
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    fig.suptitle('Pos Encoding plots')

    for i in range(3):
        for j in range(2):
            ax = axs[i, j]
            im = ax.pcolormesh(plot_xs, plot_ys, feat_interp[:, :, i, j], shading='auto')
            ax.set_title(f'Subplot {i+1},{j+1}')
            ax.set_xlabel('X')
            ax.set_ylabel("Y")
            fig.colorbar(im, ax=ax)

            # Add vertical lines
            for x in grid_pts_x:
                ax.axvline(x=x, color="white", linestyle="--", linewidth=0.5)

            # Add horizontal lines
            for y in grid_pts_y:
                ax.axhline(y=y, color="white", linestyle="--", linewidth=0.5)
    
    plt.tight_layout()
    # plt.show()

def plot_feature_grid(net, domain_bounds, grid_pts_x, grid_pts_y, n_samples_x, n_samples_y):
    xs = jnp.linspace(domain_bounds[0,0], domain_bounds[1,0], n_samples_x)
    ys = jnp.linspace(domain_bounds[0,1], domain_bounds[1,1], n_samples_y)
    plot_xs, plot_ys = jnp.meshgrid(xs, ys)
    plot_pts = jnp.stack([plot_xs, plot_ys], axis=-1)
    feat_interp = vmap(vmap(net.get_feature))(plot_pts)

    plt.figure(figsize=(20, 10))
    fig, axs = plt.subplots(1, 5, figsize=(20, 10))
    fig.suptitle('Feature Grid Plot')

    for i in range(5):
        ax = axs[i]
        im = ax.pcolormesh(plot_xs, plot_ys, feat_interp[:, :, i], shading='auto')
        ax.set_title(f'Subplot {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax)

        # Add vertical lines
        for x in grid_pts_x:
            ax.axvline(x=x, color="white", linestyle="--", linewidth=0.5)

        # Add horizontal lines
        for y in grid_pts_y:
            ax.axhline(y=y, color="white", linestyle="--", linewidth=0.5)
    
    plt.tight_layout()
    # plt.show()