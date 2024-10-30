from os import path
from typing import *

import equinox as eqx
import igl
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax import vmap

from .datalog import *
from .geometry import *
from .loss import *
from .models import GridNet3D
from .plot import *
from .train import *
from .utils import *

# jax.config.update('jax_log_compiles', True)
jax.config.update("jax_compilation_cache_dir", f"{path.dirname(__file__)}/__pycache__/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_debug_nans", True)

# TODO: Check if this is the right way to seed
# Or should user provide key when calling any of the functions below?
# Seed random key
key = jax.random.PRNGKey(0)

def create_data_loaders(data_config: Dict, data_dir: str = None) -> Tuple[SDFDataloader, SDFDataloader, MeshSampler]:
    """Create train and test data loaders from config.
    
    Returns:
        Tuple of (train_loader, test_loader, sampler)
    """
    # Load and scale mesh
    mesh_path = data_config["mesh_file"]
    if not path.isfile(mesh_path) and data_dir is not None:
        mesh_path = path.join(data_dir, mesh_path)
    if not path.isfile(mesh_path):
        raise ValueError(
            f"Could not open mesh file at {data_config['mesh_file']} or {mesh_path}"
        )

    v, f = igl.read_triangle_mesh(mesh_path)
    v = float(data_config["scale_factor"]) * v
    b_min = np.min(v, axis=0)
    b_max = np.max(v, axis=0)
    b_lens = b_max - b_min

    if data_config["domain_bounds"] is None:
        domain_bounds = np.array([b_min - 0.1 * b_lens, b_max + 0.1 * b_lens])
    else:
        domain_bounds = np.array(data_config["domain_bounds"])

    # Initialize mesh sampler
    sampler = MeshSampler(
        v,
        f,
        domain_bounds,
        ratio=data_config["sampler_ratio"],
        std=data_config["sampler_std"] * b_lens.min(),
    )

    train_loader = None
    if data_config["train_samples"] > 0:
        train_loader = SDFDataloader(
            sampler,
            num_samples=data_config["train_samples"],
            batch_size=data_config["train_batch_size"],
            shuffle=False,
            resample=False,
        )
    test_loader = None
    if data_config["test_samples"] > 0:
        test_loader = SDFDataloader(
            sampler,
            num_samples=data_config["test_samples"],
            batch_size=data_config["test_batch_size"],
            shuffle=False,
            resample=False,
        )

    return train_loader, test_loader, sampler

def train_model(config: Dict,
                out_dir: str,
                model_name: str,
                train_loader: Optional[SDFDataloader] = None,
                test_loader: Optional[SDFDataloader] = None, 
                data_dir: str = None) -> Tuple[GridNet3D, str, str]:
    """Train a neural SDF model with the specified configuration.

    Args:
        config (Dict): User-provided configuration dictionary for training
        out_dir (str): Parent directory where training outputs will be stored
        model_name (str): Name of the model, used for creating output directories and files
        train_loader (Optional[SDFDataloader]): Pre-computed training data loader. If None,
            will be created from config
        test_loader (Optional[SDFDataloader]): Pre-computed test data loader. If None,
            will be created from config
        data_dir (str, optional): Base directory for finding mesh files specified in config

    Returns:
        Tuple containing:
            - GridNet3D: The trained model
            - str: Path to the stored configuration file
            - str: Path to the saved model weights

    Raises:
        ValueError: If mesh file specified in config cannot be found
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Create or use existing data loaders
    if train_loader is None or test_loader is None:
        train_loader, test_loader, sampler = create_data_loaders(config, data_dir)
    else:
        sampler = train_loader.sampler
    
    domain_bounds = train_loader.domain_bounds
    domain_bounds_jnp = jnp.asarray(domain_bounds)
    config["model"]["domain_bounds"] = domain_bounds.tolist()

    # -------------------------------------------------------------------------
    # Store final configuration
    # -------------------------------------------------------------------------

    # Store the complete configuration file in output directory
    # so we know what generated the results
    store_config_path = path.join(out_dir, f"{model_name}_config.yaml")
    with open(store_config_path, "w") as f:
        yaml.dump(config, f)

    # -------------------------------------------------------------------------
    # Model architecture
    # -------------------------------------------------------------------------

    # Initialize network
    model = GridNet3D(
        domain_bounds=domain_bounds,
        num_grid_points=config['model']['num_grid_points'],
        feature_size=config['model']['feature_size'],
        width_size=config['model']['width_size'],
        out_size=config['model']['out_size'],
        key=key,
        interp_method=config['model']['interp_method'],
        activation_fn=activation_fn_from_str(config["model"]["activation_fn"])
    )

    # Define optimization variables and helper functions
    filter_spec = jax.tree.map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda net: net.feature_grid.f, filter_spec, True)
    filter_spec = eqx.tree_at(lambda net: net.mlp, filter_spec, eqx.is_array)

    (
        extract_optimization_variables_from_model,
        combine_optimization_variable_w_model,
        unflatten_opt_vars,
    ) = create_opt_vars_helpers_from_filter_spec(model, filter_spec)

    opt_vars, static_model = extract_optimization_variables_from_model(model)

    # -------------------------------------------------------------------------
    # Loss function and auxiliary metrics
    # -------------------------------------------------------------------------
    sdf_w = config["loss"]["sdf_mse_weight"]
    eik_w = config["loss"]["eikonal_weight"]
    cp_w = config["loss"]["closest_point_weight"]

    loss_weights = {"Max SDF error": sdf_w, "Max Eikonal error": eik_w, "Max closest point error": cp_w}
    loss_aux_metrics = {name: 0.0 for name, weight in loss_weights.items() if weight > 0.0}

    def loss_fn(model, x_samples, true_sdf_vals):
        if eik_w > 0.0 or cp_w > 0.0:
            # Compile time branching
            pred_sdf_vals, grad_xs = vmap(value_and_grad(model))(x_samples)
        else:
            pred_sdf_vals = vmap(model)(x_samples)

        loss_combined = 0.0
        metrics = loss_aux_metrics
        if sdf_w > 0.0:
            # Compile time branching
            loss_mse, max_sdf_error = mse_loss_with_max(pred_sdf_vals, true_sdf_vals)
            loss_combined = loss_combined + sdf_w * loss_mse
            metrics = eqx.tree_at(lambda m: m["Max SDF error"], metrics, max_sdf_error, is_leaf=lambda x: x is None)

        if eik_w > 0.0:
            # Compile time branching
            loss_eik, max_eik_error = eikonal_loss_with_max(grad_xs)
            loss_combined = loss_combined + eik_w * loss_eik
            metrics = eqx.tree_at(lambda m: m["Max Eikonal error"], metrics, max_eik_error, is_leaf=lambda x: x is None)

        if cp_w > 0.0:
            # Compile time branching
            loss_cp, max_cp_error = closest_point_loss_with_max(model, x_samples, pred_sdf_vals, grad_xs, domain_bounds_jnp)
            loss_combined = loss_combined + cp_w * loss_cp
            metrics = eqx.tree_at(lambda m: m["Max closest point error"], metrics, max_cp_error, is_leaf=lambda x: x is None)

        return loss_combined, metrics

    # -------------------------------------------------------------------------
    # Logging and plotting during training
    # -------------------------------------------------------------------------
    writer = create_summary_writer(out_dir)

    def write_metric_fn(metrics, epoch):
        print_metrics(metrics, epoch)
        log_metrics(writer, metrics, epoch)

    def write_plot_fn(model, epoch):
        write_slice_comparison(
            model,
            sampler.sdf,
            writer,
            epoch,
            domain_bounds,
            slice_axis=2,
            location=0.5,
            n_pts=300,
        )

    def save_fn(model, epoch):
        # Return the filepath for saved model
        filepath = path.join(out_dir, f"{model_name}_epoch_{epoch}.eqx")
        model.save(filepath)
        return filepath

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    # Initialize optimizer
    lr = float(config["train"]["learning_rate"])
    opt = optax.adabelief(lr)

    # Run optimization
    model, saved_model_path = train(
        model,
        extract_optimization_variables_from_model,
        combine_optimization_variable_w_model,
        train_loader,
        test_loader,
        loss_fn,
        loss_aux_metrics,
        opt,
        num_epochs=config["train"]["num_epochs"],
        write_fn=write_metric_fn,
        save_fn=save_fn,
        save_interval=config["train"]["save_interval"],
        plot_fn=write_plot_fn,
        plot_interval=config["train"]["plot_interval"],
    )

    return model, store_config_path, saved_model_path

def evaluate_model(model: GridNet3D, model_name: str, out_dir: str, test_loader: SDFDataloader):
    """Evaluate a trained model using various metrics."""
    # Inference time
    #  - Need to check average time per batch
    # Memory
    #  - How much memory for the batch size
    # Accuracy:
    #  - MSE loss
    #  - Max SDF error
    #  - Max SDF error percentage

    print(f"Evaluating model {model_name}...")

    # JIT the evaluation function
    def value_and_grad_fn(model, x_samples):
        return vmap(value_and_grad(model))(x_samples)
    jitted_value_and_grad_fn = jax.jit(value_and_grad_fn)

    # We will want need both SDF values and gradients
    total_inference_time = 0.0
    total_memory_use = 0.0
    total_mse_loss = 0.0
    max_sdf_error = 0.0
    max_sdf_error_frac = 0.0

    for i, batch in enumerate(test_loader):
        x_batch, true_sdf_vals = batch

        # Get predictions
        # And record time taken
        # TODO: Check memory usage
        tik = time.time()
        pred_sdf_vals, grad_xs = jitted_value_and_grad_fn(model, x_batch)
        jax.block_until_ready(pred_sdf_vals)
        jax.block_until_ready(grad_xs)
        total_inference_time += time.time() - tik

        # Calculate metrics
        mse_loss, max_sdf_err_batch, max_sdf_err_frac_batch = (
            mse_loss_with_max_and_frac(pred_sdf_vals, true_sdf_vals)
        )

        total_mse_loss += mse_loss
        max_sdf_error = max(max_sdf_error, max_sdf_err_batch)
        max_sdf_error_frac = max(max_sdf_error_frac, max_sdf_err_frac_batch)
    
    avg_inference_time = total_inference_time / len(test_loader)

    # print and output metrics
    print(f"Model {model_name} evaluation metrics:")
    print(f"  - Avg inference time per {test_loader.batch_size} samples: {avg_inference_time}")
    print(f"  - Total MSE loss: {total_mse_loss}")
    print(f"  - Max SDF error: {max_sdf_error}")
    print(f"  - Max SDF error percentage: {max_sdf_error_frac}")

    eval_results = {
        "model_name": model_name,
        "num_samples": test_loader.num_samples,
        "batch_size": test_loader.batch_size,
        "avg_inference_time": avg_inference_time,
        "total_mse_loss": total_mse_loss,
        "max_sdf_error": max_sdf_error,
        "max_sdf_error_frac": max_sdf_error_frac,
    }

    json.dump(eval_results, open(path.join(out_dir, f"{model_name}_eval_results.json"), "w"))

    return eval_results