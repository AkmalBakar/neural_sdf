import sys
import os
from os import path
import glob
import argparse
import yaml

# Add root directory to sys path
sys.path.append(path.realpath(path.dirname(path.dirname(__file__))))

from neural_sdf import core, models, datalog

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Neural SDF model')
    parser.add_argument('--config', type=str, help='Path to config file (YAML)')
    parser.add_argument('--model_name', type=str, default=None, 
                       help='Name of the model (default: derived from config filename)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Root directory for outputs (default: ../experiment_outputs)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load config from file if provided, otherwise use default config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default minimal config
        config = {
            "data": {"mesh_file": "usb_male.obj"},
            "model": {"num_grid_points": [5, 5, 5]},
            "train": {"num_epochs": 2000}
        }
    
    # Complete the config with default values
    config = core.complete_config(config)

    # Use provided model name or derive from config file
    if args.model_name:
        model_name = args.model_name
    elif args.config:
        # Use config filename without extension as model name
        model_name = path.splitext(path.basename(args.config))[0]
    else:
        model_name = "resolution_5_5_5"

    # Use provided output directory or default to experiment_outputs
    if args.output_dir:
        root_output_dir = path.realpath(args.output_dir)
    else:
        root_output_dir = path.realpath(path.join(path.dirname(__file__), "..", "experiment_outputs"))
    
    os.makedirs(root_output_dir, exist_ok=True)
    out_dir = path.join(root_output_dir, model_name)

    # Folder where the meshes can be found
    data_dir = path.join(path.dirname(__file__), "..", "data")

    # Create dataloaders
    train_loader, test_loader, sampler = core.create_data_loaders(config["data"], data_dir)

    # Train model
    model, stored_config_path, saved_model_path = core.train_model(
        config, out_dir, model_name, train_loader, test_loader, data_dir=data_dir
    )

    # Load the latest model
    model_files = glob.glob(path.join(out_dir, f"{model_name}_epoch_*.eqx"))
    model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    model = models.GridNet3D.load(model_files[-1])

    # Evaluate model
    eval_results = core.evaluate_model(model, model_name, out_dir, test_loader)
    print(eval_results)

    print("Done!")