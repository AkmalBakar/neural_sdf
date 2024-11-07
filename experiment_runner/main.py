import sys
import os
from os import path
import glob
# Add root directory to sys path
sys.path.append(path.realpath(path.dirname(path.dirname(__file__))))

from neural_sdf import core, models, datalog

if __name__ == "__main__":
    # TODO: Make this run multiple experiments
    # And evaluate different models on inference time, memory usage, etc.
    resolutions = [5, 10, 15, 20, 25]
    feature_sizes = [10, 15, 20, 25, 30]
    width_sizes = [20, 40, 80, 160, 320]

    for resolution in resolutions:
        for feature_size in feature_sizes:
            for width_size in width_sizes:
                config = {
                    "data": {"mesh_file": "usb_male.obj"},
                    "model": {
                        "num_grid_points": [resolution, resolution, resolution],
                        "feature_size": feature_size,
                        "width_size": width_size,
                    },
                    "train": {"num_epochs": 2000},
                }
                # Complete the config with default values
                config = core.complete_config(config)

                # Parent folder for outputs: model definition, model weights, tensorboard logs (summary and plots)
                root_output_dir = path.realpath(
                    path.join(path.dirname(__file__), "..", "experiment_outputs")
                )
                os.makedirs(root_output_dir, exist_ok=True)

                # Folder where the meshes can be found
                data_dir = path.join(path.dirname(__file__), "..", "data")

                # Create dataloaders
                train_loader, test_loader, sampler = core.create_data_loaders(
                    config["data"], data_dir
                )

                # Name of this specific model
                model_name = f"resolution_{resolution}_{resolution}_{resolution}_feature_size_{feature_size}_width_size_{width_size}"

                # Train model
                out_dir = path.join(root_output_dir, model_name)
                model, stored_config_path, saved_model_path = core.train_model(
                    config,
                    out_dir,
                    model_name,
                    train_loader,
                    test_loader,
                    data_dir=data_dir,
                )

                # Load the latest model
                model_files = glob.glob(path.join(out_dir, f"{model_name}_epoch_*.eqx"))
                model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                model = models.GridNet3D.load(model_files[-1])

                # Evaluate model
                eval_results = core.evaluate_model(
                    model, model_name, out_dir, test_loader
                )
                print(eval_results)

                print("Done!")
