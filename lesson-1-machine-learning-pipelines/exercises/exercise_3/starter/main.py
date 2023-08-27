import mlflow
import os
import wandb
import hydra
from omegaconf import DictConfig

# mlflow.set_tracking_uri("azureml://westeurope.api.azureml.ms/mlflow/v1.0/subscriptions/cdfebb73-07a2-4fa8-b753-62da76199209/resourceGroups/Udacity/providers/Microsoft.MachineLearningServices/workspaces/MLflows")
mlflow.set_experiment("my_exp")

# Your ML code and MLflow logging here

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "download_data"),
        "main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        },
    )

    ##################
    # Your code here: use the artifact we created in the previous step as input for the `process_data` step
    # and produce a new artifact called "cleaned_data".
    # NOTE: use os.path.join(root_path, "process_data") to get the path
    # to the "process_data" component
    ##################
    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        "main",
        parameters={
            "input_artifact": "iris.csv:latest",
            "artifact_name": "cleaned_data.csv",
            "artifact_type": "processed_data",
            "artifact_description": "Cleaned data"
        },
    )


if __name__ == "__main__":
    go()
