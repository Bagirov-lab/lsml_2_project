import comet_ml
from dotenv import load_dotenv
from os import environ
import torch.nn as nn
import torch
from torchvision.models import get_model
from loguru import logger as logging

from app.train import create_model_with_lora, LORALayer

load_dotenv(dotenv_path=".env")


def download_model_from_comet():
    try:
        comet_ml.login()

        api = comet_ml.API()

        model_comet = api.get_model(
            workspace=environ.get("COMET_WORKSPACE"),
            model_name=environ.get("COMET_MODEL_NAME"),
        )

        model_comet.download(
            version=environ.get("COMET_MODEL_NAME_VERSION"),
            output_folder="./",
            expand=True,
        )
        logging.info("Model Downloaded")
    except Exception:
        logging.error(f"Model Was Not Loaded, due to error: {Exception.__traceback__}")

    return None


def create_env_file(env_dict: dict, 
                    file_path: str, 
                    logging = logging):
    """
    Create a .env file from a given dictionary.

    Args:
        env_dict (dict): Dictionary containing environment variables as key-value pairs.
        file_path (str): Path to the .env file (default is ".env").
    """
    try:
        with open(file_path, "w") as env_file:
            for key, value in env_dict.items():
                env_file.write(f"{key}={value}\n")
        try:
            logging.info(f"Env file successfully created at: {file_path}")
        except Exception:
            print(f"Env file successfully created at: {file_path}")
    except Exception as e:
        try:
            logging.error(f"Error creating env file: {e}")
        except Exception:
            print(f"Error creating env file: {e}")


def load_model():
    final_layer_n_classes = int(environ.get("FINAL_LAYER_N_CLASSES"))
    weights = environ.get("WEIGHTS")
    base_layer_name = environ.get("BASE_LAYER_NAME")
    comet_model_file = environ.get("COMET_MODEL_FILE")

    model = get_model(name=base_layer_name, weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, final_layer_n_classes)
    model.fc = LORALayer(model.fc)

    # Load the state dictionary into the model
    state_dict = torch.load(
        f=comet_model_file, map_location=torch.device("cpu"), weights_only=True
    )
    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    logging.info("Logging to the Comet ML service...")
    comet_ml.login()

    logging.info("Download model from registry...")
    download_model_from_comet()

    logging.info("Load Comet Model...")
    comet_api = comet_ml.API()

    model_comet = comet_api.get_model(
        workspace=environ.get("COMET_WORKSPACE"),
        model_name=environ.get("COMET_MODEL_NAME"),
    )

    logging.info("Load Version of the Model...")
    model_comet_details = model_comet.get_details(
        version=environ.get("COMET_MODEL_NAME_VERSION")
    )

    logging.info("Get the Experiment Key of the Version...")
    model_comet_details_experiment_key = model_comet_details.get("experimentKey")

    comet_experiment = comet_api.get_experiment_by_key(
        model_comet_details_experiment_key
    )

    logging.info("Retrive Experiment Hyper Params...")
    
    logging.info("Retrive base_layer_name hyperparameter...")
    base_layer_name = comet_experiment.get_parameters_summary("base_layer_name")
    
    if len(base_layer_name) > 0:
        base_layer_name = base_layer_name.get("valueCurrent")
    else:
        raise ValueError(
            "Experiment does not have base_layer_name parameter. "
            f"\ncheck id {model_comet_details_experiment_key}"
        )

    logging.info("Retrive final_layer_n_classes hyperparameter...")
    final_layer_n_classes = comet_experiment.get_parameters_summary(
        "final_layer_n_classes"
    )
    if len(final_layer_n_classes) > 0:
        final_layer_n_classes = final_layer_n_classes.get("valueCurrent")
        try:
            final_layer_n_classes = int(final_layer_n_classes)
        except ValueError:
            raise ValueError(
                "Experiment has final_layer_n_classes parameter which is not int"
                f"\ncheck id {model_comet_details_experiment_key}"
            )
    else:
        raise ValueError(
            "Experiment does not have final_layer_n_classes parameter. "
            f"\ncheck id {model_comet_details_experiment_key}"
        )

    logging.info("Retrive weights hyperparameter...")
    weights = comet_experiment.get_parameters_summary("weights")
    if len(weights) > 0:
        weights = weights.get("valueCurrent")
    else:
        logging.warning("No weights param, using default value 'DEFAULT'...")
        weights = "DEFAULT"

    logging.info("Initialise of the Full Model...")
    model = create_model_with_lora(
        base_layer_name=base_layer_name,
        final_layer_n_classes=final_layer_n_classes,
        weights=weights,
    )

    logging.info("Load state dictionary...")
    comet_model_file = environ.get("COMET_MODEL_FILE")
    state_dict = torch.load(
        f=comet_model_file, map_location=torch.device("cpu"), weights_only=True
    )
    model.load_state_dict(state_dict)

    logging.info("Save model params...")
    create_env_file(
        env_dict={
            "FINAL_LAYER_N_CLASSES": final_layer_n_classes,
            "WEIGHTS": weights,
            "BASE_LAYER_NAME": base_layer_name,
            "COMET_MODEL_FILE": comet_model_file,
        },
        file_path=".env.model",
    )