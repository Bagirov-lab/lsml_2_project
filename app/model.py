import comet_ml
from dotenv import load_dotenv
from os import environ
import torch.nn as nn
import torch
from torchvision.models import get_model
import logging

from .train import create_model_with_lora, LORALayer

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
            expand=True
        )
        logging.info("Model Downloaded")
    except Exception:
        logging.error(f"Model Was Not Loaded, due to error: {Exception.__traceback__}")

    return None

def load_model():
    
    # Initiate LoRA
    # Apply LORA to the last layer of the model
    
    final_layer_n_classes = 37
    model = get_model(name='resnet18', weights="DEFAULT")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, final_layer_n_classes)
    model.fc = LORALayer(model.fc)
    
    # Load the state dictionary into the model
    MODEL_FILE = 'model-data/comet-torch-model.pth'
    state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'),
                            weights_only=True)
    model.load_state_dict(state_dict)
    
    return model

if __name__ == "__main__":
    logging.info("Logging to the Comet ML service...")
    comet_ml.login()
    
    logging.info("Download model from registry...")
    download_model_from_comet()
    
    logging.info("Load Experiment Params...")
    comet_api = comet_ml.API() 
    
    model_comet = comet_api.get_model(workspace=environ.get("COMET_WORKSPACE"), model_name=environ.get("COMET_MODEL_NAME"))
    
    model_comet_details = model_comet.get_details(version=environ.get("COMET_MODEL_NAME_VERSION"))

    model_comet_details_experiment_key = model_comet_details.get("experimentKey")
    
    comet_experiment = comet_api.get_experiment_by_key(model_comet_details_experiment_key)

    # final_layer_n_classes
    base_layer_name = comet_experiment.get_parameters_summary("base_layer_name").get("valueCurrent")
    if len(base_layer_name) > 0:
        base_layer_name  = base_layer_name.get("valueCurrent")
    else:
        raise ValueError(
            "Experiment does not have base_layer_name parameter. "
            f"\ncheck id {model_comet_details_experiment_key}"
        )
    
    final_layer_n_classes = comet_experiment.get_parameters_summary("final_layer_n_classes")
    if len(final_layer_n_classes) > 0:
        try:
            final_layer_n_classes  = final_layer_n_classes.get("valueCurrent")
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
    
    weights = comet_experiment.get_parameters_summary("weights")
    if len(weights) > 0:
        weights  = weights.get("valueCurrent")
    else:
        logging.warning("No weights param, using default value 'DEFAULT'...")
        weights = 'DEFAULT'
    
    logging.info("Initialise of the Full Model...")
    model = create_model_with_lora(
        base_layer_name=base_layer_name,
        final_layer_n_classes=final_layer_n_classes,
        weights=weights
    )
    
    logging.info("Load state dictionary...")
    state_dict = torch.load(
        environ.get("COMET_MODEL_FILE"), 
        map_location=torch.device('cpu'),
        weights_only=True
    )
    model.load_state_dict(state_dict)
