import comet_ml
from dotenv import load_dotenv
from os import environ
import torch.nn as nn
import torch
from torchvision.models import get_model
import logging


load_dotenv(dotenv_path=".env")

def download_model():
    # Download Model State
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

    # Declaire LoRA
    class LORALayer(nn.Module):
        def __init__(self, adapted_layer, rank=16):
            super(LORALayer, self).__init__()
            self.adapted_layer = adapted_layer
            self.A = nn.Parameter(torch.randn(adapted_layer.weight.size(1), rank))
            self.B = nn.Parameter(torch.randn(rank, adapted_layer.weight.size(0)))

        def forward(self, x):
            low_rank_matrix = self.A @ self.B
            adapted_weight = self.adapted_layer.weight + low_rank_matrix.t()  # Ensure correct shape
            return nn.functional.linear(x, adapted_weight, self.adapted_layer.bias)
    
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
    print("Starting the model download...")
    download_model()

    # print("Loading the model...")
    # model = load_model()

    # if model:
    #     print("Model successfully loaded!")
    # else:
    #     print("Failed to load the model.")