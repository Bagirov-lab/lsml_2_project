import torch
import uvicorn
import comet_ml
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
import comet_ml
from dotenv import load_dotenv
from os import environ
import torch.nn as nn
import torch
from torchvision.models import get_model


load_dotenv(dotenv_path=".env")


def load_model():
    
    # Load Model State
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
    state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    return model

# Load environment variables
load_dotenv(dotenv_path=".env")
comet_ml.login()

# Initialize the FastAPI app
app = FastAPI()

# Define transformation for incoming images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the model on startup
model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the pet type from an uploaded image.
    """
    try:
        # Read and preprocess the image
        image = Image.open(file.file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Return the predicted class
        return {"filename": file.filename, "prediction": predicted.item()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
