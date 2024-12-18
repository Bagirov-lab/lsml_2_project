import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from loguru import logger as logging

from app.comet import load_model

from app.model import class_to_breed

load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path=".env.model")

# Load the model on startup
model = load_model()

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow all origins. Replace "*" with specific origins if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.).
    allow_headers=["*"],  # Allow all headers.
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the pet type from an uploaded image.
    """
    logging.debug("Predict: Got Request")
    try:
        # Read and preprocess the image
        image = Image.open(file.file).convert("RGB")
        logging.debug("Read input into RGB")
        
        # Define transformation for incoming image
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        logging.debug("Transformed incoming image")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor) 
            # Convert logits to probabilities using the softmax function
            probabilities = torch.softmax(outputs, dim=1)

            # Find the position of the highest probability class
            max_prob_index = torch.argmax(probabilities, dim=1).item()

            # Get the corresponding probability
            max_prob_value = probabilities[0, max_prob_index].item()

        logging.debug("Maked prediction")
        
        # Map prediction to breed name
        breed_name = class_to_breed.get(max_prob_index, "Unknown Breed")
        logging.debug("Maped prediction to breed name")

        # Return the predicted class and breed name
        return {
            "filename": file.filename,
            "probabilty": max_prob_value,
            "breed_name": breed_name,
        }
    except Exception as e:
        logging.error(f"Predict: Error {e.__traceback__})")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
