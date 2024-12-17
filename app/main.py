import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
import logging

from app.model import load_model, class_to_breed

load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path=".env.model")

# Load the model on startup
model = load_model()

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. Replace "*" with specific origins if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.).
    allow_headers=["*"],  # Allow all headers.
)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the pet type from an uploaded image.
    """
    logging.info("Predict: Got Request")
    try:
        # Read and preprocess the image
        image = Image.open(file.file).convert("RGB")
        # Define transformation for incoming images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Map prediction to breed name
        class_index = predicted.item()
        breed_name = class_to_breed.get(class_index, "Unknown Breed")
        
        logging.info("Predict: Success")
        
        # Return the predicted class and breed name
        return {
            "filename": file.filename,
            "prediction": class_index,
            "breed_name": breed_name
        }
    except Exception as e:
        logging.info(f"Predict: Error {e.__traceback__})")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
