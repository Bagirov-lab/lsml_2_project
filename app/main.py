import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

from app.model import load_model

load_dotenv(dotenv_path=".env")

class_to_breed = {
    0: "Abyssinian",
    1: "American Bulldog",
    2: "American Pit Bull Terrier",
    3: "Basset Hound",
    4: "Beagle",
    5: "Bengal",
    6: "Birman",
    7: "Bombay",
    8: "Boxer",
    9: "British Shorthair",
    10: "Chihuahua",
    11: "Egyptian Mau",
    12: "English Cocker Spaniel",
    13: "English Setter",
    14: "German Shorthaired",
    15: "Great Pyrenees",
    16: "Havanese",
    17: "Japanese Chin",
    18: "Keeshond",
    19: "Leonberger",
    20: "Maine Coon",
    21: "Miniature Pinscher",
    22: "Newfoundland",
    23: "Persian",
    24: "Pomeranian",
    25: "Pug",
    26: "Ragdoll",
    27: "Russian Blue",
    28: "Saint Bernard",
    29: "Samoyed",
    30: "Scottish Terrier",
    31: "Shiba Inu",
    32: "Siamese",
    33: "Sphynx",
    34: "Staffordshire Bull Terrier",
    35: "Wheaten Terrier",
    36: "Yorkshire Terrier"
}

# Load environment variables
load_dotenv(dotenv_path=".env")

# Load the model on startup
model = load_model()

# Initialize the FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the pet type from an uploaded image.
    """
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
        
        # Return the predicted class and breed name
        return {
            "filename": file.filename,
            "prediction": class_index,
            "breed_name": breed_name
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
