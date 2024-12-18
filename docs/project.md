# Task Definition: Lovely Pets Project 🐾

This document defines the task, input and output specifications, chosen approach to solve the task, the dataset used for training, and the runtime architecture for the resulting service.

---

## Task Definition

The task is to develop a **full-stack machine learning service** for pet classification using a pre-trained deep learning model. The service allows users to classify pet images via an API and serves the results to a front-end application.

---

## Input and Output Description

### Input

- **Back-End**:
  - The input to the back-end service is an image of a pet in standard image formats (e.g., `JPG`, `PNG`).
  - The API receives the image as a POST request with multipart form data.

- **Front-End**:
  - The input is provided via a user-friendly web interface where users can upload an image.

### Output

- **Back-End**:
  - The back-end service processes the image and returns the predicted pet type as a JSON response:

     ```json
     {
            "probabilty": 0.6,
            "breed_name": "Abyssinian",
    }
     ```

- **Front-End**:
  - The front-end displays the prediction result (breed name and confidence score) to the user.

---

## Approach to Solve the Task

### Model Description

The service uses a **LoRA (Low-Rank Adaptation)** fine-tuned **ResNet-18** model, which is lightweight and efficient for classification tasks. The approach involves the following steps:

1. **Model Fine-Tuning**:
   - A pre-trained ResNet-18 model is adapted using LoRA to classify pet images into categories like `Abyssinian`, `Bengal`, etc.
   - LoRA reduces the number of trainable parameters, making the model lightweight and efficient.

2. **Comet ML Integration**:
   - The model is trained and logged to the **Comet ML** platform.
   - During deployment, the back-end service downloads the model from the Comet registry.

PS: Duing train I ended up on **ResNet-18**, but for addition I also tried other versions of **ResNet** and not only.

---

## Dataset for Model Training

The model is trained on a publicly available pet image dataset, such as:

- **Oxford-IIIT Pet Dataset**:
  - Contains 37 categories of pets (cats and dogs).
  - Includes around 7,400 labeled images.

### Dataset Summary

- **Number of Images**: ~7,400
- **Labels**: 37 pet breeds
- **Format**: Images in `JPG` format with corresponding class labels.

---

## Data Transformations for Training

During training, a series of **data transformations** are applied to the training samples to augment the dataset and improve the generalization of the model.

---

### Transformation Pipeline

The following transformations are applied to each training sample:

| **Transformation**              | **Description**                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| `Resize((224, 224))`            | Resizes the input image to a fixed size of 224x224 pixels, which matches the ResNet-18 input shape.  |
| `RandomHorizontalFlip(p=0.5)`   | Randomly flips the image horizontally with a probability of 50%, introducing variation to the dataset. |
| `RandomRotation(degrees=15)`    | Rotates the image randomly within a range of ±15 degrees, simulating minor orientation changes.     |
| `ColorJitter`                   | Randomly adjusts the image’s brightness, contrast, saturation, and hue.                            |
| `RandomAffine`                  | Applies random affine transformations, allowing slight translation (10% in both x and y directions). |
| `RandomResizedCrop`             | Randomly crops and resizes the image within a scale range of 80% to 100% of the original size.      |
| `ToTensor()`                    | Converts the image from a PIL Image to a PyTorch tensor.                                           |
| `Normalize`                     | Normalizes the image using the ImageNet mean and standard deviation values.                        |

By applying these transformations, the training pipeline becomes more robust and capable of handling diverse inputs, leading to improved model performance and generalization.

## Runtime Architecture

The project follows a microservices-based architecture with the following components:

### 1. **Back-End Service**

- Built using **Python** and **FastAPI**.
- Responsibilities:
  - Download the pre-trained model from Comet ML.
  - Process user-uploaded pet images and make predictions.
  - Return the results in a JSON format.
- Dockerized for scalability.

### 2. **Front-End Service**

- A static web application served using **NGINX**.
- Responsibilities:
  - Provides a user interface to upload pet images.
  - Communicates with the back-end API to fetch predictions.
  - Displays the prediction results (pet type and confidence score).

### 3. **Docker Compose**

- Local and production environments are managed with **Docker Compose**.
  - **Local Development**: Builds images locally and runs both services.
  - **Production**: Uses prebuilt Docker images from a registry.

---

Here’s the Architecture Diagram section updated to fit .md format with proper Markdown syntax:

## Architecture Diagram

Below is the simplified architecture:

```plaintext
+------------------+         +------------------+
|    Front-End     |         |    Back-End      |
|  (NGINX + HTML)  | <-----> | (FastAPI + Model)|
+------------------+         +------------------+
           |                           |
           |      Docker Compose       |
           +---------------------------+
                   |       |
       Local Environment   Production Environment
```

---

### Notes

- **Front-End**: Serves the user interface and static content (HTML, CSS, JS).
- **Back-End**: Provides API endpoints and runs the machine learning model.
- **Docker Compose**: Manages service orchestration for both local development and production environments.

This format uses a plaintext code block to ensure consistent rendering on GitHub and maintains clean alignment.

---

## Summary

| Component        | Description                           |
|------------------|---------------------------------------|
| **Task**         | Pet classification using ML.         |
| **Input**        | Pet image (`JPG`, `PNG`).            |
| **Output**       | Predicted pet type and confidence.   |
| **Model**        | LoRA fine-tuned ResNet-18.           |
| **Dataset**      | Oxford-IIIT Pet Dataset.             |
| **Back-End**     | Python + FastAPI + Docker.           |
| **Front-End**    | Static web app served via NGINX.     |
| **Deployment**   | Managed via Docker Compose.          |

---

This document defines the task clearly and outlines the solution, from input/output definitions to the chosen model, dataset, and runtime architecture.
