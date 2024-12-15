# %% [markdown] {"id":"LBH3tsiIPDjB","jupyter":{"outputs_hidden":false}}
# # Config

# %% [code] {"execution":{"iopub.status.busy":"2024-12-09T12:03:26.072741Z","iopub.execute_input":"2024-12-09T12:03:26.073692Z","iopub.status.idle":"2024-12-09T12:03:26.078809Z","shell.execute_reply.started":"2024-12-09T12:03:26.073643Z","shell.execute_reply":"2024-12-09T12:03:26.077694Z"},"jupyter":{"outputs_hidden":false}}
class CONFIG:
    batch_size = 128 # PARAM
    val_ratio = 0.1 # PARAM
    base_layer_name = 'resnet34' # PARAM
    final_layer_n_classes = 37
    num_epochs = 25 # PARAM
    optimizer_lr = 0.001
    scheduler_step_size = 7
    scheduler_gamma = 0.1
    def model_log_name(self):
        return f"LoRA_Pet_{CONFIG.base_layer_name}"

# %% [code] {"id":"efc801a8","execution":{"iopub.status.busy":"2024-12-09T09:37:49.147358Z","iopub.execute_input":"2024-12-09T09:37:49.147830Z","iopub.status.idle":"2024-12-09T09:38:03.864737Z","shell.execute_reply.started":"2024-12-09T09:37:49.147789Z","shell.execute_reply":"2024-12-09T09:38:03.862833Z"},"jupyter":{"outputs_hidden":false}}
#!pip install torchvision torch
!pip install comet_ml

# %% [code] {"id":"eCa3rXOjJnA9","execution":{"iopub.status.busy":"2024-12-09T09:38:03.867226Z","iopub.execute_input":"2024-12-09T09:38:03.867627Z","iopub.status.idle":"2024-12-09T09:38:08.477820Z","shell.execute_reply.started":"2024-12-09T09:38:03.867587Z","shell.execute_reply":"2024-12-09T09:38:08.476597Z"},"jupyter":{"outputs_hidden":false}}
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
from torch.optim import Adam

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Data

# %% [code] {"id":"_ra5IOvYJuaL","outputId":"8d935377-f33a-415f-e5f7-e09e3c0a5653","execution":{"iopub.status.busy":"2024-12-09T09:38:08.479031Z","iopub.execute_input":"2024-12-09T09:38:08.479507Z","iopub.status.idle":"2024-12-09T09:38:24.518665Z","shell.execute_reply.started":"2024-12-09T09:38:08.479469Z","shell.execute_reply":"2024-12-09T09:38:24.517322Z"},"jupyter":{"outputs_hidden":false}}
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the model input size
    transforms.ToTensor(),
])

# Load the dataset
train_dataset = OxfordIIITPet(root='data/', split='trainval', download=True, transform=transform)
test_dataset = OxfordIIITPet(root='data/', split='test', download=True, transform=transform)

# %% [code] {"id":"d6cc341a","outputId":"f031be64-7ba9-4bb0-e553-b2de5cce7b85","execution":{"iopub.status.busy":"2024-12-09T09:38:24.521953Z","iopub.execute_input":"2024-12-09T09:38:24.522485Z","iopub.status.idle":"2024-12-09T09:38:48.641215Z","shell.execute_reply.started":"2024-12-09T09:38:24.522431Z","shell.execute_reply":"2024-12-09T09:38:48.639837Z"},"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage

# Assuming 'train_dataset' and 'test_dataset' are already loaded

# Print the size of train and test datasets
print(f"Size of Training Dataset: {len(train_dataset)}")
print(f"Size of Test Dataset: {len(test_dataset)}")

# Display total labels (assuming label information is available in the dataset)
total_labels = len(np.unique([label for _, label in train_dataset]))
print(f"Total Labels: {total_labels}")

# Function to show sample images for 5 labels
def show_sample_images(dataset, num_labels=5):
    fig, axs = plt.subplots(1, num_labels, figsize=(15, 3))
    label_samples = {}

    for image, label in dataset:
        # Convert tensor image to PIL for display
        image = ToPILImage()(image)

        if label not in label_samples:
            label_samples[label] = image
        if len(label_samples) == num_labels:
            break

    for i, (label, image) in enumerate(label_samples.items()):
        axs[i].imshow(image)
        axs[i].set_title(f"Label: {label}")
        axs[i].axis('off')

    plt.show()

# Show sample images from the training dataset
show_sample_images(train_dataset)

# %% [markdown] {"id":"ba0a3f6c","jupyter":{"outputs_hidden":false}}
# # DataLoader

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# Batch Size Recommendations:
# 
# 1.	ResNet-18 or ResNet-34:
# 
# * Start with a batch size of 128.
# * Increase to 256 if VRAM permits or use mixed precision training for optimization.
# 
# 2.	ResNet-50 or ResNet-101:
# 
# * Start with a batch size of 64.
# * Use gradient accumulation to simulate larger batch sizes if needed.

# %% [code] {"id":"DdrghwHwLWMK","execution":{"iopub.status.busy":"2024-12-09T09:38:48.642661Z","iopub.execute_input":"2024-12-09T09:38:48.643048Z","iopub.status.idle":"2024-12-09T09:38:48.663843Z","shell.execute_reply.started":"2024-12-09T09:38:48.643009Z","shell.execute_reply":"2024-12-09T09:38:48.662409Z"},"jupyter":{"outputs_hidden":false}}
from torch.utils.data import DataLoader
from torch.utils.data import random_split


batch_size = CONFIG.batch_size # PARAM
val_ratio = CONFIG.val_ratio # PARAM


# Define the size of the validation set
val_size = int(val_ratio * len(train_dataset))
train_size = len(train_dataset) - val_size

# Split the dataset
train_data, val_data = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Model

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## COMET

# %% [code] {"execution":{"iopub.status.busy":"2024-12-09T09:38:48.665541Z","iopub.execute_input":"2024-12-09T09:38:48.666028Z","iopub.status.idle":"2024-12-09T09:38:48.671623Z","shell.execute_reply.started":"2024-12-09T09:38:48.665977Z","shell.execute_reply":"2024-12-09T09:38:48.670351Z"},"jupyter":{"outputs_hidden":false}}
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-09T09:38:48.673832Z","iopub.execute_input":"2024-12-09T09:38:48.674325Z","iopub.status.idle":"2024-12-09T09:38:55.681324Z","shell.execute_reply.started":"2024-12-09T09:38:48.674270Z","shell.execute_reply":"2024-12-09T09:38:55.679291Z"},"jupyter":{"outputs_hidden":false}}
import comet_ml
experiment = comet_ml.start(
  api_key=user_secrets.get_secret("comet_api_key"),
  project_name="pet-recognition",
  workspace=user_secrets.get_secret("comet_workspace")
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Load Model

# %% [code] {"id":"d8e143dc","execution":{"iopub.status.busy":"2024-12-09T09:38:55.683781Z","iopub.execute_input":"2024-12-09T09:38:55.684168Z","iopub.status.idle":"2024-12-09T09:38:56.403721Z","shell.execute_reply.started":"2024-12-09T09:38:55.684123Z","shell.execute_reply":"2024-12-09T09:38:56.402571Z"},"jupyter":{"outputs_hidden":false}}
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import get_model, get_model_weights

# Load a pre-trained ResNet model
base_layer_name = CONFIG.base_layer_name # PARAM
model = get_model(name=base_layer_name, weights="DEFAULT")
experiment.log_parameter('base_layer_name', base_layer_name)

num_ftrs = model.fc.in_features

# Adjust the final layer for 37 classes
final_layer_n_classes = CONFIG.final_layer_n_classes
experiment.log_parameter('final_layer_n_classes', final_layer_n_classes)
model.fc = nn.Linear(num_ftrs, final_layer_n_classes)

# LORA adaptation
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

# Apply LORA to the last layer of the model
model.fc = LORALayer(model.fc)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Train

# %% [code] {"execution":{"iopub.status.busy":"2024-12-09T09:38:56.404979Z","iopub.execute_input":"2024-12-09T09:38:56.405792Z","iopub.status.idle":"2024-12-09T09:55:30.590746Z","shell.execute_reply.started":"2024-12-09T09:38:56.405753Z","shell.execute_reply":"2024-12-09T09:55:30.589400Z"},"jupyter":{"outputs_hidden":false}}
from torch.optim import lr_scheduler

# PARAM
experiment.log_parameter('num_epochs', CONFIG.num_epochs)


# Check if CUDA (GPU support) is available and use it; otherwise, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
experiment.log_parameter('device_train', device)

# Move the model to the selected device
model = model.to(device)

# Track losses and accuracies
train_losses = []
val_losses = []
test_losses = []

train_accuracies = []
val_accuracies = []
test_accuracies = []

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.optimizer_lr)
experiment.log_parameter('optimizer_lr', CONFIG.optimizer_lr)
experiment.log_parameter('optimizer', "Adam")

criterion = nn.CrossEntropyLoss()
experiment.log_parameter('criterion', "CrossEntropyLoss")


# Implement a learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG.scheduler_step_size, gamma=CONFIG.scheduler_gamma) # PARAM
experiment.log_parameter('scheduler_step_size', CONFIG.scheduler_step_size)
experiment.log_parameter('scheduler_gamma', CONFIG.scheduler_gamma)

best_val_accuracy = 0.0
best_model_wts = model.state_dict()

for epoch in range(CONFIG.num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()  # Adjust the learning rate based on the scheduler
    train_accuracy = 100 * correct / total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    val_losses.append(val_running_loss / len(val_loader))
    val_accuracies.append(val_accuracy)

    # Test phase
    model.eval()
    test_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    test_losses.append(test_running_loss / len(test_loader))
    test_accuracies.append(test_accuracy)

    # Save the model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_wts = model.state_dict()

    train_loss_epoch = running_loss / len(train_loader)
    val_loss_epoch = val_running_loss / len(val_loader)
    test_running_loss = test_running_loss / len(test_loader)
                                            
    experiment.log_metrics({
        "Train Loss Epoch": train_loss_epoch,
        "Train Accuracy Epoch": train_accuracy,
        "Val Loss Epoch": val_loss_epoch,
        "Val Accuracy Epoch": val_accuracy,
        "Test Loss Epoch": val_loss_epoch,
        "Test Accuracy Epoch": test_accuracy
    }, epoch = epoch)

    print(f'Epoch {epoch}, Train Loss: {train_loss_epoch}, Train Accuracy: {train_accuracy}%, Val Loss: {val_loss_epoch}, Val Accuracy: {val_accuracy}%')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Save

# %% [code] {"id":"iw-duEPrGq_p","execution":{"iopub.status.busy":"2024-12-09T10:08:55.600515Z","iopub.execute_input":"2024-12-09T10:08:55.601440Z","iopub.status.idle":"2024-12-09T10:08:55.938409Z","shell.execute_reply.started":"2024-12-09T10:08:55.601375Z","shell.execute_reply":"2024-12-09T10:08:55.937166Z"},"jupyter":{"outputs_hidden":false}}
from comet_ml.integration.pytorch import log_model

# Load the best model weights
model.load_state_dict(best_model_wts)

# Log the pytorch model to Comet
log_model(
    experiment=experiment,
    model=model,
    model_name=CONFIG().model_log_name()
)

experiment.register_model(CONFIG().model_log_name())

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Test

# %% [code] {"id":"698c7922","outputId":"065db04f-ca4c-426a-ebf0-00552e64e88e","execution":{"iopub.status.busy":"2024-12-09T10:13:45.845436Z","iopub.execute_input":"2024-12-09T10:13:45.845871Z","iopub.status.idle":"2024-12-09T10:15:01.438454Z","shell.execute_reply.started":"2024-12-09T10:13:45.845831Z","shell.execute_reply":"2024-12-09T10:15:01.436850Z"},"jupyter":{"outputs_hidden":false}}
# Test phase (after training is complete and best model is loaded)
model.eval()
correct = 0
total = 0
fig = plt.figure(figsize=(25, 5))  # Define figure size

# We will visualize the first 10 images of the test set
for i, (images, labels) in enumerate(test_loader, start=1):
    if i > 10:  # Stop after visualizing 10 images
        break
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    ax = fig.add_subplot(2, 5, i)  # Plotting 10 images in 2 rows and 5 columns
    ax.imshow(images[0].cpu().numpy().transpose((1, 2, 0)))
    ax.set_title(f"True: {labels[0].item()}, Pred: {predicted[0].item()}")
    ax.axis('off')

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy}%')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-09T10:36:48.043840Z","iopub.execute_input":"2024-12-09T10:36:48.044301Z","iopub.status.idle":"2024-12-09T10:37:25.104377Z","shell.execute_reply.started":"2024-12-09T10:36:48.044262Z","shell.execute_reply":"2024-12-09T10:37:25.103057Z"},"jupyter":{"outputs_hidden":false}}
from tqdm import tqdm
# import tensorflow as tf

labels_all = None
predicted_all = None
for i, (images, labels) in tqdm(enumerate(test_loader, start=1)):
    images, labels = images.to(device), labels.to(device)
    
    if labels_all is None:
        labels_all = labels
    else:
        labels_all = torch.cat((labels_all, labels))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    if predicted_all is None:
        predicted_all = predicted
    else:
        predicted_all = torch.cat((predicted_all, predicted))
    
    
total = labels_all.size(0)
correct = (predicted_all == labels_all).sum().item()
test_accuracy = 100 * correct / total

experiment.log_metric("Test Accuracy", test_accuracy, step=None)
experiment.log_confusion_matrix(labels_all, predicted_all)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # End Experiment

# %% [code] {"id":"85U2wAFAGvfe","execution":{"iopub.status.busy":"2024-12-09T10:50:54.266850Z","iopub.execute_input":"2024-12-09T10:50:54.267339Z","iopub.status.idle":"2024-12-09T10:50:56.341464Z","shell.execute_reply.started":"2024-12-09T10:50:54.267300Z","shell.execute_reply":"2024-12-09T10:50:56.340281Z"},"jupyter":{"outputs_hidden":false}}
experiment.end()