{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"},"kaggle":{"accelerator":"gpu","dataSources":[],"dockerImageVersionId":30805,"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":true}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"import comet_ml\nfrom comet_ml.integration.pytorch import log_model\nfrom comet_ml import CometExperiment\n\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nimport torch\nimport torch.nn as nn\nfrom torch.optim import lr_scheduler\nfrom torch.utils.data import DataLoader, random_split\n\nimport torchvision.transforms as transforms\nfrom torchvision.transforms import ToPILImage\nfrom torchvision.datasets import OxfordIIITPet\nfrom torchvision.models import get_model\n\nfrom sklearn.metrics import precision_score, recall_score, f1_score\n\nfrom tqdm import tqdm\n\nfrom kaggle_secrets import UserSecretsClient\n\n\nclass CONFIG:\n    batch_size = 128\n    val_ratio = 0.1\n    base_layer_name = \"resnet34\"\n    weights = \"DEFAULT\"\n    final_layer_n_classes = 37\n    num_epochs = 25\n    optimizer_lr = 0.001\n    scheduler_step_size = 7\n    scheduler_gamma = 0.1\n    patience = 5  # Number of epochs to wait before stopping if no improvement\n\n    def model_log_name(self):\n        return f\"LoRA_Pet_{CONFIG.base_layer_name}\"\n\n\ndef calculate_metrics(y_true, y_pred):\n    y_true = y_true.cpu().numpy()\n    y_pred = y_pred.cpu().numpy()\n\n    precision = precision_score(y_true, y_pred, average=\"weighted\", zero_division=0)\n    recall = recall_score(y_true, y_pred, average=\"weighted\", zero_division=0)\n    f1 = f1_score(y_true, y_pred, average=\"weighted\", zero_division=0)\n\n    return precision, recall, f1\n\n\ndef show_sample_images(dataset, num_labels=5):\n    fig, axs = plt.subplots(1, num_labels, figsize=(15, 3))\n    label_samples = {}\n\n    for image, label in dataset:\n        # Convert tensor image to PIL for display\n        image = ToPILImage()(image)\n\n        if label not in label_samples:\n            label_samples[label] = image\n        if len(label_samples) == num_labels:\n            break\n\n    for i, (label, image) in enumerate(label_samples.items()):\n        axs[i].imshow(image)\n        axs[i].set_title(f\"Label: {label}\")\n        axs[i].axis(\"off\")\n\n    plt.show()\n\n\nclass LORALayer(nn.Module):\n    def __init__(self, adapted_layer, rank=16):\n        super(LORALayer, self).__init__()\n        self.adapted_layer = adapted_layer\n        self.A = nn.Parameter(torch.randn(adapted_layer.weight.size(1), rank))\n        self.B = nn.Parameter(torch.randn(rank, adapted_layer.weight.size(0)))\n\n    def forward(self, x):\n        low_rank_matrix = self.A @ self.B\n        adapted_weight = self.adapted_layer.weight + low_rank_matrix.t()\n        return nn.functional.linear(x, adapted_weight, self.adapted_layer.bias)\n\n\ndef create_model_with_lora(\n    base_layer_name: str,\n    final_layer_n_classes: int,\n    weights: str,\n    exp: CometExperiment | None = None,\n):\n    # Load Base Learner\n    model = get_model(name=base_layer_name, weights=weights)\n\n    num_ftrs = model.fc.in_features\n\n    model.fc = nn.Linear(num_ftrs, final_layer_n_classes)\n    model.fc = LORALayer(model.fc)\n\n    if exp is not None:\n        exp.log_parameter(\"base_layer_name\", base_layer_name)\n        exp.log_parameter(\"num_ftrs\", num_ftrs)\n        exp.log_parameter(\"final_layer_n_classes\", final_layer_n_classes)\n\n    return model\n\n\nif __name__ == \"__main__\":\n    user_secrets = UserSecretsClient()\n\n    experiment = comet_ml.start(\n        api_key=user_secrets.get_secret(\"comet_api_key\"),\n        project_name=\"pet-recognition\",\n        workspace=user_secrets.get_secret(\"comet_workspace\"),\n    )\n\n    transform_train = transforms.Compose(\n        [\n            transforms.Resize((224, 224)),  # Resize to fit model input\n            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally\n            transforms.RandomRotation(degrees=15),  # Random rotation\n            transforms.ColorJitter(\n                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1\n            ),  # Random color adjustment\n            transforms.RandomAffine(\n                degrees=15, translate=(0.1, 0.1)\n            ),  # Slight translation\n            transforms.RandomResizedCrop(\n                224, scale=(0.8, 1.0)\n            ),  # Crop randomly within the size range\n            transforms.ToTensor(),  # Convert to tensor\n            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize\n        ]\n    )\n\n    transform_test = transforms.Compose(\n        [\n            transforms.Resize((224, 224)),  # Resize images to fit the model input size\n            transforms.ToTensor(),\n        ]\n    )\n\n    # Load the dataset\n    train_dataset = OxfordIIITPet(\n        root=\"data/\", split=\"trainval\", download=True, transform=transform_train\n    )\n    test_dataset = OxfordIIITPet(\n        root=\"data/\", split=\"test\", download=True, transform=transform_test\n    )\n\n    experiment.log_parameter(\"train_class_names\", train_dataset.classes)\n    experiment.log_parameter(\"train_class_names\", test_dataset.classes)\n\n    # Assuming 'train_dataset' and 'test_dataset' are already loaded\n\n    # Print the size of train and test datasets\n    print(f\"Size of Training Dataset: {len(train_dataset)}\")\n    print(f\"Size of Test Dataset: {len(test_dataset)}\")\n\n    # Display total labels (assuming label information is available in the dataset)\n    total_labels = len(np.unique([label for _, label in train_dataset]))\n    print(f\"Total Labels: {total_labels}\")\n\n    # Show sample images from the training dataset\n    show_sample_images(train_dataset)\n\n    batch_size = CONFIG.batch_size  # PARAM\n    val_ratio = CONFIG.val_ratio  # PARAM\n\n    # Define the size of the validation set\n    val_size = int(val_ratio * len(train_dataset))\n    train_size = len(train_dataset) - val_size\n\n    # Split the dataset\n    train_data, val_data = random_split(train_dataset, [train_size, val_size])\n\n    # Create DataLoaders\n    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)\n    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)\n    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n\n    ### Model\n    model = create_model_with_lora(\n        base_layer_name=CONFIG.base_layer_name,\n        final_layer_n_classes=CONFIG.final_layer_n_classes,\n        weights=CONFIG.weights,\n        exp=experiment,\n    )\n\n    # Train\n    experiment.log_parameter(\"num_epochs\", CONFIG.num_epochs)\n\n    # Check if CUDA (GPU support) is available and use it; otherwise, fall back to CPU\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    experiment.log_parameter(\"device_train\", device)\n\n    # Move the model to the selected device\n    model = model.to(device)\n\n    # Track losses and accuracies\n    train_losses = []\n    val_losses = []\n    test_losses = []\n\n    train_accuracies = []\n    val_accuracies = []\n    test_accuracies = []\n\n    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.optimizer_lr)\n    experiment.log_parameter(\"optimizer_lr\", CONFIG.optimizer_lr)\n    experiment.log_parameter(\"optimizer\", \"Adam\")\n\n    criterion = nn.CrossEntropyLoss()\n    experiment.log_parameter(\"criterion\", \"CrossEntropyLoss\")\n\n    # Implement a learning rate scheduler\n    scheduler = lr_scheduler.StepLR(\n        optimizer, step_size=CONFIG.scheduler_step_size, gamma=CONFIG.scheduler_gamma\n    )  # PARAM\n    experiment.log_parameter(\"scheduler_step_size\", CONFIG.scheduler_step_size)\n    experiment.log_parameter(\"scheduler_gamma\", CONFIG.scheduler_gamma)\n\n    best_val_accuracy = 0.0\n    best_model_wts = model.state_dict()\n    patience = CONFIG.patience\n    experiment.log_parameter(\"patience\", patience)\n    best_val_loss = float(\"inf\")\n    patience_counter = 0\n\n    for epoch in range(CONFIG.num_epochs):\n        model.train()\n        running_loss = 0.0\n        correct = 0\n        total = 0\n\n        for images, labels in train_loader:\n            images, labels = images.to(device), labels.to(device)\n            optimizer.zero_grad()\n            outputs = model(images)\n            loss = criterion(outputs, labels)\n            loss.backward()\n            optimizer.step()\n\n            running_loss += loss.item()\n            _, predicted = torch.max(outputs.data, 1)\n            total += labels.size(0)\n            correct += (predicted == labels).sum().item()\n\n        # Training Metrics\n        train_labels_all = []\n        train_preds_all = []\n\n        with torch.no_grad():\n            for images, labels in train_loader:\n                images, labels = images.to(device), labels.to(device)\n                outputs = model(images)\n                _, predicted = torch.max(outputs, 1)\n\n                train_labels_all.append(labels)\n                train_preds_all.append(predicted)\n\n        # Concatenate all batches into single tensors\n        train_labels_all = torch.cat(train_labels_all)\n        train_preds_all = torch.cat(train_preds_all)\n\n        # Calculate metrics\n        train_precision, train_recall, train_f1 = calculate_metrics(\n            train_labels_all, train_preds_all\n        )\n\n        scheduler.step()\n        train_accuracy = 100 * correct / total\n        train_losses.append(running_loss / len(train_loader))\n        train_accuracies.append(train_accuracy)\n\n        # Validation Phase\n        model.eval()\n        val_running_loss = 0.0\n        correct = 0\n        total = 0\n\n        with torch.no_grad():\n            for images, labels in val_loader:\n                images, labels = images.to(device), labels.to(device)\n                outputs = model(images)\n                loss = criterion(outputs, labels)\n                val_running_loss += loss.item()\n                _, predicted = torch.max(outputs.data, 1)\n                total += labels.size(0)\n                correct += (predicted == labels).sum().item()\n\n        val_labels_all = []\n        val_preds_all = []\n\n        with torch.no_grad():\n            for images, labels in val_loader:\n                images, labels = images.to(device), labels.to(device)\n                outputs = model(images)\n                loss = criterion(outputs, labels)\n                val_running_loss += loss.item()\n                _, predicted = torch.max(outputs.data, 1)\n\n                val_labels_all.append(labels)\n                val_preds_all.append(predicted)\n\n        # Concatenate all batches into single tensors\n        val_labels_all = torch.cat(val_labels_all)\n        val_preds_all = torch.cat(val_preds_all)\n\n        # Calculate metrics\n        val_precision, val_recall, val_f1 = calculate_metrics(\n            val_labels_all, val_preds_all\n        )\n\n        val_loss_epoch = val_running_loss / len(val_loader)\n        val_accuracy = 100 * correct / total\n        val_losses.append(val_loss_epoch)\n        val_accuracies.append(val_accuracy)\n\n        # Early Stopping Check\n        if val_loss_epoch < best_val_loss:\n            best_val_loss = val_loss_epoch\n            best_model_wts = model.state_dict()\n            patience_counter = 0  # Reset patience counter\n        else:\n            patience_counter += 1\n\n        if patience_counter >= patience:\n            print(f\"Early stopping triggered at epoch {epoch}\")\n            break\n\n        # Log metrics to Comet\n        experiment.log_metrics(\n            {\n                \"Train Loss Epoch\": running_loss / len(train_loader),\n                \"Train Accuracy Epoch\": train_accuracy,\n                \"Train Precision Epoch\": train_precision,\n                \"Train Recall Epoch\": train_recall,\n                \"Train F1-Score Epoch\": train_f1,\n                \"Val Loss Epoch\": val_loss_epoch,\n                \"Val Accuracy Epoch\": val_accuracy,\n                \"Val Precision Epoch\": val_precision,\n                \"Val Recall Epoch\": val_recall,\n                \"Val F1-Score Epoch\": val_f1,\n            },\n            epoch=epoch,\n        )\n\n        print(\n            f\"Epoch {epoch} \"\n            f\"Train Loss: {running_loss / len(train_loader):.4f}, \"\n            f\"Train Accuracy: {train_accuracy:.2f}%, \"\n            f\"Val Loss: {val_loss_epoch:.4f}, \"\n            f\"Val Accuracy: {val_accuracy:.2f}%, \"\n            f\"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, \"\n            f\"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}\"\n        )\n\n        torch.save(model.state_dict(), f\"model_epoch_{epoch}.pt\")\n        print(f\"Saved model at epoch {epoch}\")\n\n    # # Save\n\n    # Load the best model weights\n    model.load_state_dict(best_model_wts)\n\n    # Log the pytorch model to Comet\n    log_model(experiment=experiment, model=model, model_name=CONFIG().model_log_name())\n\n    experiment.register_model(CONFIG().model_log_name())\n\n    # # Test\n\n    # Test phase (after training is complete and best model is loaded)\n    model.eval()\n    correct = 0\n    total = 0\n    fig = plt.figure(figsize=(25, 5))  # Define figure size\n\n    # We will visualize the first 10 images of the test set\n    for i, (images, labels) in enumerate(test_loader, start=1):\n        if i > 10:  # Stop after visualizing 10 images\n            break\n        images, labels = images.to(device), labels.to(device)\n        outputs = model(images)\n        _, predicted = torch.max(outputs, 1)\n\n        total += labels.size(0)\n        correct += (predicted == labels).sum().item()\n\n        ax = fig.add_subplot(2, 5, i)  # Plotting 10 images in 2 rows and 5 columns\n        ax.imshow(images[0].cpu().numpy().transpose((1, 2, 0)))\n        ax.set_title(f\"True: {labels[0].item()}, Pred: {predicted[0].item()}\")\n        ax.axis(\"off\")\n\n    test_accuracy = 100 * correct / total\n    print(f\"Test Accuracy: {test_accuracy}%\")\n    plt.show()\n\n    labels_all = None\n    predicted_all = None\n    for i, (images, labels) in tqdm(enumerate(test_loader, start=1)):\n        images, labels = images.to(device), labels.to(device)\n\n        if labels_all is None:\n            labels_all = labels\n        else:\n            labels_all = torch.cat((labels_all, labels))\n\n        outputs = model(images)\n        _, predicted = torch.max(outputs, 1)\n\n        if predicted_all is None:\n            predicted_all = predicted\n        else:\n            predicted_all = torch.cat((predicted_all, predicted))\n\n    total = labels_all.size(0)\n    correct = (predicted_all == labels_all).sum().item()\n    test_accuracy = 100 * correct / total\n\n    experiment.log_metric(\"Test Accuracy\", test_accuracy, step=None)\n    experiment.log_confusion_matrix(labels_all, predicted_all)\n\n    # # End Experiment\n\n    experiment.end()","metadata":{"_uuid":"e6455bbc-de74-402f-895c-a085a3c99171","_cell_guid":"7a21a764-826a-41a5-aa19-0011ee5941bd","trusted":true,"collapsed":false,"jupyter":{"outputs_hidden":false}},"outputs":[],"execution_count":null}]}