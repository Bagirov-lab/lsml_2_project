from comet_ml.api import API

api = API() 
api.download_registry_model("bagirov-lab", "lora_pet_resnet18", version="1.1.0", output_path="./", expand=True, stage=None)