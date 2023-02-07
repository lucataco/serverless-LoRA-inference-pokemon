# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import torch
from huggingface_hub import model_info
from diffusers import StableDiffusionPipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model_path = "pcuenq/pokemon-lora"
    info = model_info(model_path)
    model_base = info.cardData["base_model"]
    model = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)

if __name__ == "__main__":
    download_model()