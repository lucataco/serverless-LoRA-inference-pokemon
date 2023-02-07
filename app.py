import torch
import base64
from io import BytesIO
from huggingface_hub import model_info
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    # LoRA weights ~3 MB
    model_path = "pcuenq/pokemon-lora"
    info = model_info(model_path)
    model_base = info.cardData["base_model"]
    model = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
    model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
    model.unet.load_attn_procs(model_path)
    model.to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    steps = model_inputs.get('inference_steps', 30)
    scale = model_inputs.get('guidance_scale', 7)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    image = model(prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
     # Return the results as a dictionary
    return {'image_base64': image_base64}
