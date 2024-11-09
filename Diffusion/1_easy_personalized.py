import gc
import multiprocessing

import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline

# Set multiprocessing and memory management settings
multiprocessing.set_start_method("spawn", force=True)
gc.collect()
torch.cuda.empty_cache()

# Load pre-trained Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"  # Personalizable by prompt
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to("cpu")

# Choose a prompt for the type of image you want to generate
prompt = "a beautiful landscape with mountains and a lake during sunset"

# Generate an image based on the prompt
with torch.no_grad():
    generated_image = pipeline(prompt, height=128, width=128, num_images_per_prompt=1).images[0]


# Display the generated image
plt.imshow(generated_image)
plt.axis("off")
plt.show()
