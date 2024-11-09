import matplotlib.pyplot as plt
import torch
from diffusers import DDPMPipeline

# Load pre-trained diffusion pipeline
model_id = "google/ddpm-celebahq-256"  # Celebrities high quality faces
pipeline = DDPMPipeline.from_pretrained(model_id)
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Generate an image from random noise
with torch.no_grad():
    generated_image = pipeline(num_inference_steps=50).images[0]

# Display the generated image
plt.imshow(generated_image)
plt.axis("off")
plt.show()
