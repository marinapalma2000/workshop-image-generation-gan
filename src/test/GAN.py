import matplotlib.pyplot as plt
import torch

from ..train.GAN import Generator

# Crear una instancia del Generador y los pesos entrenados
G = Generator()
G.load_state_dict(torch.load("trained_models/GAN/generator.pth", weights_only=True))
G.eval()  # Modelo en modo evaluación

# Generar una nueva imagen usando ruido aleatorio
noise = torch.randn(1, 128)
generated_image = G(noise)

# Convertir la imagen generada al formato correcto (28x28)
generated_image = (generated_image + 1) / 2
generated_image = generated_image.view(28, 28).detach().numpy()

# Guardar y mostrar
plt.imshow(generated_image, cmap="gray")
plt.title("Imagen generada")
plt.imsave("tmp_results/GAN/new_generated_image.png", generated_image, cmap="gray")
plt.show()
