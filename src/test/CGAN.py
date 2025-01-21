import matplotlib.pyplot as plt
import torch

from ..train.CGAN import ConditionalGenerator


def generate_specific_number(number):
    # Crear una instancia del Generador y los pesos entrenados
    G = ConditionalGenerator()
    G.load_state_dict(torch.load("trained_models/CGAN/generator.pth"))
    G.eval()

    noise = torch.randn(1, 128)
    label = torch.tensor([number])  # Etiqueta para el número escogido

    with torch.no_grad():
        generated_image = G(noise, label)

    generated_image = (generated_image + 1) / 2
    generated_image = generated_image.view(28, 28).detach().numpy()

    plt.imshow(generated_image, cmap="gray")
    plt.title(f"Imagen generada del número {number}")
    plt.imsave("tmp_results/CGAN/new_generated_image.png", generated_image, cmap="gray")
    plt.show()


# Generar una imagen del número deseado (ejemplo: 3)
generate_specific_number(6)
