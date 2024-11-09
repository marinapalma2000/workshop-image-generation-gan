import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        self.label_emb = nn.Embedding(10, 128)  # Embedding para convertir la etiqueta a un vector
        self.model = nn.Sequential(
            nn.Linear(128 + 128, 256),  # Se combina el ruido con la etiqueta
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)  # Convertir la etiqueta en un vector
        x = torch.cat([noise, label_embedding], dim=1)  # Combinar el ruido y la etiqueta
        return self.model(x)


latent_dim = 128  # Dimensión del vector de ruidos


# Cargar el generador y generar una nueva imagen de un número específico
def generate_specific_number(number):
    G = ConditionalGenerator()
    G.load_state_dict(torch.load("model_pro/conditional_generator.pth"))
    G.eval()

    noise = torch.randn(1, latent_dim)
    label = torch.tensor([number])  # Etiqueta para el número escogido

    with torch.no_grad():
        generated_image = G(noise, label)

    generated_image = (generated_image + 1) / 2
    generated_image = generated_image.view(28, 28).detach().numpy()

    plt.imshow(generated_image, cmap="gray")
    plt.title(f"Imagen generada del número {number}")
    plt.show()

    plt.imsave("results_pro/new_generated_image.png", generated_image, cmap="gray")


# Generar una imagen del número deseado (ejemplo: 3)
generate_specific_number(9)
