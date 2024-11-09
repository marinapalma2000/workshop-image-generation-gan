import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Definición del Generador (debe ser idéntico al definido durante el entrenamiento)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
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

    def forward(self, x):
        return self.model(x)


# Crear una instancia del Generador
G = Generator()

# Cargar los pesos entrenados
G.load_state_dict(torch.load("model_medium/generator.pth"))

# Asegúrate de que el modelo esté en modo evaluación
G.eval()

# Generar una nueva imagen usando ruido aleatorio
noise = torch.randn(1, 128)  # Vector de ruido aleatorio de tamaño 128
generated_image = G(noise)

# La imagen generada está en el rango [-1, 1], la llevamos al rango [0, 1] para visualizarla
generated_image = (generated_image + 1) / 2

# Convertir la imagen generada al formato correcto (28x28)
generated_image = generated_image.view(28, 28).detach().numpy()

# Mostrar la nueva imagen generada
plt.imshow(generated_image, cmap="gray")
plt.title("Imagen generada")
plt.show()

# También puedes guardar la imagen si prefieres
plt.imsave("results_medium/new_generated_image.png", generated_image, cmap="gray")
