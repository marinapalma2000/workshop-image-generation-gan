import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Crear la carpeta "results" si no existe
os.makedirs("results_pro", exist_ok=True)


# Definición del Generador condicional
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


# Definición del Discriminador condicional
class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 784)  # Embedding para convertir la etiqueta a un vector
        self.model = nn.Sequential(
            nn.Linear(784 + 784, 1024),  # Se combina la imagen con la etiqueta
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)  # Convertir la etiqueta en un vector
        x = torch.cat([img, label_embedding], dim=1)  # Combinar la imagen y la etiqueta
        return self.model(x)


# Definir hiperparámetros
batch_size = 64
lr = 0.0002
epochs = 50
latent_dim = 128  # Dimensión del vector de ruido

# Dataset MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Inicialización del modelo condicional
G = ConditionalGenerator()
D = ConditionalDiscriminator()

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Entrenamiento
for epoch in range(epochs):
    for i, (real_images, labels) in enumerate(trainloader):
        batch_size = real_images.size(0)
        real_images = real_images.view(batch_size, -1)

        # Generar ruido y etiquetas aleatorias para las imágenes falsas
        noise = torch.randn(batch_size, latent_dim)
        random_labels = torch.randint(0, 10, (batch_size,))

        # Entrenar al Discriminador
        optimizer_D.zero_grad()

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Salidas para las imágenes reales con las etiquetas correspondientes
        outputs = D(real_images, labels)
        loss_real = criterion(outputs, real_labels)

        # Generar imágenes falsas con las etiquetas aleatorias
        fake_images = G(noise, random_labels)

        # Salidas para las imágenes falsas
        outputs = D(fake_images.detach(), random_labels)
        loss_fake = criterion(outputs, fake_labels)

        # Calcular y propagar el error para el Discriminador
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Entrenar al Generador
        optimizer_G.zero_grad()
        outputs = D(fake_images, random_labels)
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{epochs}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Guardar ejemplos de imágenes generadas cada 10 épocas
    if epoch % 10 == 0:
        noise = torch.randn(10, latent_dim)  # Generar 10 imágenes
        sample_labels = torch.arange(10)  # Números del 0 al 9
        generated_images = G(noise, sample_labels)
        generated_images = generated_images.view(10, 28, 28).detach().numpy()

        # Guardar las imágenes generadas
        for j in range(10):
            plt.imshow(generated_images[j], cmap="gray")
            plt.title(f"Generado número {sample_labels[j]}")
            plt.savefig(f"results_pro/generated_image_epoch_{epoch}_num_{sample_labels[j]}.png")
            plt.close()

# Guardar el modelo del Generador y Discriminador
torch.save(G.state_dict(), "model_pro/conditional_generator.pth")
torch.save(D.state_dict(), "model_pro/conditional_discriminator.pth")
