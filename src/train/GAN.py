import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# Definición del Generador mejorado
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),  # Normalización por batch
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),  # Normalización por batch
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),  # Normalización por batch
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


# Definición del Discriminador mejorado
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),  # Nueva capa con más neuronas
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Definir hiperparámetros
    batch_size = 64
    lr = 0.0001  # Learning rate más pequeño para entrenar más lentamente y con más precisión
    epochs = 50  # Aumentar las épocas para mejorar la calidad de las imágenes

    # Dataset MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Inicialización del modelo
    G = Generator()
    D = Discriminator()

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    # Entrenamiento
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(trainloader):
            # Obtén el tamaño actual del batch dinámicamente
            batch_size = real_images.size(0)  # Tamaño dinámico, puede ser menor en el último batch

            real_images = real_images.view(batch_size, -1)

            # Entrenar al Discriminador
            optimizer_D.zero_grad()

            # Crear las etiquetas con el tamaño del batch dinámico
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Salidas para las imágenes reales
            outputs = D(real_images)
            loss_real = criterion(outputs, real_labels)

            # Crear ruido y generar imágenes falsas
            noise = torch.randn(batch_size, 128)  # Vector de ruido más grande (128)
            fake_images = G(noise)

            # Salidas para las imágenes falsas
            outputs = D(fake_images.detach())
            loss_fake = criterion(outputs, fake_labels)

            # Calcular y propagar el error para el Discriminador
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # Entrenar al Generador
            optimizer_G.zero_grad()
            outputs = D(fake_images)
            loss_G = criterion(outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()

        # Mostrar el progreso y guardar las imágenes cada 2 épocas
        if epoch % 2 == 0:
            print(f"Epoch [{epoch}/{epochs}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

            # Guardar la imagen generada
            fake_image = fake_images[0].view(28, 28).detach().numpy()
            plt.imshow(fake_image, cmap="gray")
            plt.savefig(f"tmp_results/GAN/epoch_{epoch}.png")
            plt.close()  # Cerrar la figura para liberar memoria

    # Guardar los modelos:
    torch.save(G.state_dict(), "trained_models/GAN/generator.pth")
    torch.save(D.state_dict(), "trained_models/GAN/discriminator.pth")
