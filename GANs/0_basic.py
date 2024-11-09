import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# Definición del Generador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784), nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# Definición del Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Definir hiperparámetros
batch_size = 64
lr = 0.0002
epochs = 10

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
    for real_images, _ in trainloader:
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
        noise = torch.randn(batch_size, 100)
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

    # Mostrar el progreso
    if epoch % 2 == 0:
        print(f"Epoch [{epoch}/{epochs}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        plt.imshow(fake_images[0].view(28, 28).detach().numpy(), cmap="gray")
        plt.show()
