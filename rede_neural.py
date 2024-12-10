import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# Definindo a conversão de imagem para tensor
transform = transforms.ToTensor()

# Carregando os conjuntos de treino e validação
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)

dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)  # Correção: usar next(dataiter) em vez de dataiter.next()
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
plt.show()

print(imagens[0].shape)  # Para ver as dimensões do tensor imagem
print(etiquetas[0].shape)  # Para ver as dimensões do tensor etiqueta

print(imagens[0])
print(etiquetas[0])

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, X):
        X = X.view(X.shape[0], -1)  # Convertendo a imagem para vetor de 784 elementos
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return X

def treino(modelo, trainloader, device):
    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5)
    inicio = time()
    criterio = nn.CrossEntropyLoss()  # Usando CrossEntropyLoss que inclui softmax
    EPOCHS = 30
    modelo.train()

    for epoch in range(EPOCHS):
        perda_acumulada = 0
        for imagens, etiquetas in trainloader:
            imagens, etiquetas = imagens.to(device), etiquetas.to(device)
            otimizador.zero_grad()
            output = modelo(imagens)
            perda_instantanea = criterio(output, etiquetas)
            perda_instantanea.backward()
            otimizador.step()
            perda_acumulada += perda_instantanea.item()

        print(f"Epoch {epoch+1} - Perda resultante: {perda_acumulada/len(trainloader):.4f}")

    print("\nTempo de treino (em minutos) =", (time() - inicio) / 60)

def validacao(modelo, valloader, device):
    modelo.eval()
    conta_corretas, conta_todas = 0, 0
    with torch.no_grad():
        for imagens, etiquetas in valloader:
            imagens, etiquetas = imagens.to(device), etiquetas.to(device)
            output = modelo(imagens)
            _, predicoes = torch.max(output, 1)
            conta_corretas += (predicoes == etiquetas).sum().item()
            conta_todas += etiquetas.size(0)

    print(f"Total de imagens testadas = {conta_todas}")
    print(f"\nPrecisão do modelo = {100 * conta_corretas / conta_todas:.2f}%")

# Função para visualizar os pesos da primeira camada
def visualizar_pesos(modelo):
    pesos = modelo.linear1.weight.detach().cpu().numpy()  # Obtendo os pesos da primeira camada
    num_neuronios = pesos.shape[0]
    fig, axes = plt.subplots(8, 16, figsize=(15, 8))  # Ajuste do grid para exibir os neurônios
    for i, ax in enumerate(axes.flat):
        if i < num_neuronios:
            ax.imshow(pesos[i].reshape(28, 28), cmap='gray')
            ax.set_title(f'Neuron {i + 1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Inicializando e treinando o modelo
modelo = Modelo()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

# Treinando o modelo
treino(modelo, trainloader, device)

# Visualizando os pesos da primeira camada
visualizar_pesos(modelo)

# Salvando e carregando o modelo
torch.save(modelo.state_dict(), './meu_modelo.pt')
modelo_carregado = Modelo()
modelo_carregado.load_state_dict(torch.load('./meu_modelo.pt'))
modelo_carregado.eval()
modelo_carregado.to(device)
