import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import gdown

from torchvision import transforms as T
from torchvision.ops import Permute
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.notebook import tqdm
from matplotlib import gridspec

device = 'cuda'

def label_one_hot(vocab, num_classes, labels):
    """Convierte una lista de etiquetas en representaciones one-hot.

    Parameters:
    --------------------------
      vocab: dict
        Un diccionario que mapea caracteres a índices.
      num_classes: int
        Número total de clases posibles.
      labels: list or tuple
        Lista o tupla de etiquetas a convertir en one-hot.

    Returns:
    --------------------------
      labels_one_hot: torch.Tensor
        Tensor que contiene la representación one-hot de las etiquetas.

    """
    # Inicializa una lista vacía para almacenar los vectores one-hot de las etiquetas
    labels_one_hot = []

    # Itera sobre cada etiqueta en la lista de etiquetas
    for label in labels:
        # Inicializa un vector de ceros de tamaño num_classes para representar el one-hot de una etiqueta
        one_hot_vec = [0.]*num_classes

        # Itera sobre cada carácter en la etiqueta actual
        for char in label:
          # Obtiene el índice correspondiente al carácter en el vocabulario
          idx=vocab[char]
          # Incrementa en 1 la posición correspondiente al índice del carácter en el vector one-hot
          one_hot_vec[idx]+=1
        # Agrega el vector one-hot de la etiqueta actual a la lista de one-hot de etiquetas
        labels_one_hot.append(one_hot_vec)

    # Convierte la lista de vectores one-hot en un tensor
    labels_one_hot = torch.tensor(labels_one_hot, device=device)

    return labels_one_hot

def train_hvae(model,
               train_loader,
               max_epochs=100,
               device=device,
               warmup=lambda epoch: 1.0,
               prior_weight=lambda epoch: 1.0,
               vocab={char:index for index, char in enumerate("qwertyuiopasdfghjklzxcvbnm26")},
               conditioned=False):
    """Entrena un modelo HVAE utilizando un conjunto de datos de entrenamiento.

    Parameters:
    --------------------------
      model: nn.Module
        Modelo HVAE a entrenar.
      train_loader: torch.utils.data.DataLoader
        DataLoader que contiene los datos de entrenamiento.
      max_epochs: int (Default:100)
        Número máximo de épocas de entrenamiento.
      device: str
        Dispositivo de entrenamiento ('cuda' o 'cpu').
      warmup: callable (Default:lambda epoch: 1.0)
        Función de epoch que pondera el término de consistencia.
      prior_weight: callable (Default:lambda epoch: 1.0)
        Función de epoch que pondera el término de prior matching.

    Returns:
    --------------------------
      train_loss: list
        Lista con la pérdida total en cada época de entrenamiento.
      rect_loss: list
        Lista con pérdida de recostrucción de cada época.
      prio_loss: list
        Lista con la pérdida de prior matching almacenada en cada época.
      cons_loss: list
        Lista con la pérdida de consistencia de cada época.

    """
    model.train()
    train_loss = []
    rect_loss = []
    prio_loss = []
    cons_loss = []

    for epoch in tqdm(range(max_epochs)):
        tloss = 0
        rloss = 0
        ploss = 0
        closs = 0

        for i, (X, y) in enumerate(train_loader, 0):
            X = X.to(device)
            model.optimizer.zero_grad()

            if conditioned:
                num_chars = len(vocab)
                one_hot_vec = label_one_hot(vocab, num_chars, y)
                X_, mus, logvars = model(X, one_hot_vec)
            else:
                X_, mus, logvars = model(X)

            rect, prio, cons = model.loss(X, X_, mus, logvars, terms=True)

            a = prior_weight(epoch)
            b = warmup(epoch)
            loss = rect + prio*a + cons*b

            # Detiene ejecución si hay nan
            if torch.isnan(loss).sum().item() > 0:
                if torch.isnan(rect).sum().item() > 0:
                    print('Reconstruction NaN')
                if torch.isnan(prio).sum().item() > 0:
                    print('Prior NaN')
                if torch.isnan(cons).sum().item() > 0:
                    print('Consistency NaN')
                return train_loss, rect_loss, prio_loss, cons_loss

            # Calcula la pérdida promedio para cada tipo de pérdida
            rloss += rect.item() / X.shape[0]
            ploss += prio.item()
            closs += cons.item()
            tloss += loss.item() / X.shape[0]

            loss.backward()
            model.optimizer.step()

        # Cálculo de pérdidas promedio por época
        rloss /= len(train_loader)
        ploss /= len(train_loader)
        closs /= len(train_loader)
        tloss /= len(train_loader)

        # Almacenamiento de pérdidas por época
        train_loss.append(tloss)
        rect_loss.append(rloss)
        prio_loss.append(ploss)
        cons_loss.append(closs)

    return train_loss, rect_loss, prio_loss, cons_loss

def plot_loss(loss, rect_loss, prio_loss, cons_loss):
    """Función para graficar las pérdidas a lo largo de las épocas
       durante el entrenamiento.

    Parameters:
    --------------------------
      loss: list
        Lista de valores de pérdida total por época.
      rect_loss: list
        Lista de valores de pérdida de reconstrucción por época.
      prio_loss: list
        Lista de valores de pérdida de prior matching por época.
      cons_loss: list
        Lista de valores de pérdida de consistencia por época.

    Returns:
    --------------------------
      None
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.plot(loss)
    ax2.plot(rect_loss, color='red')
    ax3.plot(prio_loss, color='purple')
    ax4.plot(cons_loss, color='green')

    ax1.set_title('Train Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.set_title('Reconstruction Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    ax3.set_title('Prior Matching Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')

    ax4.set_title('Consistency Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')

    plt.show()
