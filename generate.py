import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib import gridspec

from .trainer import label_one_hot

device = 'cuda'

def get_prior(model, dataloader, device=device):
    """Calcula la media y la desviación estándar de la distribución a priori en el espacio latente.

    Parameters:
    --------------------------
      model: nn.Module
        Modelo para obtener la distribución a priori.
      dataloader: torch.utils.data.DataLoader
        DataLoader que proporciona los datos de entrenamiento.
      device: torch.device
        Dispositivo de cómputo (CPU o GPU).

    Returns:
    --------------------------
      mu_m: torch.Tensor
        Media de la distribución a priori en el espacio latente.
      mu_s: torch.Tensor
        Desviación estándar de la distribución a priori en el espacio latente.
      lv_m: torch.Tensor
        Media de los logaritmos de varianza de la distribución a priori en el espacio latente.
      lv_s: torch.Tensor
        Desviación estándar de los logaritmos de varianza de la distribución a priori en el espacio latente.
    """
    # Listas para almacenar las medias y logaritmos de varianza
    mus = []
    lvs = []

    # Obtención de las medias y logaritmos de la desviación estándar
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader, 0):
            X = X.to(device)
            _, mu, lv = model.encode(X)
            mus.append(mu[-1])
            lvs.append(lv[-1])

    # Concatena las listas en tensores
    mus = torch.cat(mus)
    lvs = torch.cat(lvs)

    # Calcula la media y la desviación estándar
    mu_m = mus.mean(axis=0)
    mu_s = mus.std(axis=0)
    lv_m = lvs.mean(axis=0)
    lv_s = lvs.std(axis=0)

    return mu_m, mu_s, lv_m, lv_s


def get_samples(n, model, priors, device=device):
    """Genera muestras a partir de la distribución a priori en el espacio latente.

    Parameters:
    --------------------------
      n: int
        Número de muestras a generar.
      model: nn.Module
        Modelo utilizado para la generación.
      priors: tuple of torch.Tensor
        Tupla que contiene la media y la desviación estándar de la distribución a priori.
      device: torch.device
        Dispositivo de cómputo (CPU o GPU).

    Returns:
    --------------------------
      generated_samples: torch.Tensor
        Tensor que contiene las muestras generadas a partir de la distribución a priori en el espacio latente.
    """
    mu_m, mu_s, lv_m, lv_s = priors # Obtiene las medias y distribuciones estándar de la distribución a priori

    # Genera muestras latentes usando distribuciones normales
    mu = mu_m + mu_s*torch.normal(0, 1, size=(n, mu_m.shape[0]), device=device)
    lv = lv_m + lv_s*torch.normal(0, 1, size=(n, lv_m.shape[0]), device=device)

    # Realiza muestreo en el espacio latente
    Z = model.sample_lattent(mu, lv)

    # Genera muestras a partir del espacio latente
    generated_samples = model.generate(Z)

    return generated_samples


def show_batch(images, CMAP='gray'):
    """Visualiza un batch de imágenes.

    Parameters:
    --------------------------
      images: torch.Tensor
        Tensor que contiene el batch de imágenes a mostrar.
      CMAP: str (Default: gray)
        Mapa de colores para la visualización de imágenes.

    Returns:
    --------------------------
      None

    """
    # Redimensiona las imágenes para obtener una vista de las imágenes en 2D
    images_ = images.view(images.shape[0], -1)
    # Calcula el número de filas y columnas para la cuadrícula
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn*2.5, 2*sqrtn))
    gs = gridspec.GridSpec(sqrtn*2, sqrtn//2)
    gs.update(wspace=0.05, hspace=0.05)

    # Itera sobre cada imagen y la muestra en un subgráfico de la cuadrícula
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        # Muestra la imagen en el subgráfico usando el mapa de colores especificado
        plt.imshow(img.permute(1, 2, 0), cmap=CMAP)

def plot_samples(n, model, dataloader, device=device):
    """Genera y visualiza muestras a partir de la distribución a priori en el espacio latente.

    Parameters:
    --------------------------
      n: int
        Número de muestras a generar.
      model: nn.Module
        Modelo utilizado para generar las muestras.
      dataloader: torch.utils.data.DataLoader
        DataLoader que contiene los datos para el cálculo de la distribución a priori.
      device: torch.device
        Dispositivo donde se realizarán los cálculos (por defecto: device).

    Returns:
    --------------------------
      None

    """
    # Obtiene los parámetros de la distribución a priori en el espacio latente
    priors = get_prior(model, dataloader, device=device)
    # Obtiene las muestras generadas utilizando los parámetros de la distribución a priori
    samples = get_samples(n, model, priors, device=device)
    samples = samples.cpu().detach()
    # Visualiza las muestras generadas
    show_batch(samples)

def plot_samples_by_normal(n, model, device=device):
    mu = torch.zeros((n, model.lattent_dim)).to(device)
    lv = torch.zeros((n, model.lattent_dim)).to(device)

    Z = model.sample_lattent(mu, lv)

    samples = model.generate(Z)
    samples = samples.cpu().detach()
    show_batch(samples)

def get_cond_samples(n, model, priors, label, vocab, device=device):
    """Genera muestras condicionales a partir de un modelo.

    Parameters:
    --------------------------
      n: int
        Número de muestras a generar.
      model: nn.Module
        Modelo utilizado para la generación.
      priors: tuple of torch.Tensor
        Tupla que contiene la media y la desviación estándar de la distribución a priori.
      label: str
        Etiqueta para condicionar las muestras.
      device: torch.device
        Dispositivo de cómputo (CPU o GPU).

    Returns:
    --------------------------
      samples_generated: torch.Tensor
        Tensor que contiene las muestras generadas condicionadas por la etiqueta.

    """
    mu_m, mu_s, lv_m, lv_s = priors # Obtiene las medias y distribuciones estándar de la distribución a priori

    # Genera muestras latentes usando distribuciones normales
    mu = mu_m + mu_s*torch.normal(0, 1, size=(n, mu_m.shape[0]), device=device)
    lv = lv_m + lv_s*torch.normal(0, 1, size=(n, lv_m.shape[0]), device=device)

    # Crea una representación one-hot de la etiqueta
    y = label_one_hot(vocab=vocab, num_classes=len(vocab), labels=label*n)

    # Realiza muestreo en el espacio latente
    Z = model.sample_lattent(mu, lv)

    # Condiciona las muestras latentes
    Z_cond = model.condition_on_label(Z, y)

    # Genera muestras a partir de las muestras latentes condicionadas
    samples_generated = model.generate(Z_cond)

    return samples_generated

def plot_cond_samples(n, model, dataloader, label, device=device):
    """Genera y visualiza muestras condicionadas a partir de la distribución a
       priori en el espacio latente.

    Parameters:
    --------------------------
      n: int
        Número de muestras a generar.
      model: nn.Module
        Modelo utilizado para generar las muestras.
      dataloader: torch.utils.data.DataLoader
        DataLoader que contiene los datos para el cálculo de la distribución a priori.
      label: str
        Etiqueta para condicionar las muestras.
      device: torch.device
        Dispositivo donde se realizarán los cálculos (por defecto: device).

    Returns:
    --------------------------
      None

    """
    # Obtiene los parámetros de la distribución a priori en el espacio latente
    priors = get_prior(model, dataloader, device=device)
    # Obtiene las muestras condicionadas generadas utilizando los parámetros de la distribución a priori
    samples = get_cond_samples(n, model, priors, label, device=device)
    samples = samples.cpu().detach()
    # Visualiza las muestras generadas
    show_batch(samples)