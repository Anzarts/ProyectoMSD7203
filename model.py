import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision import transforms as T
from torchvision.ops import Permute
from torch.nn.utils import spectral_norm

def identity(module):
    """Función identidad que devuelve el mismo módulo sin modificarlo.

    Parameters:
    --------------------------
      module: torch.nn.Module
        Módulo de red neuronal de PyTorch.

    Returns:
    --------------------------
      module: torch.nn.Module
        Devuelve el mismo módulo de red neuronal sin realizar cambios.
    """
    return module

class InnerBlock(nn.Module):
    """Bloque interno configurable para construcción de capas de modelo HVAE.

    Parameters:
    --------------------------
      block_config: list
        Lista que describe la configuración del bloque, especificando cada capa con sus parámetros.
      size: tuple
        Tupla que indica el tamaño espacial de la entrada en formato (alto, ancho).
      in_c: int (Default: 1)
        Número de canales de entrada.
      layer_norm: func (Default: identity)
        Función que aplica normalización por capas.
      act: torch.nn.Module (Default: ReLU)
        Función de activación a utilizar en las capas convolucionales.

    """

    def __init__(self,
                 block_config,
                 size,
                 in_c=1,
                 layer_norm=identity,
                 act=nn.ReLU
                 ):
        super(InnerBlock, self).__init__()
        self.config = block_config

        H, W = size # Dimensiones de la entrada
        block = [] # Lista para almacenar los módulos del bloque
        skip_dims = {} # Diccionario para almacenar las dimensiones de las conexiones de skip

        # Itera sobre cada capa especificada en block_config
        for layer in block_config:
            if layer[0] == 'conv':
                # Agrega capa convolucional seguida de batch normalization y función de activación
                block.append(nn.Sequential(layer_norm(nn.Conv2d(in_c, layer[1], 3, padding=1)),
                                           act()))
                in_c = layer[1]
            elif layer[0] == 'conv-bn':
                # Agrega capa convolucional seguida de batch normalization y función de activación
                block.append(nn.Sequential(layer_norm(nn.Conv2d(in_c, layer[1], 3, padding=1)),
                                           nn.BatchNorm2d(layer[1]),
                                           act()))
                in_c = layer[1]
            elif layer[0] == 'convt':
                # Agrega capa convolucional transpuesta seguida de una función de activación
                block.append(nn.Sequential(layer_norm(nn.ConvTranspose2d(in_c, layer[1], 3, padding=1)),
                                           act()))
                in_c = layer[1]
            elif layer[0] == 'convt-bn':
                # Agrega capa convolucional transpuesta seguida de batch normalization y función de activación
                block.append(nn.Sequential(layer_norm(nn.ConvTranspose2d(in_c, layer[1], 3, padding=1)),
                                           nn.BatchNorm2d(layer[1]),
                                           act()))
                in_c = layer[1]
            elif layer[0] == 'pool':
                # Agrega capa de max pooling para reducir las dimensiones espaciales
                block.append(nn.MaxPool2d(kernel_size=(layer[1], layer[1])))
                H = H // layer[1]
                W = W // layer[1]
            elif layer[0] == 'upsample':
                # Agrega capa de upsample para incrementar las dimensiones espaciales
                block.append(nn.Upsample(scale_factor=(layer[1], layer[1])))
                H = H * layer[1]
                W = W * layer[1]
            elif layer[0] == 'skip-up':
                # Realiza un cambio de dimensiones en el tensor para su uso futuro
                block.append(Permute((0, 2, 3, 1)))
                skip_dims[layer[1]] = in_c # Almacena la información relevante para el caso de skip-down
            elif layer[0] == 'skip-down':
                # Calcula las dimensiones para fusionar los saltos con el tensor actual
                skip_dim = skip_dims[layer[1]] + in_c
                # Añade las operaciones requeridas para fusionar los skip-down con el tensor actual
                block.append(nn.Sequential(nn.Linear(skip_dim, in_c),
                                           Permute((0, 3, 1, 2))))

        # Almacena los módulos construidos como una lista de módulos
        self.block = nn.ModuleList(block)
        self.output_size = (H, W)
        self.output_channels = in_c
        self.n_layers = len(self.config)


    def forward(self, x):
        """Método que procesa la entrada a través de las capas definidas en el bloque.

        Parameters:
        --------------------------
          x : torch.Tensor
            Tensor de entrada que se procesará a través de las capas del bloque.

        Returns:
        --------------------------
          x : torch.Tensor
            Tensor resultante después de procesar todas las capas del bloque.
        """
        # Diccionario para almacenar las activaciones de las capas 'skip-up'
        skip_x = {}

        # Itera sobre las capas definidas en el bloque
        for l in range(self.n_layers):
            if self.config[l][0] == 'skip-up':
                # Almacena la salida de la capa en el diccionario 'skip_x' para su uso posterior
                skip_x[self.config[l][1]] = self.block[l](x)
            elif self.config[l][0] == 'skip-down':
                # Concatena y permuta los tensores para preparar la entrada a la capa actual
                x = torch.cat([torch.permute(x, (0, 2, 3, 1)),
                               skip_x[self.config[l][1]]], -1)
                # Pasa la entrada a través de la capa actual
                x = self.block[l](x)
            else:
                # Pasa la entrada a través de la capa actual si no es una capa 'skip-up' o 'skip-down'
                x = self.block[l](x)

        return x