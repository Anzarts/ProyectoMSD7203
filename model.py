import numpy as np
import matplotlib.pyplot as plt
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

# Configuración de las capas de entrada y salida pre-definida
outer_config = [('conv-bn', 32), ('pool', 2), ('conv-bn', 64), ('pool', 2), ('conv-bn', 64)]

# Configuración de las capas internas pre-definida
inner_config = [[('conv-bn', 32), ('skip-up', 0), ('pool', 2),
                 ('conv-bn', 64), ('skip-up', 1), ('pool', 2),
                 ('conv-bn', 64), ('conv-bn', 64),
                 ('upsample', 2), ('skip-down', 1), ('conv-bn', 32),
                 ('upsample', 2), ('skip-down', 0), ('conv-bn', 1)]]

class HVAE(nn.Module):
    """Clase de modelo de Hierarchical Variational Autoencoder.

    Parameters:
    --------------------------
      name: str
        Nombre del objeto.
      input_channels: int
        Número de canales del input.
      input_size: (int, int)
        Dimensiones de las imágenes de entrada (50 pixeles de altura y 250 pixeles de ancho).
      inner_size: (int, int)
        Dimensiones de las capas internas.
      lattent_dim: int
        Dimensión del espacio latente.
      outer_config: [(str, int)]
        Configuración que capas de entrada y salida.
      inner_config: [[(str, int)]]
        Configuración de capas internas.
      lr: float
        Learning rate.
      activation: nn.Module
        Función de activación.
      encoder_norm: callable[nn.Module] -> nn.Module
        Normalización de capas del encoder.
    """

    def __init__(self,
                 name='HVAE',
                 input_channels=3,
                 input_size=(50, 250),
                 inner_size=(24, 24),
                 lattent_dim=512,
                 outer_config=outer_config,
                 inner_config=inner_config,
                 lr=5e-5,
                 activation=nn.ReLU,
                 encoder_norm=spectral_norm,
                 conditioned=False,
                 vocab={char:index for index, char in enumerate("qwertyuiopasdfghjklzxcvbnm26")}
                 ):
        super(HVAE, self).__init__()

        self.name = name
        self.conditioned = conditioned

        self.outer_config = outer_config
        self.inner_config = inner_config

        self.act = activation

        # Asignación de parámetros de entrada
        self.input_channels = input_channels
        self.input_size = input_size
        self.inner_size = inner_size
        self.lattents = len(inner_config) + 1
        self.lattent_dim = lattent_dim
        self.encoder_norm = encoder_norm

        # Construcción del encoder y decoder
        self.build_encoder()
        self.build_decoder()

        # Capas lineales para la media y varianza de la distribución en el espacio latente
        self.mu_layers = [self.encoder_norm(nn.Linear(2*lattent_dim, lattent_dim)) for l in range(self.lattents)]
        self.mu_layers += [nn.Linear(2*lattent_dim, lattent_dim) for l in range(self.lattents-1)]
        self.mu_layers = nn.ModuleList(self.mu_layers)

        self.lv_layers = [self.encoder_norm(nn.Linear(2*lattent_dim, lattent_dim)) for l in range(self.lattents)]
        self.lv_layers += [nn.Linear(2*lattent_dim, lattent_dim) for l in range(self.lattents-1)]
        self.lv_layers = nn.ModuleList(self.lv_layers)

        if conditioned:
            self.num_classes = len(vocab)
            self.projector_label = nn.Sequential(nn.Linear(self.num_classes, self.lattent_dim), nn.ReLU())

        # OPTIMIZER ----------------------------------------------
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.parameters(), lr)

    def build_encoder(self):
        """Construye las capas del encoder según la configuración dada.

        Esta función construye las capas convolucionales y lineales
        para el proceso de codificación de los datos de entrada, utilizando
        la configuración definida en `self.config`. Cada bloque en la
        configuración representa un nivel en el encoder.

        Parameters:
        --------------------------
          None

        Returns:
        --------------------------
          None: Las capas del encoder se almacenan en `self.encoder`
                como una lista de módulos de PyTorch.

        """
        self.encoder = []

        in_c = self.input_channels  # Número de canales de entrada
        H, W = self.input_size # Dimensiones de entrada

        block = self.outer_config # Configuración de las capas del encoder
        enc_block = [] # Lista para almacenar las capas del bloque actual

        # Iteración sobre cada capa especificada por outer_config
        for layer in block:
            if layer[0] == 'conv':
                # Agrega capa convolucional seguida de una función de activación
                enc_block.append(self.encoder_norm(nn.Conv2d(in_c, layer[1], 3, padding=1)))
                enc_block.append(self.act())
                in_c = layer[1]
            elif layer[0] == 'conv-bn':
                # Agrega capa convolucional seguida de batch normalization y función de activación
                enc_block.append(self.encoder_norm(nn.Conv2d(in_c, layer[1], 3, padding=1)))
                enc_block.append(nn.BatchNorm2d(layer[1]))
                enc_block.append(self.act())
                in_c = layer[1]
            elif layer[0] == 'pool':
                # Agrega capa de max pooling para reducir las dimensiones espaciales
                enc_block.append(nn.MaxPool2d(kernel_size=(layer[1], layer[1])))
                # Actualiza las dimensiones de entrada para la próxima capa
                H = H // layer[1]
                W = W // layer[1]

        # Capas lineales al final del bloque para las dimensiones latentes
        output_fc = nn.Sequential(nn.Flatten(),
                                  self.encoder_norm(nn.Linear(in_c*H*W, 2*self.lattent_dim)),
                                  self.act(),
                                  self.encoder_norm(nn.Linear(2*self.lattent_dim, 2*self.lattent_dim)),
                                  self.act())

        enc_block.append(output_fc) # Agrega las capas lineales al bloque actual

        self.encoder.append(nn.Sequential(*enc_block)) # Agrega el bloque al encoder

        # Iteración sobre cada capa especificada por inner_config
        for block in self.inner_config:
            H, W = self.inner_size

            # Capas iniciales del bloque
            input_fc = nn.Sequential(self.encoder_norm(nn.Linear(self.lattent_dim, self.lattent_dim*2)),
                                     self.act(),
                                     self.encoder_norm(nn.Linear(self.lattent_dim*2, H*W)),
                                     self.act(),
                                     nn.Unflatten(dim=-1, unflattened_size=(1, H, W)))

            enc_block = [input_fc] # Lista para almacenar las capas del bloque

            # Agrega el bloque interno a la lista de capas
            inner_block = InnerBlock(block, (H, W), layer_norm=self.encoder_norm, act=self.act)
            in_c = inner_block.output_channels
            H, W = inner_block.output_size
            enc_block.append(inner_block)

            # Capas lineales al final del bloque para las dimensiones latentes
            output_fc = nn.Sequential(nn.Flatten(),
                                      self.encoder_norm(nn.Linear(in_c*H*W, 2*self.lattent_dim)),
                                      self.act(),
                                      self.encoder_norm(nn.Linear(2*self.lattent_dim, 2*self.lattent_dim)),
                                      self.act())

            enc_block.append(output_fc) # Agrega las capas finales

            self.encoder.append(nn.Sequential(*enc_block)) # Agrega el bloque interno al encoder

        self.encoder = nn.ModuleList(self.encoder)

    def build_decoder(self):
        """Construye las capas del decoder según la configuración dada.

        Este método construye las capas del decoder utilizando la configuración
        de capas invertida definida en `self.config`. Cada bloque de capas se
        construye en orden inverso, generando las capas necesarias para decodificar
        desde las dimensiones latentes hasta la salida original.

        Parameters:
        --------------------------
          None

        Returns:
        --------------------------
          None: Las capas del decoder se almacenan en `self.decoder`
            como una lista de módulos de PyTorch.

        """
        rev_config = self.inner_config[::-1]

        self.decoder = []

        # Construcción de los bloques internos del decoder
        for block in rev_config:
            H, W = self.inner_size

            # Capa lineal de entrada al bloque del decoder
            input_fc = nn.Sequential(self.encoder_norm(nn.Linear(self.lattent_dim, self.lattent_dim*2)),
                                     self.act(),
                                     self.encoder_norm(nn.Linear(self.lattent_dim*2, H*W)),
                                     self.act(),
                                     nn.Unflatten(dim=-1, unflattened_size=(1, H, W)))

            dec_block = [input_fc]

            # Agrega el bloque interno
            inner_block = InnerBlock(block, (H, W), act=self.act)
            in_c = inner_block.output_channels
            H, W = inner_block.output_size
            dec_block.append(inner_block)

            # Capas lineales al final del bloque para las dimensiones latentes
            output_fc = nn.Sequential(nn.Flatten(),
                                      self.encoder_norm(nn.Linear(in_c*H*W, 2*self.lattent_dim)),
                                      self.act(),
                                      self.encoder_norm(nn.Linear(2*self.lattent_dim, 2*self.lattent_dim)),
                                      self.act())

            dec_block.append(output_fc)

            self.decoder.append(nn.Sequential(*dec_block)) # Agrega el bloque interno al decoder

        # Reconstruction block

        in_c = self.input_channels
        H, W = self.input_size

        block = self.outer_config
        dec_block = []

        last_conv = True
        last_upsample = True

        # Construcción inversa de las capas del bloque de reconstrucción
        for layer in block:
            if layer[0] == 'conv':
                if last_conv:
                    # Agrega una última capa convolucional seguida de función de activación
                    dec_block = [nn.ConvTranspose2d(layer[1], in_c, 3, padding=1),
                                 nn.Sigmoid()] + dec_block
                    last_conv = False
                else:
                    # Agrega capa convoluional seguida de una función de activación
                    dec_block = [nn.ConvTranspose2d(layer[1], in_c, 3, padding=1),
                                 self.act()] + dec_block
                in_c = layer[1]
            elif layer[0] == 'conv-bn':
                if last_conv:
                    # Agrega una última capa convolucional seguida de batch normalization y función de activación
                    dec_block = [nn.ConvTranspose2d(layer[1], in_c, 3, padding=1),
                                 nn.Sigmoid()] + dec_block
                    last_conv = False
                else:
                    # Agrega capa convolucional seguida de batch normalization y función de activación
                    dec_block = [nn.ConvTranspose2d(layer[1], in_c, 3, padding=1),
                                 nn.BatchNorm2d(in_c),
                                 self.act()] + dec_block
                in_c = layer[1]
            elif layer[0] == 'pool':
                if last_upsample:
                    # Agrega una última capa de upsample para incrementar las dimensiones espaciales
                    dec_block = [nn.Upsample(size=self.input_size)] + dec_block
                    last_upsample = False
                else:
                    # Agrega capa de upsample para incrementar las dimensiones espaciales
                    dec_block = [nn.Upsample(scale_factor=(layer[1], layer[1]))] + dec_block
                # Actualiza las dimensiones de entrada para la próxima capa
                H = H // layer[1]
                W = W // layer[1]

        # Agrega las capas lineales para el inicio del bloque de reconstrucción
        input_fc = nn.Sequential(nn.Linear(self.lattent_dim, 2*self.lattent_dim),
                                 self.act(),
                                 nn.Linear(2*self.lattent_dim, in_c*H*W),
                                 self.act(),
                                 nn.Unflatten(dim=-1, unflattened_size=(in_c, H, W)))

        dec_block = [input_fc] + dec_block

        self.decoder.append(nn.Sequential(*dec_block)) # Agrega el bloque de reconstrucción al decoder

        self.decoder = nn.ModuleList(self.decoder)

    def sample_lattent(self, mu, logvar):
        """ Realiza el muestreo en el espacio latente utilizando la técnica de reparametrización.

        Esta función toma la media (mu) y el logaritmo de la varianza (logvar)
        calculadas por el encoder y utiliza la técnica de reparametrización para
        muestrear un vector latente z.

        Parameters:
        --------------------------
          mu: torch.Tensor
            Tensor de medias calculadas por el encoder.
          logvar: torch.Tensor
            Tensor de logaritmo de varianzas calculado por el encoder.

        Returns:
        --------------------------
          z: torch.Tensor
            Tensor resultante del muestreo en el espacio latente.

        """
        eps = torch.normal(0, 1, size=mu.shape, device=mu.device)
        z = mu + torch.exp(0.5*logvar) * eps # Aplica el reparametrization trick

        return z

    def encode(self, x):
        """Codifica la entrada x en el espacio latente utilizando el encoder definido.

        Este método toma la entrada x y pasa a través del encoder, generando las
        medias (mu) y los logaritmos de varianza (lv) para cada nivel latente.

        Parameters:
        --------------------------
          x: torch.Tensor
            Tensor de entrada para ser codificado en el espacio latente.

        Returns:
        --------------------------
          x: torch.Tensor
            Tensor que representa la salida final del modelo en el espacio latente.
          mus: list
            Lista de tensores que representan las medias (mu) para cada nivel latente.
          lvs: list
            Lista de tensores que representan los logaritmos de varianza (lv) para cada nivel latente.

        """
        mus = [] # Lista para almacenar las medias de cada nivel latente
        lvs = [] # Lista para almacenar los logaritmos de varianza de cada nivel latente

        # Itera sobre cada nivel latente en el modelo
        for l in range(self.lattents):
            x = self.encoder[l](x)
            mu = self.mu_layers[l](x)
            lv = self.lv_layers[l](x)

            mus.append(mu)
            lvs.append(lv)

            # Realiza muestreo en el espacio latente con la media y varianza actuales
            x = self.sample_lattent(mu, lv)

        return x, mus, lvs

    def decode(self, x):
        """Decodifica la entrada latente x para obtener la salida reconstruida.

        Este método toma la entrada latente x y la decodifica a través del decoder,
        generando las medias (mu) y los logaritmos de varianza (lv) para cada
        nivel latente. Utiliza la función `sample_lattent` para muestrear un vector
        latente en cada nivel.

        Parameters:
        --------------------------
          x: torch.Tensor
            Tensor de entrada latente a ser decodificado para generar la salida.

        Returns:
        --------------------------
          x: torch.Tensor
            Tensor resultante de la decodificación, que representa la salida reconstruida.
          mus: list
            Lista de tensores que representan las medias (mu) para cada nivel latente.
          lvs: list
            Lista de tensores que representan los logaritmos de varianza (lv) para cada nivel latente.

        """
        mus = [] # Lista para almacenar las medias de cada nivel latente
        lvs = [] # Lista para almacenar los logaritmos de varianza de cada nivel latente

        # Iteración a través de los niveles latentes, excepto el último
        for l in range(self.lattents-1):
            x = self.decoder[l](x)
            mu = self.mu_layers[self.lattents+l](x)
            lv = self.lv_layers[self.lattents+l](x)

            mus.append(mu)
            lvs.append(lv)

            # Realiza muestreo en el espacio latente con la media y varianza actuales
            x = self.sample_lattent(mu, lv)

        # Decodifica la entrada latente final para obtener la salida reconstruida
        x = self.decoder[-1](x)

        return x, mus, lvs

    def forward(self, x, y=None):
        """Realiza la propagación hacia adelante a través del modelo.

        Este método realiza la propagación hacia adelante del modelo, codificando
        la entrada x en el espacio latente utilizando el método `encode` y luego
        decodificando la representación latente obtenida utilizando el método `decode`.
        Devuelve la salida reconstruida, así como las listas combinadas de medias
        y logaritmos de varianza para la codificación y decodificación.

        Parameters:
        --------------------------
          x: torch.Tensor
            Tensor de entrada para el proceso de codificación y decodificación.
          y: torch.Tensor | None
            Tensor con etiqueta en codificación one-hot.

        Returns:
        --------------------------
          x: torch.Tensor
            Tensor resultante de la decodificación, que representa la salida reconstruida.
          mus_inf+mus_gen: list
            Lista combinada de tensores que representan las medias para cada nivel latente,
              tanto de la codificación como de la decodificación.
          lvs_inf+lvs_gen: list
            Lista combinada de tensores que representan los logaritmos de varianza para cada
            nivel latente, tanto de la codificación como de la decodificación.

        """
        # Codificación
        z, mus_inf, lvs_inf = self.encode(x)

        if y is not None:
            if self.conditioned:
                z = self.condition_on_label(z, y)

        # Muestreo y decodificación
        x, mus_gen, lvs_gen = self.decode(z)

        return x, mus_inf+mus_gen, lvs_inf+lvs_gen

    def generate(self, z):
        """Genera una salida a partir de una entrada latente z.

        Este método genera una salida a partir de una entrada latente z utilizando
        el decoder. Itera a través de los niveles latentes del decoder para decodificar
        la entrada latente y obtener la salida generada correspondiente.

        Parameters:
        --------------------------
          z: torch.Tensor
            Tensor de entrada latente para la generación de la salida.

        Returns:
        --------------------------
          x: torch.Tensor
            Tensor resultante de la generación, que representa la salida generada.

        """
        # Iteración a través de los niveles latentes, excepto el último
        for l in range(self.lattents-1):
            z = self.decoder[l](z)
            mu = self.mu_layers[self.lattents+l](z)
            lv = self.lv_layers[self.lattents+l](z)

            # Realiza muestreo en el espacio latente con la media y varianza actuales
            z = self.sample_lattent(mu, lv)

        # Decodificación final para obtener la salida generada
        x = self.decoder[-1](z)

        return x

    def loss(self, x, x_, mus, logvars, terms=False):
        """Calcula la pérdida del modelo.

        Este método calcula la pérdida total del modelo HVAE utilizando los valores
        de reconstrucción y términos de regularización para las medias y logaritmos
        de varianza proporcionados.

        Parameters:
        --------------------------
          x: torch.Tensor
            Tensor de entrada original.
          x_: torch.Tensor
            Tensor de salida reconstruido.
          mus: list
            Lista de tensores que representan las medias para cada nivel latente.
          logvars: list
            Lista de tensores que representan los logaritmos de varianza para cada nivel latente.
          terms: bool (opcional)(Default:False)
            Booleano que indica si se desean los términos de la pérdida por separado.

        Returns:
        --------------------------
          torch.Tensor or tuple: Si terms es True, devuelve una tupla con los términos de la pérdida por separado
            (reconstrucción, coincidencia previa, consistencia). Si terms es False, devuelve la pérdida total combinada.

        """
        mu_T = mus[self.lattents-1] # Medias del nivel latente superior
        logvar_T = logvars[self.lattents-1] # Logaritmos de varianza del nivel latente superior

        mu_inf = mus[:self.lattents-1] # Medias de los niveles latentes de codificación
        logvar_inf = logvars[:self.lattents-1] # Logaritmos de varianza de los niveles latentes de codificación

        mu_gen = mus[self.lattents:][::-1] # Medias de los niveles latentes de generación, invertidos
        logvar_gen = logvars[self.lattents:][::-1] # Logaritmos de varianza de los niveles latentes de generación, invertidos

        # Cálculo de la pérdida de reconstrucción
        reconstruction = F.binary_cross_entropy(x_, x, reduction='sum')

        # Cálculo de divergencia KL
        prior_matching = -0.5*(1 + logvar_T - mu_T**2 - torch.exp(logvar_T))
        prior_matching = prior_matching.sum(axis=-1).mean()

        consistency = 0
        if self.lattents > 1:
            # Cálculo del término de consistencia si hay más de un nivel latente
            for l in range(self.lattents-1):
                D = 1 + logvar_inf[l] - logvar_gen[l]
                D = D - (mu_gen[l] - mu_inf[l])**2 / torch.exp(logvar_gen[l])
                D = D - torch.exp(logvar_inf[l] - logvar_gen[l])

                consistency += (-0.5*D).sum(axis=-1).mean()

            consistency = consistency
        else:
            # Si solo hay un nivel latente, la consistencia es 0
            consistency = consistency * prior_matching

        if terms:
            return reconstruction, prior_matching, consistency

        loss = reconstruction + prior_matching + consistency

        return loss

    def condition_on_label(self, z, y):
        """Realiza la condicionamiento de un vector latente 'z' con una etiqueta 'y'.

        Parameters:
        --------------------------
          z: torch.Tensor
            Tensor del vector latente.
          y: torch.Tensor
            Tensor que contiene la información de la etiqueta.

        Returns:
        --------------------------
          torch.Tensor
            Tensor resultante de la suma entre el vector latente 'z' y la proyección de la etiqueta 'y'.
        """
        # Proyección de la etiqueta mediante una capa lineal
        projected_label = self.projector_label(y)
        # Suma entre el vector latente y la proyección de la etiqueta
        latent_cond = z + projected_label

        return latent_cond

    def save(self, savedir):
         """Guarda los parámetros entrenados del modelo.
         Parameters:
         --------------------------
          savedir: str
            Ruta donde se guardará el archivo con los parámetros del modelo.
         """
         torch.save(self.state_dict(), savedir + self.name + '.pt')

    def load(self, savedir):
        """Carga los parámetros entrenados del modelo desde un archivo.

        Parameters:
        --------------------------
          savedir: str
            Ruta donde se encuentra el archivo con los parámetros del modelo.
        """
        # Carga los parámetros entrenados del modelo desde un archivo
        self.load_state_dict(torch.load(savedir))