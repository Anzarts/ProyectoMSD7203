import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CaptchaDataset(Dataset):
    """Clase que define Dataset para manejar imágenes de captcha."""

    def __init__(self, root, transforms):
        """Inicializa el Dataset Captcha.

        Carga las imágenes de captcha desde el directorio dado y establece las
        transformaciones.

        Parameters:
        --------------------------
            root: str
              Ruta al directorio que contiene las imágenes de captcha.
            transforms: callable
              Transformaciones a aplicar a las imágenes cargadas.

        """
        self.root = root
        self.transform = transforms
        self.images = os.listdir(root)


    def __len__(self):
        """Retorna la cantidad total de imágenes en el dataset.

        Returns:
        --------------------------
            int: Número total de imágenes en el dataset.
        """
        return len(self.images)


    def __getitem__(self, idx):
        """Obtiene una imagen y su etiqueta asociada según el índice especificado.

        Parameters:
        --------------------------
            idx: int
              Índice de la imagen que se va a cargar.

        Returns:
        --------------------------
            image: PIL.Image.Image
              Imagen obtenida de acuerdo al índice.
            label: str
              String con la etiqueta asociada a la imagen.

        """
        image_path = os.path.join(self.root, self.images[idx])

        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image)

        label = self.images[idx][:-5][-6:]

        return image, label.lower()