import matplotlib.pyplot as plt
import numpy as np


def plot_images(images):
    """
    Plot images
    :param images: images to plot, as a numpy array of shape (N, H, W, 3) or (H, W, 3)
    :return: None
    """
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    for i, image in enumerate(images):
        plt.figure()
        plt.imshow(image)
        plt.title(f'Image {i}')
        plt.show()

# TODO add more visualization functions like plotting accuracy, loss, etc.
