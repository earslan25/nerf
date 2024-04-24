from PIL import Image, ImageDraw, ImageFont
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

# ground_truth and prediction must have the same size
def save_result_comparison(ground_truth, prediction, path):
    ground_truth = Image.fromarray((ground_truth * 255).astype(np.uint8))
    prediction = Image.fromarray((prediction * 255).astype(np.uint8))
    
    height = ground_truth.height
    width = ground_truth.width + prediction.width

    new_img = Image.new('RGB', (width, height), 'white')
    
    new_img.paste(ground_truth, (0, 0))
    new_img.paste(prediction, (ground_truth.width, 0))

    new_img.save(path)

# TODO add more visualization functions like plotting accuracy, loss, etc.
