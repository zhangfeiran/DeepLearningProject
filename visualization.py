import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image

colors = ['yellow', 'red', 'green', 'blue']

index_class_dict = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}

train_df = pd.read_csv('../input/train.csv')
train_df[f'target_vec'] = train_df['Target'].map(lambda x: list(map(int, x.strip().split())))


def make_rgb_image_from_four_channels(channels: list, image_width=512, image_height=512) -> np.ndarray:
    """
    It makes literally RGB image from source four channels, 
    where yellow image will be yellow color, red will be red and so on  
    """
    rgb_image = np.zeros(shape=(image_height, image_width, 3), dtype=np.float)
    yellow = np.array(Image.open(channels[0]))
    # yellow is red + green
    rgb_image[:, :, 0] += yellow / 2
    rgb_image[:, :, 1] += yellow / 2
    # loop for R,G and B channels
    for index, channel in enumerate(channels[1:]):
        current_image = Image.open(channel)
        rgb_image[:, :, index] += current_image
    # Normalize image
    rgb_image = rgb_image / rgb_image.max() * 255
    return rgb_image.astype(np.uint8)


a = 4
name = [f'../input/train/{train_df["Id"][a]}_{i}.png' for i in colors]
fig = make_rgb_image_from_four_channels(name)
print([index_class_dict[int(i)] for i in train_df['target_vec'][a]])
plt.imshow(fig)
plt.show()
