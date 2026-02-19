from PIL import Image
import numpy as np
import io

def preprocess_image(image_data, target_size=(64, 64)):
    if isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, str):
        img = Image.open(image_data)
    else:
        img = image_data

    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    img_gray = img_resized.convert('L')

    img_array = np.array(img_gray)
    img_array = np.expand_dims(img_array, axis=-1)   # (64, 64, 1)
    img_array = np.expand_dims(img_array, axis=0)    # (1, 64, 64, 1)

    return img_array