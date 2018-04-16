import numpy as np
from scipy.misc import imread, imresize

def load_image(filename, size=None):
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img

FACENET_MEAN = np.array([ 0.52591038,  0.40204082,  0.34178183], dtype=np.float32)
FACENET_STD = np.sqrt(np.array([3941.30175781, 2856.94287109, 2519.35791016], dtype=np.float32) / 255.**2)

def preprocess_image(img):
    """Preprocess an image for squeezenet.
    
    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32)/255.0 - FACENET_MEAN) / FACENET_STD

def load_image_batch(filenames, size=160):
    out = np.zeros((len(filenames),size,size,3))
    for i,name in enumerate(filenames):
        out[i,:,:] = preprocess_image(load_image(name, size=size))

    return out
