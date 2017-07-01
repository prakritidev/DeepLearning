import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset = None, label = None):
	pixel_depth = 255.0  
	if dataset is not None:
		dataset = (dataset.values.reshape((-1, image_size, image_size, num_channels)).astype(np.float32) - pixel_depth/2) / pixel_depth
  if label is not None:
    label = (np.arange(num_labels) == label.values[:,None]).astype(np.float32)
  return dataset, label




