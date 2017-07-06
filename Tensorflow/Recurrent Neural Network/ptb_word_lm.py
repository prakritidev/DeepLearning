import inspect
import time

import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
	def __init__(self, config, data, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)


class PTBModel(object):
	def __init__(self, is_training, config, input_):
		self._input = input_
		batch_size = input_.batch_size
		num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size
	def lstm_cell():
		

