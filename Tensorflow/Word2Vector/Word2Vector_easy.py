
'''

Vector space models (VSMs) represent (embed) words in a continuous vector space where
semantically similar words are mapped to nearby points ('are embedded nearby each other').
For further Understanding of VSM refer below link

https://en.wikipedia.org/wiki/Vector_space_model

************************************************************************************************************************
Word2Vector It comes in two flavour 

1. Skip Diagram:
Skip-gram treats each context-target pair as a new observation, and this tends to do better when we have larger datasets

2. CBOW: 
Smoothes over a lot of the distributional information (by treating an entire context as one observation). 
For the most part, this turns out to be a useful thing for smaller datasets

For better inderstanfing of there concpets:

See this Link3

https://iksinc.wordpress.com/2015/04/13/words-as-vectors/

************************************************************************************************************************

In this example code tensorflow used 
'''



import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf

# Dataset Download 
url = 'http://mattmahoney.net/dc/'

# function to download usinf python.
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

# Reading data

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
# vocabulary contain tokenized words.
vocabulary = read_data(filename)
print('Data size', len(vocabulary))
print('--------Content in vocabulary--------')
print(vocabulary[:10])
print()

# Building the dictonary, this will contain the unique words and replace rare words with UNK token
vocabulary_size = 50000 

# Building dataset and making suitable ofr further process 
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)

print('Most common words (+UNK)', count[:5])
print()
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
	
	global data_index
	assert batch_size % num_skips == 0  # Need help to understand this step
	assert num_skips <= 2 * skip_window # 
