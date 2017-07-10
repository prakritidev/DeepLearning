
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

See this Link- > https://iksinc.wordpress.com/2015/04/13/words-as-vectors/
See this Link- > http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

************************************************************************************************************************

In this example code tensorflow used Skip Diagram. 

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

# <----------------------------------------------------- Download the file from internet ----------------------------------------------------->
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

# <---------------------------------------------------- Reading data from the file ----------------------------------------------------------->

def read_data(filename):

  """Extract the first file enclosed in a zip file as a list of words."""

  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

# <--------------------------------------------------- Building Vocab by tokenization ------------------------------------------------------->

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
# before go through this code you should know how skip-gram model works and the terminologies. 
# Step 3: Function to generate a training batch for the skip-gram model.

# <--------------------------------------------------- Building Vocab by tokenization ------------------------------------------------------->
''' 
function explanation by : Anuj Gupta
-> https://github.com/anujgupta82
'''
num_skips=2 # no of words to be picked from the window
skip_window=1 #define how much will we see on one side of the word
batch_size = 16 

data_index = 0 #global circular counter over the data

def generate_batch(batch_size, num_skips, skip_window):
    # skip window is the amount of words we're looking at from each side of a given word
    # creates a single batch
    
    global data_index

    # num_skips => # of times we select a random word within the span? so no of picks should be integer
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window # maximum no of samples picked is size of span

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    # e.g if skip_window = 2 then span = 5
    # span is the length of the whole frame we are considering for a single word (left + word + right)
    # skip_window is the length of one side
    
    span = 2 * skip_window + 1 # [ span defines the whole window, which is 2 * skip_window + the word itself ]
    
    # queue which add and pop at the end
    buffer = collections.deque(maxlen=span)
    
    #print "span = %d" %span
    
    #get words starting from index 0 to span
    for _ in range(span):
        #print "_ = %d" %_
        #print "data_index = %d" %data_index
        buffer.append(data[data_index]) # fill the buffer with elements in window
        data_index = (data_index + 1) % len(data)  #this is just to circle at the end of text corpus


    # num_skips => # of times we select a random word within the span
    # batch_size (8) and num_skips (2) (4 times)
    # batch_size (8) and num_skips (1) (8 times)
    
    #denotes the number of (input, output) pairs generated from the single window: [skip_window target skip_window]. 
    #So num_skips restrict the number of context words we would use as output words.
    
    #since num_skips = # of elements picked in each window, 
    # of windows = batch_size // num_skips
    # we iterate - for each window (i)
    #                   for each pick in given window
    #                            fit the pick in the batch
    
    # to fit the pick in the batch : jth element in ith pick = i * num_skips + j
    
    # from each window, we pick #num_skips elemnts, so to make a batch, how many windows we need ?
    num_of_windows = batch_size // num_skips 
    
    for i in range(num_of_windows):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ] # we only need to know the words around a given word, not the word itself
        
        for j in range(num_skips):
            while target in targets_to_avoid:
                # find a target word that is not the word itself
                # while loop will keep repeating until the algorithm find a suitable target word
                target = random.randint(0, span - 1)
                
            # add selected target to avoid_list for next time
            targets_to_avoid.append(target)
            
            # e.g. i=0, j=0 => 0; i=0,j=1 => 1; i=1,j=0 => 2
            batch[i * num_skips + j] = buffer[skip_window] # [skip_window] => middle element
            labels[i * num_skips + j, 0] = buffer[target]
            
        #populate the buffer with elements of next window - which is one elemnt on right
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# <--------------------------------------------------- Building and traning skip-model ------------------------------------------------------->

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


graph = tf.Graph()


with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')