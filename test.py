#Feeding through placeholder
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
import numpy 
import argparse
import sys
import gzip
import os
#import tempfile
def xrange(x):
    return iter(range(x))


def _read32(bytestream):
	dt =numpy.dtype(numpy.uint32).newbyteorder('>')
	return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
	
def extractionimages(f):
	print('extracting',f.name)
	with gzip.GzipFile(fileobj = f) as bytestream:
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError('invalid magic number %d in Mnist image file: %s' %(magic, f.name))
		num_images =_read32(bytestream)
		rows= _read32(bytestream)
		cols= _read32(bytestream)
		buf =bytestream.read(rows * cols * num_images)
		data = numpy.frombuffer(buf,dtype=numpy.uint8)
		data = data.reshape(num_images, rows,cols,1)
		print('data extracted succesfuly')
		return data
    
def dense_to_one_hot(labels_dense,num_classes):
    num_labels =labels_dense.shape[0]
    index_offset=numpy.arange(num_labels)*num_classes
    labels_one_hot=numpy.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot 

                   
def extractionlabels(f,one_hot=False, num_classes=10):
    print('extracting lablels',f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('invalid magic number %d in mnist labels file: %s'%(magic,f.name))
        num_items=_read32(bytestream)
        buf=bytestream.read(num_items)
        labels = numpy.frombuffer(buf,dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels,num_classes)
        return labels
class DataSet(object):
    """Dataset class object."""

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=True):
        """Initialize the class."""
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype = dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  #local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
   #                                SOURCE_URL + TRAIN_IMAGES)
  f=open("C:/Users/ARH/Desktop/VST'S/train-images-idx3-ubyte_2.gz","rb")
  train_images = extractionimages(f)

  #local_file = base.maybe_download(TRAIN_LABELS, train_dir,
   #                                SOURCE_URL + TRAIN_LABELS)
  fL=open("C:/Users/ARH/Desktop/VST'S/train-labels-idx1-ubyte_2.gz","rb")
  train_labels = extractionlabels(fL, one_hot=one_hot)

  #local_file = base.maybe_download(TEST_IMAGES, train_dir,
   #                                SOURCE_URL + TEST_IMAGES)
  fT=open("C:/Users/ARH/Desktop/VST'S/t10k-images-idx3-ubyte_2.gz","rb")
  test_images = extractionimages(fT)

  #local_file = base.maybe_download(TEST_LABELS, train_dir,
   #                                SOURCE_URL + TEST_LABELS)
  fTL=open("C:/Users/ARH/Desktop/VST'S/t10k-labels-idx1-ubyte_2.gz","rb")
  test_labels = extractionlabels(fTL, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  
  options = dict(dtype=dtype, reshape=reshape, seed=random.seed)
  
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)
  
  return base.Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
    return read_data_sets(train_dir)



print('mnist starts ')
path=os.system('dir')
print(path)
def main(_):
  # Import data
  #FLAGS = None
  mnist =read_data_sets("C:/Users/ARH/Desktop/VST'S", one_hot=True)
  print('main')

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  print('Every thing is ok till now ')
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print('trainig')
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

  
    
"""__name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                              help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)   
   
   
main(_)"""

