# Import TensorFlow and enable eager execution
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

# Import NumPy
import numpy as np

# Download MNIST dataset (try also Fashion MNIST!)
(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()

# Cast targets to np.int32
ytrain = ytrain.astype(np.int32)
ytest = ytest.astype(np.int32)

# Example of tf.data
train_it = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
test_it = tf.data.Dataset.from_tensor_slices((Xtest, ytest))

# Get a single image
for xb, yb in train_it.batch(1):
  break

# Show the image
import matplotlib.pyplot as plt
#plt.imshow(xb[0])
#plt.show()

# Check shape and label
print(xb.shape)
print(yb)

# Definitely not the most efficient way!
def _parse_example(x, y):
  x = tf.cast(tf.reshape(x, (28, 28, 1)), tf.float32) / tf.constant(255.0)
  return x, y

# Map all examples with the function
# Check documentation of Dataset.map here: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
train_it = train_it.map(_parse_example)
test_it = test_it.map(_parse_example)

# Architecture is taken from here: https://www.tensorflow.org/tutorials/estimators/cnn#building_the_cnn_mnist_classifier

class CNN(tf.keras.Model):
  
  def __init__(self):
    super(CNN, self).__init__()
    
    self.conv1 = tf.layers.Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    self.pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
    
    
    self.conv2 = tf.layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    self.pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)


    self.dense = tf.layers.Dense(1024, activation=tf.nn.relu)
    self.dropout = tf.layers.Dropout(0.4)
    
    self.logits = tf.layers.Dense(units=10)
    
  def call(self, x, training=False):
    
    x = self.pool1(self.conv1(x))
    x = self.pool2(self.conv2(x))
    x = self.dense(tf.reshape(x, (-1, 7 * 7 * 64)))
    x = self.dropout(x, training=training)
    return self.logits(x)

#same model of the paper
class SpectroCNN(tf.keras.Model):
  
  def __init__(self):
    super(SpectroCNN, self).__init__()
    
    self.conv1 = tf.layers.Conv2D(filters=24, kernel_size=[6, 6], strides = (1,1), padding="same", activation=tf.nn.relu)
    self.conv2 = tf.layers.Conv2D(filters=24, kernel_size=[6, 6], strides = (1,1), padding="same", activation=tf.nn.relu)
    self.conv3 = tf.layers.Conv2D(filters=48, kernel_size=[5, 5], strides = (2,2), padding="same", activation=tf.nn.relu)
    self.conv4 = tf.layers.Conv2D(filters=48, kernel_size=[5, 5], strides = (2,2), padding="same", activation=tf.nn.relu)
    self.conv5 = tf.layers.Conv2D(filters=64, kernel_size=[4, 4], strides = (2,2), padding="same", activation=tf.nn.relu)

    self.dense = tf.layers.Dense(1024, activation=tf.nn.relu)
    self.dropout = tf.layers.Dropout(0.5) #to be improved
    
    self.logits = tf.layers.Dense(units=10)
    
  def call(self, x, training=False):
    
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)

    x = self.dense(tf.reshape(x, (-1, 1024)))
    x = self.dropout(x, training=training)

    return self.logits(x)

# Initialize
#cnn = CNN()
cnn = SpectroCNN()

for xb, yb in train_it.batch(1):
  print(cnn(xb))
  break

def loss(net, x, y):
  return tf.losses.sparse_softmax_cross_entropy(logits=net(x, training=True), labels=y)

opt = tf.train.AdamOptimizer()
epochs = 10

all_acc = np.zeros(epochs)

for epoch in range(epochs):
  
  acc = tfe.metrics.SparseAccuracy()
  for xb, yb in test_it.batch(32):
    #print(xb,yb)
    ypred = cnn(xb)
    acc(predictions=ypred, labels=yb)
    
  all_acc[epoch] = acc.result().numpy()
  print('Test accuracy at epoch {} is {} %'.format(epoch, all_acc[epoch] * 100))
  
  for xb, yb in train_it.shuffle(1000).batch(32):
    opt.minimize(lambda: loss(cnn, xb, yb))

plt.plot(all_acc)