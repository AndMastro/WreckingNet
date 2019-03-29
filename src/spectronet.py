import tensorflow as tf

class SpectroCNN(tf.keras.Model):

    def __init__(self):
        super(SpectroCNN, self).__init__()

        self.conv1 = tf.layers.Conv2D(filters=24,
                                      kernel_size=[6, 2],
                                      #kernel_size=[6, 6],
                                      strides=(1, 1),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(filters=24,
                                      kernel_size=[6, 2],
                                      #kernel_size=[6, 6],
                                      strides=(1, 1),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.Conv2D(filters=48,
                                      kernel_size=[5, 1],
                                      #kernel_size=[5, 5],
                                      strides=(2, 2),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv4 = tf.layers.Conv2D(filters=48,
                                      kernel_size=[5, 1],
                                      #kernel_size=[5, 5],
                                      strides=(2, 2),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv5 = tf.layers.Conv2D(filters=64,
                                      kernel_size=[4, 1],
                                      #kernel_size=[4, 4],
                                      strides=(2, 2),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)

        self.dense = tf.layers.Dense(200, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.3)  # to be improved

        # self.logits = tf.layers.Dense(units=4, activation=tf.nn.softmax)
        self.logits = tf.layers.Dense(units=5)

    def call(self, x, training=False):
        batch_size = x.shape[0]
        #print("Spectronet - input", x.shape)
        x = self.conv1(x)
        #print("Spectronet - conv1", x.shape)
        x = self.conv2(x)
        #print("Spectronet - conv2", x.shape)
        x = self.conv3(x)
        #print("Spectronet - conv3", x.shape)
        x = self.conv4(x)
        #print("Spectronet - conv4", x.shape)
        x = self.conv5(x)
        #print("Spectronet - conv5", x.shape)
        x = self.dense(tf.reshape(x, [batch_size, -1]))  # this is not so correct
        #print("Spectronet - dense", x.shape)
        #x = self.dense(tf.reshape(x, [-1, 49152]))
        x = self.dropout(x, training=training)
        #print("Spectronet - dropout", x.shape)

        return self.logits(x)

