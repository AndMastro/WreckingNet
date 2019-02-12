import tensorflow as tf

class SpectroCNN(tf.keras.Model):

    def __init__(self):
        super(SpectroCNN, self).__init__()

        self.conv1 = tf.layers.Conv2D(filters=24,
                                      kernel_size=[6, 6],
                                      strides=(1, 1),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(filters=24,
                                      kernel_size=[6, 6],
                                      strides=(1, 1),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.Conv2D(filters=48,
                                      kernel_size=[5, 5],
                                      strides=(2, 2),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv4 = tf.layers.Conv2D(filters=48,
                                      kernel_size=[5, 5],
                                      strides=(2, 2),
                                      kernel_initializer=tf.initializers.glorot_uniform,
                                      padding="same",
                                      activation=tf.nn.relu)
        self.conv5 = tf.layers.Conv2D(filters=64,
                                      kernel_size=[4, 4],
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.dense(tf.reshape(x, [batch_size, -1]))  # this is not so correct
        #x = self.dense(tf.reshape(x, [-1, 49152]))
        x = self.dropout(x, training=training)

        return self.logits(x)

