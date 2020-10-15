
import tensorflow as tf
import numpy as np
from jenkspy import jenks_breaks


class dAutoEncoder(tf.keras.Model):

    def __init__(self, params):
        super().__init__()
        self.input_size = params['dim'] + 50  # params['dim_chars']
        self.hidden_size = int(np.floor(self.input_size/5))
        self.code_size = int(np.floor(self.hidden_size/4))

        self.hidden_layer_1 = tf.keras.layers.Dense(self.hidden_size,
                                                    name='ae_hidden_1')
        self.code_layer = tf.keras.layers.Dense(self.code_size,
                                                name='ae_code')
        self.hidden_layer_2 = tf.keras.layers.Dense(self.hidden_size,
                                                    name='ae_hidden_2')
        self.output_layer = tf.keras.layers.Dense(self.input_size,
                                                  name='ae_output')

        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, data):
        x = self.hidden_layer_1(data)
        x = self.code_layer(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)

        return x

    def train_step(self, data):
        with tf.GradientTape() as tape_:
            output = self.call(data)
            loss_value = tf.keras.losses.MSE(y_true=data, y_pred=output)
            loss_value += sum(self.losses)
        grads = tape_.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def reconstruction_error(self, data):
        output = self.call(data)
        reconstruction_error = tf.keras.losses.MSE(y_true=data, y_pred=output)

        return reconstruction_error


def get_jenks_break(tensor):
    return tf.cast(jenks_breaks(tensor.numpy(), nb_class=2)[1], tf.float32)
