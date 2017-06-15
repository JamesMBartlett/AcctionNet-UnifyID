import tensorflow as tf
from keras.engine.topology import Layer

class GAFLayer(Layer):

    def __init__(self, GAF_type='GADF', paa_size=64, **kwargs):
        super(GAFLayer, self).__init__(**kwargs)
        self.GAF_type = GAF_type
        self.paa_size = paa_size

    def build(self, input_shape):
        super(GAFLayer, self).build(input_shape)

    def call(self, x):
        rescaled = self.rescale(x)
        paa = self.paa(rescaled)
        output = self.GAF(paa)
        output.set_shape([None, self.paa_size, self.paa_size,
            x.get_shape()[2]])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.paa_size, self.paa_size, input_shape[2])

    def rescale(self, X):
        X_min = tf.reduce_min(X, reduction_indices=1, keep_dims=True)
        X_max = tf.reduce_max(X, reduction_indices=1, keep_dims=True)
        return tf.divide((X - X_min), (X_max - X_min))

    def paa(self, X):
        shape = tf.slice(tf.shape(X), [1], [1])
        size = tf.div(shape, self.paa_size)

        def paa_one(img):
            def fn(ind):
                start = ind
                sliced_img = tf.slice(img, [tf.cast(start, tf.int32)*size[0], 0, 0], [size[0], -1, -1])
                return  tf.divide(tf.reduce_sum(sliced_img, reduction_indices=0), tf.to_float(size))
            return tf.map_fn(fn, tf.range(self.paa_size, dtype=tf.float32))
        return tf.map_fn(paa_one, X)

    def GAF(self, X):
        X_reshaped = tf.squeeze(X)
        def mat_mul_one(img):
            y = tf.sqrt(1.0 - tf.square(img))
            x_expand = tf.expand_dims(tf.transpose(img), 2)
            y_expand = tf.expand_dims(tf.transpose(y), 2)
            if self.GAF_type == 'GASF':
                return tf.transpose(tf.matmul(x_expand, x_expand, transpose_b=True) - tf.matmul(y_expand, y_expand, transpose_b=True))
            elif self.GAF_type == 'GADF':
                return -tf.transpose(tf.matmul(y_expand, x_expand, transpose_b=True) - tf.matmul(x_expand, y_expand, transpose_b=True))
        return tf.map_fn(mat_mul_one, X_reshaped)
