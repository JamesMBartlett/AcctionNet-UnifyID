from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, \
        Flatten, Input, Convolution2D, MaxPooling2D, \
        AveragePooling2D, Convolution1D, GRU, Lambda, \
        MaxPooling1D, Merge, Reshape, BatchNormalization
from GADF_tensorflow import GAFLayer

def conv_model(img_rows=200, img_cols=4, channels=1, nb_classes=13):
    """
    Convolutional Model
    Default parameters represent parameters for running on raw time series
    data, imaged only in the sense that it is reshaped to a
    Nximg_rowsximg_colsxchannels tensor
    """
    nb_filters = 32
    nb_pool = 2
    kernel_size = (2, 2)

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('tanh'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, name='features'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def conv_preprocessed(img_rows=64, img_cols=64, channels=4, nb_classes=13):
    """
    Convenience function for setting parameters for convolutional model run on
    time series data imaged with GADF or GASF or MTF.
    """
    return conv_model(img_rows=img_rows, img_cols=img_cols, channels=channels,
                      nb_classes=nb_classes)

def mnist_GADF(img_rows=200, img_cols=4, channels=1, nb_classes=13, GAF_type='GADF', paa_size=64):
    """
    Convolutional model using tensorflow to compute GASDF
    Not recommended because its much slower than running
    GASDF once on all images, and then putting these into
    conv_preprocessed model.
    """
    model = Sequential()
    model.add(GAFLayer(input_shape=(img_rows, img_cols, channels)))
    model.add(conv_preprocessed(nb_classes=nb_classes))
    return model


def FCNN(img_rows=200, img_cols=4, channels=1 nb_classes=13):
    """
    Fully Connected Model
    Runs on input tensor of shape Nximg_rowsximg_colsxchannals
    """
    x = Input(shape=(img_rows, img_cols, channels))
    flat = Flatten()(x)
    dense1 = Dense(512, activation='tanh')(flat)
    dense2 = Dense(256, activation='tanh')(dense1)
    dense3 = Dense(128, activation='tanh')(dense2)
    dropout = Dropout(0.5)(dense3)
    output = Dense(nb_classes, activation='softmax')(dropout)
    return Model(x, output)

def recurrent(nb_classes=13, img_rows=200, img_cols=4, channels=1):
    """
    Recurrent Model
    Runs on input tensor of shape Nximg_rowsximg_colsxchannels
    """
    x = Input(shape=(img_rows, img_cols, channels))
    reshaped = Reshape((img_rows, img_cols * channels))(x)
    gru1 = GRU(1024, return_sequences=True, activation='tanh')(reshaped)
    gru2 = GRU(256, return_sequences=True, activation='tanh')(gru1)
    gru3 = GRU(64, activation='tanh')(gru2)
    dense = Dense(nb_classes, activation='softmax')(gru3)
    return Model(x, dense)


