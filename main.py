from __future__ import print_function
import numpy as np

from argparse import ArgumentParser

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import pickle
from tqdm import tqdm
from models import *
from data import Data

# only use one GPU, and allow memory growth
KTF.set_session(tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                         device_count={'CPU':4, 'GPU':1},
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True),
                                                 log_device_placement=False)))

def train(filename, model_name, model=mnist_preprocessed, nb_epoch=100, data=None, channels=8, **kwargs):
    # load data
    if data:
        X_train, X_test, y_train, y_test, labels, _ = data
    else:
        X_train, X_test, y_train, y_test, labels = Data(filename).get_data()
    print("Data loaded.")

    if model == FCNN:
        # FCNN doesn't have softmax layer
        runFeatures = False
    else:
        runFeatures = True

    # compute class weights to balance loss function based on freq
    # of different classes in the dataset
    class_weights = {}
    for i in range(len(labels)):
        class_weights[i] = 1. / np.sum(np.argmax(y_train, axis=1) == i)
        norm = sum([class_weights[i] for i in class_weights])
    for i in class_weights:
        class_weights[i] /= norm

    # setup model
    model = model(nb_classes=len(labels), channels=channels, *kwargs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, batch_size=128,validation_split=0.125,
              nb_epoch=nb_epoch,verbose=1)

    # save pre-softmax features
    if not data and runFeatures:
        layer_name = "features"
        no_softmax_model = Model(input=model.input, output=model.get_layer(layer_name).output)
        no_softmax_model.save("%s_features.h5" % model_name)

    # save model
    model.save("%s.h5" % model_name)
    model.save_weights("%s_weights")

    # print model evaluation
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return score

def cross_validate(filename, model_name, model=mnist_preprocessed, nb_epoch=100, channels=8,**kwargs):
    nb_validate = 30
    scores = []
    data = Data(filename)
    data.get_data()
    for i in tqdm(range(nb_validate)):
        shuffled_data = data.get_shuffled_data()
        print("Seed: %s" % shuffled_data[-1])
        name = model_name + '_c_v_' + str(i)
        score = train(filename, name, model, nb_epoch, shuffled_data, channels=channels)
        scores.append((score[0], score[1]))
    print(scores)
    with open('cross_validate_%s' % model_name, 'w') as f:
        pickle.dump(scores, f)

def get_args():
    parser = ArgumentParser(description="Train various models on IMU-MEMS data")
    parser.add_argument('--name', help="name to use when saving models")
    parser.add_argument('--model', help="exact name of function to get model from")
    parser.add_argument('--data_file', nargs='+', help="name of file to load
                        data from")
    parser.add_argument('--nb_e', help='number of epochs to run', type=int)
    parser.add_argument('--cross_validate', action='store_true', help="flag for
                        whether to cross validate")
    parser.add_argument('--channels', type=int, help="number of channels in
                        input data")
    args = parser.parse_args()
    return args

def main(args):
    if args.nb_e:
        nb_epoch = args.nb_e
    else:
        nb_epoch = 100

    if args.channels:
        channels= args.channels
    else:
        channels = 4

    if args.cross_validate and args.model and args.filename:
        model = globals()[args.model]
        print("cross validating")
        cross_validate(args.filename, args.name, model, nb_epoch=nb_epoch,
                       channels=channels)

    elif args.model and args.filename:
        print("training")
        model = globals()[args.model]
        train(args.filename, args.name, model, nb_epoch=nb_epoch, channels=channels)



if __name__ == '__main__':
    args = get_args()
    main(args)
