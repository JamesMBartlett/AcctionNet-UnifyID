import matplotlib as mpl
mpl.use('Agg')
from keras.models import load_model
from cleverhans.attacks import fgsm
from keras import backend as K
from argparse import ArgumentParser
import six
import pickle
import numpy as np
import matplotlib.pyplot as plt
from GADF_tensorflow import GAFLayer
from data import Data

def generate_attack(model, X_test, y_test, epsilon, attack):
    adv_X = attack(model.inputs[0], model.outputs[0], eps=epsilon)
    sess = K.get_session()
    X_test_adv, = batch_eval_keras(sess, model.inputs, [adv_X], [X_test], 128)
    score = model.evaluate(X_test_adv, y_test)
    print("Score on Adversarial Examples: ", score, " with epsilon: ", epsilon)
    return score

def generate_fgsm_plots(model, name, X_test, y_test, attack):
    accs = []
    scores = []
    epsilons = np.linspace(0, 0.5, 21)
    for eps in epsilons:
        score = generate_attack(model, X_test, y_test, eps, attack)
        accs.append(score[1])
        scores.append(score)
    with open('adversarial_scores_%s.pkl' % name, 'w') as f:
        pickle.dump(scores, f)
    #line = plt.plot(epsilons, accs)
    #line[0].figure.savefig(name + '.png')

def batch_eval_keras(sess, tf_inputs, tf_outputs, numpy_inputs, batch_size):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    :param sess:
    :param tf_inputs:
    :param tf_outputs:
    :param numpy_inputs:
    :param batch_size:
    """

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in six.moves.xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    with sess.as_default():
        for start in six.moves.xrange(0, m, batch_size):
            batch = start // batch_size
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Compute batch start and end indices
            start = batch * batch_size
            end = start + batch_size
            numpy_input_batches = [numpy_input[start:end]
                                   for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            feed_dict[K.learning_phase()] = False
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model', help="name of saved model file")
    parser.add_argument('--data', help="name of data file")
    parser.add_argument('--epsilon', type=float, help="epsilon for fgsm attack")
    parser.add_argument('--mode', type=str, help="mode to run either single or plots")
    parser.add_argument('--attack')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    X_train, X_test, y_train, y_test, labels = Data(args.data).get_data()
    model = load_model(args.model)
    attack = globals()[args.attack]
    if args.mode == 'single':
        generate_fgsm(model, X_test, y_test, args.epsilon, attack)
    elif args.mode == 'plots':
        generate_fgsm_plots(model, args.model, X_test, y_test, attack)
    else:
        raise ValueError("invalid mode")
