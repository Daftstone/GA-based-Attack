import tensorflow as tf
import numpy as np
from keras import backend as K
from utils import get_adv
from tensorflow.python.platform import flags

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import ElasticNetMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import DeepFool
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import SPSA

FLAGS = flags.FLAGS


def get_adv_examples(sess, wrap, attack_type, X, Y):
    """
        detect adversarial examples
        :param sess: target model session
        :param wrap: wrap model
        :param attack_type:  attack for generating adversarial examples
        :param X: examples to be attacked
        :param Y: correct label of the examples
        :return: x_adv: adversarial examples
    """
    x = tf.placeholder(tf.float32, shape=(None, X.shape[1], X.shape[2],
                                          X.shape[3]))
    y = tf.placeholder(tf.float32, shape=(None, Y.shape[1]))
    adv_label = np.copy(Y)
    batch_size = FLAGS.batch_size

    # Define attack method parameters
    if (attack_type == 'fgsm'):
        attack_params = {
            'eps': FLAGS.eps,
            'clip_min': 0.,
            'clip_max': 1.
        }
        attack_object = FastGradientMethod(wrap, sess=sess)
    elif (attack_type == 'mi-fgsm'):
        attack_object = MomentumIterativeMethod(wrap, back='tf', sess=sess)
        if (FLAGS.eps < 0.05):
            attack_params = {'eps': FLAGS.eps, 'eps_iter': FLAGS.eps,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'batch_size': FLAGS.batch_size
                             }
        else:
            attack_params = {'eps': FLAGS.eps, 'eps_iter': 0.05,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'batch_size': FLAGS.batch_size
                             }
    elif (attack_type == 'bim'):
        attack_object = BasicIterativeMethod(wrap, back='tf', sess=sess)
        if (FLAGS.eps < 0.05):
            attack_params = {'eps': FLAGS.eps, 'eps_iter': FLAGS.eps,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'batch_size': FLAGS.batch_size
                             }
        else:
            attack_params = {'eps': FLAGS.eps, 'eps_iter': 0.05,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'batch_size': FLAGS.batch_size
                             }
    elif (attack_type == 'pgd'):
        attack_object = ProjectedGradientDescent(wrap, back='tf', sess=sess)
        if (FLAGS.eps < 0.05):
            attack_params = {'eps': FLAGS.eps, 'eps_iter': FLAGS.eps,
                             'nb_iter': 100, 'clip_min': 0.,
                             'clip_max': 1., 'y': y, 'batch_size': FLAGS.batch_size
                             }
        else:
            attack_params = {'eps': FLAGS.eps, 'eps_iter': 0.05,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'y': y, 'batch_size': FLAGS.batch_size, 'rand_init_eps': FLAGS.eps
                             }
    elif (attack_type == 'l.l.class'):
        attack_object = BasicIterativeMethod(wrap, back='tf', sess=sess)
        if (FLAGS.eps < 0.05):
            attack_params = {'eps': FLAGS.eps, 'eps_iter': FLAGS.eps,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'y_target': y, 'batch_size': FLAGS.batch_size
                             }
        else:
            attack_params = {'eps': FLAGS.eps, 'eps_iter': 0.05,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'y_target': y, 'batch_size': FLAGS.batch_size
                             }
        ll = np.argmin(Y, axis=-1)
        for i in range(len(Y)):
            ind = ll[i]
            adv_label[i] = np.zeros([Y.shape[1]])
            adv_label[i, ind] = 1
    elif (attack_type == 'enm'):
        attack_object = ElasticNetMethod(wrap, back='tf', sess=sess)
        attack_params = {'y_target': y, 'max_iterations': 10, 'batch_size': 128}
        ll = np.argmin(Y, axis=-1)
        for i in range(len(Y)):
            ind = ll[i]
            adv_label[i] = np.zeros([Y.shape[1]])
            adv_label[i, ind] = 1
    elif (attack_type == 'cw'):
        attack_object = CarliniWagnerL2(wrap, back='tf', sess=sess)
        attack_params = {
            'binary_search_steps': 1,
            'y': y,
            'max_iterations': 100,
            'learning_rate': .2,
            'batch_size': 128,
            'initial_const': 10
        }
    elif (attack_type == 'df'):
        attack_object = DeepFool(wrap, back='tf', sess=sess)
        attack_params = {
            'max_iterations': 50,
            'clip_min': 0., 'clip_max': 1.,
            'overshoot': 0.02
        }
    elif (attack_type == 'spsa'):
        attack_object = SPSA(wrap, back='tf', sess=sess)
        batch_size = 1

    if (attack_type == 'spsa'):
        adv_x = attack_object.generate(x=x, y=y, eps=FLAGS.eps, clip_min=0., clip_max=1., nb_iter=100)
    else:
        adv_x = attack_object.generate(x, **attack_params)

    # Get adversarial examples
    if (attack_type == 'l.l.class' or attack_type == 'enm'):
        x_adv = get_adv(sess, x, y, adv_x, X, adv_label, batch_size=batch_size)
    else:
        x_adv = get_adv(sess, x, y, adv_x, X, Y, batch_size=batch_size)
    return x_adv
