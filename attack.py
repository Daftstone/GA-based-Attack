import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.platform import flags
import tensorflow as tf
import time
import sys
import random

sys.path.append("cleverhans")
from cleverhans.utils_keras import KerasModelWrapper
from datasets import data_fmnist, data_cifar10, data_mnist, data_svhn
from utils import get_model, model_eval, GetGradient, get_value
from get_adv import get_adv_examples
from GA import GA

# Create TF session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def attack():
    """
    :return:
    """

    # get dataset
    dataset_dir = {"mnist": data_mnist, "cifar10": data_cifar10, "fmnist": data_fmnist, 'svhn': data_svhn}
    x_train, y_train, x_test, y_test = dataset_dir[FLAGS.dataset]()
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    Model = get_model(FLAGS.dataset, nb_classes)
    Model.train(x_train, y_train, x_test, y_test, 128, FLAGS.nb_epochs, FLAGS.is_train)
    model = Model.model

    acc = model_eval(sess, x, y, model(x), x_test, y_test, batch_size=FLAGS.batch_size)
    print('Test accuracy on legitimate examples: %.4f' % acc)

    # select successful classified examples
    pred_all = model.predict(x_test)
    classfied_flag = np.argmax(pred_all, axis=-1) == np.argmax(y_test, axis=-1)
    x_test = x_test[classfied_flag]
    y_test = y_test[classfied_flag]

    # use the first 1000 samples of the testset
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    acc = model_eval(sess, x, y, model(x), x_test, y_test, batch_size=FLAGS.batch_size)
    print('Test accuracy on legitimate examples: %.4f' % acc)

    # wrap keras model
    wrap = KerasModelWrapper(model)

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    grad_temp = GetGradient(wrap, sess=sess)
    params = {'clip_min': 0., 'clip_max': 1.}
    grad = grad_temp.generate(x, **params)
    # Consider the attack to be constant
    gradient_tensor = tf.stop_gradient(grad)
    # Evaluate the accuracy of the trained model on adversarial examples
    gradient = get_value(sess, x, gradient_tensor, x_test, batch_size=FLAGS.batch_size)
    print('Test accuracy on fgsm examples: %0.4f\n' % model_eval(sess, x, y, model(x),
                                                                 np.clip(
                                                                     x_test + np.sign(gradient) * FLAGS.eps,
                                                                     0, 1), y_test, batch_size=FLAGS.batch_size))

    # test BIM attack
    d1 = get_adv_examples(sess, wrap, "bim", x_test, y_test)
    print('Test accuracy on bim examples: %0.4f\n' % model_eval(sess, x, y, model(x), d1, y_test,
                                                                batch_size=FLAGS.batch_size))
    # test MI-FGSM attack
    d2 = get_adv_examples(sess, wrap, "mi-fgsm", x_test, y_test)
    print('Test accuracy on mi-fgsm examples: %0.4f\n' % model_eval(sess, x, y, model(x), d2, y_test,
                                                                    batch_size=FLAGS.batch_size))
    # test SPSA attack
    d3 = get_adv_examples(sess, wrap, "spsa", x_test, y_test)
    print('Test accuracy on spsa examples: %0.4f\n' % model_eval(sess, x, y, model(x), d3, y_test,
                                                                 batch_size=FLAGS.batch_size))

    # Testing GA-based attacks
    success = list()
    for i in range(len(x_test)):
        true_class = np.argmax(y_test[i])
        if (FLAGS.solve_method == 'GA'):
            grad = None
            multi_fit = False
        elif (FLAGS.solve_method == 'Hybrid-GA'):
            grad = gradient[i]
            multi_fit = False
        elif (FLAGS.solve_method == 'MF-GA'):
            grad = None
            multi_fit = True
        elif (FLAGS.solve_method == 'Hybrid-MF-GA'):
            grad = gradient[i]
            multi_fit = True
        else:
            print("Wrong method! GA will be used")
            grad = None
            multi_fit = False
        tt, eval_num, save = GA(FLAGS.n_point, FLAGS.generation,
                                img_rows * img_cols * nchannels, model,
                                x_test[i],
                                true_class, FLAGS.eps, FLAGS.batch_size,
                                grad, multi_fit)

        x_adv = np.clip(x_test[FLAGS.image] + np.reshape(tt, x_test.shape[1:]) * FLAGS.eps, 0, 1)
        pred = model.predict(np.expand_dims(x_adv, axis=0))
        predict = np.argmax(pred, axis=-1)
        success.append(predict[0] != true_class)
        print(i, ", success attack: ", success[-1])
    print(sum(success) / len(success))


def main(argv=None):
    attack()


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_float("eps", 0.03, 'perturbation step')
    flags.DEFINE_integer("n_point", 50, 'Number of individuals in the population')
    flags.DEFINE_integer("generation", 500, 'Number of evolutions of the algorithm')
    flags.DEFINE_string("dataset", "mnist", "data to test")
    flags.DEFINE_string("solve", "GA", "attack method")
    flags.DEFINE_bool("is_train", False, "traing online or load weight from file")
    flags.DEFINE_integer("batch_size", 128, 'the batch size in training')
    flags.DEFINE_integer("nb_epochs", 100, 'the number of epochs of training')

    tf.app.run()
