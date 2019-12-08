import os
import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import warnings
from datetime import timedelta
#from IPython.display import clear_output
#from tensorflow.keras.backend import set_session
import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from utils import *
from model import *
import argparse
#warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
# import logging
# logging.getLogger("tensorflow").setLevel(logging.WARNING)
# tf.autograph.set_verbosity(1)



"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of Attribute GANs for data augmentation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of slasses')
    parser.add_argument('--generation_rate', type=int, default=2, help='Generation rate of images (per epoch)')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--model_name', type=str, default=None, help='path to model if starting from checkpoint')
    parser.add_argument('--truncated', type=bool, default=True, help='If truncated normal should be True or False')


    # parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
    #                     help='Directory name to save the checkpoints')
    # parser.add_argument('--result_dir', type=str, default='results',
    #                     help='Directory name to save the generated images')
    # parser.add_argument('--log_dir', type=str, default='logs',
    #                     help='Directory name to save training logs')

    return parser.parse_args()
def main(args):

    BATCH_SIZE = args.batch_size
    NUM_CLASSES = args.num_classes
    EPOCHS = args.epoch
    GENERATION_RATE = args.generation_rate
    LEARNING_RATE = args.lr
    truncated_uniform_scale_flag = args.truncated

    IMG_WIDTH, IMG_HEIGHT = 128, 128
    #BETA_1, BETA_2 = 0.5, 0.999
    BETA_1, BETA_2 = 0.5, 0.99
    thres_int = 0.5
    
    """ learning rate decay definitions """
    #global_step = tf.Variable(0, trainable=False)
    #LEARNING_RATE = tf.train.exponential_decay(learning_rate = LEARNING_RATE,
    #                                            global_step = global_step,
    #                                            decay_steps = ,
    #                                            decay_rate = ,
    #                                            staircase=False,
    #                                            name="exp_decay_lr")


    """ saving paths """
    if args.model_name is None:
        model_name = time.strftime('%Y-%m-%d_%H:%M:%S_%z') + "_" + str(BATCH_SIZE)
        output_dir = "output"
        model_dir = '{}/{}'.format(output_dir, model_name)
        image_dir = '{}/images'.format(model_dir)
        checkpoints_dir = '{}/checkpoints'.format(model_dir)
        for path in [output_dir, model_dir, image_dir, checkpoints_dir]:
            if not os.path.exists(path):
                os.mkdir(path)
        print("created model")
    else:
        model_name = args.model_name
        print("proceeding to load model: {}".format(model_name))

    """ tf session definitions """
    tf.reset_default_graph()
    tf.random.set_random_seed(42)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)


    """ load TFRecordDataset """
    training_data = Barkley_Deep_Drive('../../self-driving-AttGAN/resources/train.tfrecords')
    validation_data = Barkley_Deep_Drive('../../self-driving-AttGAN/resources/test.tfrecords')

    train_iterator = training_data.get_batch(EPOCHS, BATCH_SIZE, shuffle = True)
    val_iterator = validation_data.get_batch(EPOCHS, BATCH_SIZE, shuffle = True)

    train_image_iterator, train_label_iterator = train_iterator.get_next()
    val_image_iterator, val_label_iterator = val_iterator.get_next()

    """ Placeholders """
    xa = tf.placeholder(tf.float32,shape=[BATCH_SIZE,IMG_WIDTH,IMG_HEIGHT,3],name="xa") #orignal image
    z = encoder(xa, reuse=tf.compat.v1.AUTO_REUSE ) #encoder output

    a = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES],name="a") #original attributes
    b = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES],name="b") #desired attributes

    xb_hat = decoder(z, b, reuse=tf.compat.v1.AUTO_REUSE ) #decoder output
    with tf.control_dependencies([xb_hat]):
        xa_hat = decoder(z, a, reuse=tf.compat.v1.AUTO_REUSE ) #decoder output

    xa_logit_D, xa_logit_C = classifier_and_discriminator(xa, reuse=tf.compat.v1.AUTO_REUSE,  NUM_CLASSES=1)
    xb_logit_D, xb_logit_C = classifier_and_discriminator(xb_hat, reuse=tf.compat.v1.AUTO_REUSE,  NUM_CLASSES=1)
    
    """ penalty """
    lambda_ = {"3" : 1, "2" : 10, "1" : 100}

    """ interpolated image noise"""
    #epsilon = tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)
    #interpolated_image = xa + epsilon * (xb_hat - xa)
    #c_interpolated, d_interpolated = classifier_and_discriminator(interpolated_image, reuse=tf.compat.v1.AUTO_REUSE, NUM_CLASSES=1)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[IMG_WIDTH,1], 
        minval=0.,
        maxval=1.
    )
    differences =  xb_hat - xa
    interpolates = xa + (alpha*differences)
    gradients = tf.gradients(classifier_and_discriminator(interpolates, reuse=tf.AUTO_REUSE), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gp = tf.reduce_mean((slopes-1.)**2)

    """ D loss """
    loss_adv_D =  - ( tf.reduce_mean(xa_logit_D) - tf.reduce_mean(xb_logit_D) )
    #grad_d_interpolated = tf.gradients(d_interpolated, [interpolated_image])[0]
    #slopes = tf.sqrt(1e-10 + tf.reduce_sum(tf.square(grad_d_interpolated), axis=[1, 2, 3]))
    #gp = tf.reduce_mean((slopes - 1.) ** 2)
    
    loss_cls_C = tf.losses.sigmoid_cross_entropy(a, xa_logit_C)

    D_loss = loss_adv_D + gp * lambda_['2'] + loss_cls_C

    """ G loss """
    loss_adv_G = -tf.reduce_mean(xb_logit_D)
    loss_cls_G = tf.losses.sigmoid_cross_entropy(b, xb_logit_C)
    loss_rec = tf.losses.absolute_difference(xa, xa_hat)

    G_loss =  loss_adv_G + lambda_['2'] * loss_cls_G + lambda_['1'] * loss_rec


    """ Training """
    # divide trainable variables into a group for D and a group for G
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'C_D' in var.name ]
    g_vars = [var for var in t_vars if 'G_' in var.name]
    assert(len(t_vars) == len(d_vars ) + len(g_vars )), "mismatch in variable names"

    d_optim = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE,
                                     beta1 = BETA_1,
                                     beta2 = BETA_2).minimize(D_loss, var_list=d_vars)

    g_optim = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE,
                                     beta1 = BETA_1,
                                     beta2 = BETA_2).minimize(G_loss, var_list=g_vars)

    """ Summary """
    d_summary = summary({
        loss_adv_D: 'loss_adv_D',
        gp: 'gp',
        loss_cls_C: 'loss_cls_C',
    }, scope='D_')

    g_summary = summary({
        loss_adv_G: 'loss_adv_G',
        loss_cls_G: 'loss_cls_G',
        loss_rec: 'loss_rec',
    }, scope='G_')

    """ init """
    summary_writer = tf.summary.FileWriter('./graphs', sess.graph)
    saver = tf.train.Saver()
    S_ = tf.Summary()

    """ checkpoints load """
    try:
        #TO DO: partial fix, needs testing
        load_checkpoint(checkpoints_dir, sess, t_vars)
    except:
        sess.run(tf.global_variables_initializer())


    """ mapping defintions """
    d_loss_epoch, g_loss_epoch = [], []
    if NUM_CLASSES == 2:
        label_mapping = {'daytime': [1, 0], 'night': [0, 1]}
    else:
        label_mapping = {'daytime': [1], 'night': [0]}
        flip = {'1':[0], '0': [1] }


    """ main training loop """

    for epoch_no in tqdm.tqdm(range(EPOCHS), total=EPOCHS):

        sess.run(val_iterator.initializer)
        sess.run(train_iterator.initializer)

        try:
            step = 0
            start_time = time.monotonic()
            d_loss_per_batch = []
            g_loss_per_batch = []
            while True:
                # Sample batch from dataset
                image_batch, label_batch = sess.run([train_image_iterator, train_label_iterator])

                # Transform label batch in our simple one hot encoded version
                a_label_batch = np.array([label_mapping[label.decode("utf-8")] for label in label_batch], dtype=np.float32)
                if truncated_uniform_scale_flag:
                    b_label_batch = a_label_batch.copy().astype(np.float32)
                    np.random.shuffle(b_label_batch)
                    a_label_batch = (a_label_batch * 2 - 1) * thres_int
                    b_label_batch = (b_label_batch * 2 - 1) * (np.random.uniform(size=b_label_batch.shape) + 2) / 4.0 * (2 * thres_int)
                    #b_label_batch = tf.random_shuffle(a_label_batch)
                    #a_label_batch = (tf.to_float(a_label_batch) * 2 - 1) * thres_int
                    #b_label_batch = (tf.to_float(b_label_batch) * 2 - 1) * (tf.truncated_normal(tf.shape(b_label_batch)) + 2) / 4.0 * (2 * thres_int)
                    #a_label_batch = sess.run(a_label_batch)
                    #b_label_batch = sess.run(b_label_batch)
                else:
                    b_label_batch = [flip[str(label[0])] for label in a_label_batch]
                    a_label_batch = np.asarray(a_label_batch, dtype=np.float32)
                    b_label_batch = np.asarray(b_label_batch, dtype=np.float32)

                # Optimize
                d_summary_opt, _, D_loss_val = sess.run([d_summary, d_optim, D_loss], feed_dict={xa:image_batch,
                                                                        a: a_label_batch, b: b_label_batch})
                if step % 5 == 0:
                    g_summary_opt, _, G_loss_val = sess.run([g_summary, g_optim, G_loss], feed_dict={xa:image_batch,
                                                                        a: a_label_batch, b: b_label_batch})

                d_loss_per_batch.append(D_loss_val)
                g_loss_per_batch.append(G_loss_val)

                summary_writer.add_summary(d_summary_opt, epoch_no)
                summary_writer.add_summary(g_summary_opt, epoch_no)

                if step % 250 == 0:
                    print("At step ", step, "we have")
                    print("Gen loss: ",np.mean(g_loss_per_batch), " and Desc loss:", np.mean(d_loss_per_batch) , "\n ")
                    S_.ParseFromString(d_summary_opt)
                    print(S_)
                    S_.ParseFromString(g_summary_opt)
                    print(S_)

                step += 1
        except tf.errors.OutOfRangeError:
                checkpoint_save_path = saver.save(sess, '{}/Epoch_{}_{}.ckpt'.format(checkpoints_dir, str(epoch_no), str(step)))
                print('Model is saved at {}!'.format(checkpoint_save_path))
                
                # Generating reconstructed image xa_hat and flipped attribute image xb_hat
                image_batch, label_batch = sess.run([val_image_iterator, val_label_iterator])

                # Transform label batch in our simple one hot encoded version
                a_label_batch = np.array([label_mapping[label.decode("utf-8")] for label in label_batch])#, dtype=np.float64)
                if truncated_uniform_scale_flag:
                    b_label_batch = tf.random_shuffle(a_label_batch)
                    a_label_batch = (a_label_batch * 2 - 1) * thres_int
                    b_label_batch = (tf.to_float(b_label_batch) * 2 - 1) * (tf.truncated_normal(tf.shape(b_label_batch)) + 2) / 4.0 * (2 * thres_int)
                    b_label_batch = sess.run(b_label_batch)
                else:
                    b_label_batch = [flip[str(label[0])] for label in a_label_batch]
                    a_label_batch = np.asarray(a_label_batch, dtype=np.float32)
                    b_label_batch = np.asarray(b_label_batch, dtype=np.float32)

                step_xb_hat = sess.run(xb_hat, feed_dict={a:a_label_batch, b:b_label_batch, xa:image_batch})
                step_xa_hat = sess.run(xa_hat, feed_dict={a:a_label_batch, b:b_label_batch, xa:image_batch})
                
                """image saving"""
                output_path = os.path.join(image_dir, "epoch_no_"+ str(epoch_no) +"_" +".png")
                plot_block_after_epoch(output_path, label_batch, image_batch, step_xa_hat, step_xb_hat, examples = 3)

                end_time = time.monotonic()
                print("END OF EPOCH")
                fmt = "Epoch duration: {}".format(timedelta(seconds=end_time - start_time))
                print(fmt)
                d_loss_epoch.append(np.mean(d_loss_per_batch))
                g_loss_epoch.append(np.mean(g_loss_per_batch))
                print("Discriminator loss: ", d_loss_epoch[-1])
                print("Generator loss: ", g_loss_epoch[-1])
                print("-"*len(fmt))
                pass


    checkpoint_save_path = saver.save(sess, '{}/Epoch_{}_{}.ckpt'.format(checkpoints_dir, str(epoch_no), str(step)))
    print('Finished training\nModel has been saved at {}!'.format(checkpoint_save_path))
    sess.close()
    
    try:
        to_json = {"d_loss": d_loss_epoch,
               "g_loss": g_loss_epoch}
        with open(model_name+'.json', 'w') as f:
            json.dump(to_json, f)
    except:
        print(d_loss_epoch)
        print(g_loss_epoch)

if __name__ == "__main__":
    #tf.app.run(main=main)
    args = parse_args()
    if args is None:
        exit()
    main(args)
