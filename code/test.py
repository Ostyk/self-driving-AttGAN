import os
import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import warnings
from datetime import timedelta
import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from utils import *
from model import *
import argparse
tf.get_logger().setLevel('ERROR')



"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of Attribute GANs for data augmentation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--images_number', type=int, default=128, help='images number to produce')
    parser.add_argument('--model_name', type=str, default=None, help='path to model if starting from checkpoint')
    parser.add_argument('--graph_path', type=str, default=None, help='path to graph to restore and write to')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')


    return parser.parse_args()

def main(args):
    BATCH_SIZE =  args.images_number
    NUM_CLASSES = args.num_classes
    EPOCHS = args.epoch
    restore_graph_path = args.graph_path
    images_number = args.images_number

    IMG_WIDTH, IMG_HEIGHT = 128, 128
    BETA_1, BETA_2 = 0.4, 0.99
    thres_int = 0.5

    """ saving paths """
    output_dir = "output"
    
    if args.model_name is None:
        model_name = time.strftime('%Y-%m-%d_%H:%M:%S_%z') + "_" + str(BATCH_SIZE)
        print("[*] created model folder")
        model_dir = '{}/{}'.format(output_dir, model_name)
    else:
        model_name = args.model_name
        print("[*] proceeding to load model: {}".format(model_name))
        model_dir = model_name
        
    image_dir = '{}/images'.format(model_dir)
    checkpoints_dir = '{}/checkpoints'.format(model_dir)
    for path in [output_dir, model_dir, image_dir, checkpoints_dir]:
        if not os.path.exists(path):
            os.mkdir(path)

    """ tf session definitions """
    tf.reset_default_graph()
    tf.random.set_random_seed(args.rand_seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)


    """ load TFRecordDataset """
    validation_data = Barkley_Deep_Drive('../resources/test.tfrecords')

    val_iterator = validation_data.get_batch(EPOCHS, BATCH_SIZE, shuffle = True)

    val_image_iterator, val_label_iterator = val_iterator.get_next()

    """ Placeholders """
    xa = tf.placeholder(tf.float32,shape=[BATCH_SIZE,IMG_WIDTH,IMG_HEIGHT,3],name="xa") #orignal image
    z = encoder(xa, reuse=tf.AUTO_REUSE) #encoder output

    a = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES],name="a") #original attributes
    b = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES],name="b") #desired attributes

    xb_hat = decoder(z, b, reuse=tf.AUTO_REUSE) #decoder output
    with tf.control_dependencies([xb_hat]):
        xa_hat = decoder(z, a, reuse=tf.AUTO_REUSE) #decoder output

    xa_logit_D, xa_logit_C = classifier_and_discriminator(xa, reuse=tf.AUTO_REUSE, NUM_CLASSES = NUM_CLASSES)
    xb_logit_D, xb_logit_C = classifier_and_discriminator(xb_hat, reuse=tf.AUTO_REUSE, NUM_CLASSES = NUM_CLASSES)


    """ checkpoints load """
    saver = tf.train.Saver()
    
    try:
        epoch_to_restore = load_checkpoint(checkpoints_dir, sess)
        print("[*] restoring from epoch".format(epoch_to_restore))

    except:
        epoch_to_restore = 0
        print("[*] failed to load checkpoints")
        sess.run(tf.global_variables_initializer())

    """ mapping defintions """
    d_loss_epoch, g_loss_epoch = [], []
    if NUM_CLASSES == 2:
        a_label_mapping = {'daytime': [1, 0], 'night': [0, 1]}
        b_label_mapping = {'daytime': [0, 1], 'night': [1, 0]}
    else:
        label_mapping = {'daytime': [1], 'night': [0]}
        flip = {'1':[0], '0': [1] }

        
    """ image generation """
    
    sess.run(val_iterator.initializer)

    # Generating reconstructed image xa_hat and flipped attribute image xb_hat
    image_batch, label_batch = sess.run([val_image_iterator, val_label_iterator])

    # Transform label batch in our simple one hot encoded version
    if NUM_CLASSES == 2:
        a_label_batch = [a_label_mapping[label.decode("utf-8")] for label in label_batch]
        b_label_batch = [b_label_mapping[label.decode("utf-8")] for label in label_batch]
	else:
        a_label_batch = [label_mapping[label.decode("utf-8")] for label in label_batch]
        b_label_batch = [flip[str(label[0])] for label in a_label_batch]


    # Transform label batch in our simple one hot encoded version
    if truncated_uniform_scale_flag:
        #TO DO: fix a_label_batch dtype from list to array
        b_label_batch = tf.random_shuffle(a_label_batch)
        a_label_batch = (tf.to_float(a_label_batch) * 2 - 1) * thres_int
        b_label_batch = (tf.to_float(b_label_batch) * 2 - 1) * (tf.truncated_normal(tf.shape(b_label_batch)) + 2) / 4.0 * (2 * thres_int)
        a_label_batch = sess.run(a_label_batch)
        b_label_batch = sess.run(b_label_batch)
    else:
        a_label_batch = np.asarray(a_label_batch, dtype=np.float32)
        b_label_batch = np.asarray(b_label_batch, dtype=np.float32)

    	step_xb_hat = sess.run(xb_hat, feed_dict={a:a_label_batch, b:b_label_batch, xa:image_batch})
        step_xa_hat = sess.run(xa_hat, feed_dict={a:a_label_batch, b:b_label_batch, xa:image_batch})
            
            
    """image saving loop"""
    for i in tqdm.tqdm(range(images_number), total = images_number):
    	output_path_xa = os.path.join(image_dir, "xa_image_" + str(i+1) + ".png")
    	output_path_xa_hat = os.path.join(image_dir, "xa_hat_image_" + str(i+1) + ".png")
    	output_path_xb_hat = os.path.join(image_dir, "xb_hat_image_" + str(i+1) + ".png")
    	
    	plt.imsave(output_path_xa, image_batch[i]*255).astype(np.uint8))
    	plt.imsave(output_path_xa_hat, step_xa_hat[i]*255).astype(np.uint8))
    	plt.imsave(output_path_xb_hat, step_xb_hat[i]*255).astype(np.uint8))
    	
        
    print("[*] image saved!")



if __name__ == "__main__":
    #tf.app.run(main=main)
    args = parse_args()
    if args is None:
        exit()
    main(args)
