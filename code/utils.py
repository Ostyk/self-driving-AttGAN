import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

class Barkley_Deep_Drive(object):
    def __init__(self, tfrecord_path):
        self.dataset = tf.data.TFRecordDataset(tfrecord_path)

    @staticmethod
    def normalize(image, label):
        """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
        #image = tf.cast(image, tf.float64) * (2. / 255)-1
        image = tf.cast(image, tf.float64) * (1. / 255)
        return image, label

    @staticmethod
    def decode(serialized_example):
        """
        Parses an image and label from the given `serialized_example`.
        It is used as a map function for `dataset.map`
        """
        IMAGE_SHAPE = (128,128,3)

        # 1. define a parser
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
            })

        # 2. Convert the data
        image = tf.decode_raw(features['image'], tf.float32)
        label = features['label']

        # 3. reshape
        image = tf.convert_to_tensor(tf.reshape(image, IMAGE_SHAPE))

        return image, label

    def get_batch(self, EPOCHS=20, batch_size = 32, shuffle = True, num_threads = -1, buffer_size=4096):

        self.dataset = self.dataset.map(self.decode, num_parallel_calls=num_threads)

        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size)

        self.dataset = self.dataset.map(self.normalize)
        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)
        #self.dataset = self.dataset.repeat(EPOCHS)

        #iterator = self.dataset.make_initializable_iterator()
        iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)

        return iterator

def summary(tensor_collection,
            summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram'],
            scope=None):
    """Summary.

    usage:
        1. summary(tensor)
        2. summary([tensor_a, tensor_b])
        3. summary({tensor_a: 'a', tensor_b: 'b})
    """
    def _summary(tensor, name, summary_type):
        """Attach a lot of summaries to a Tensor."""
        if name is None:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
            name = re.sub(':', '-', name)

        summaries = []
        if len(tensor.shape) == 0:
            summaries.append(tf.summary.scalar(name, tensor))
        else:
            if 'mean' in summary_type:
                mean = tf.reduce_mean(tensor)
                summaries.append(tf.summary.scalar(name + '/mean', mean))
            if 'stddev' in summary_type:
                mean = tf.reduce_mean(tensor)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                summaries.append(tf.summary.scalar(name + '/stddev', stddev))
            if 'max' in summary_type:
                summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
            if 'min' in summary_type:
                summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
            if 'sparsity' in summary_type:
                summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
            if 'histogram' in summary_type:
                summaries.append(tf.summary.histogram(name, tensor))
        return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]

    with tf.name_scope(scope, 'summary'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)


def load_checkpoint(ckpt_dir_or_file, session, var_list=None):
    """
    Load checkpoint
    """
    if os.path.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    restorer = tf.train.Saver(var_list)
    restorer.restore(session, ckpt_dir_or_file)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)

    
    
    

    
    
def plot_block_after_epoch(save_path, label_batch, image_batch, step_xa_hat, step_xb_hat,
                           examples = 3, figsize=(10, 10), plot=False):
    """
    Function that plots outputs after every epoch
    :param save_path: full path directory with extension
    :param examples: number of rows default 
    :return None: saves to file
    """
    number_of_images = 3
    fig, axes = plt.subplots(examples, number_of_images, figsize=figsize)#,  squeeze=False, sharey=True, sharex=True)
    for row in range(examples):
        for column in range(number_of_images):
            
            if column==0:
                axes[row,column].set_title(label_batch[row].decode("utf-8"))                           
                axes[row,column].imshow( (image_batch[row] * 255).astype(np.uint8) )
            elif column==1:
                axes[row,column].set_title("$x_a$ reconstruction")                           
                axes[row,column].imshow( (step_xa_hat[row] * 255).astype(np.uint8) )
            elif column==2:
                axes[row,column].set_title("$x_b$ output")    
                axes[row,column].imshow( (step_xb_hat[row] * 255).astype(np.uint8) )
            axes[row,column].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    if not plot:
        plt.close()
    plt.savefig(save_path, bbox_inches = "tight")