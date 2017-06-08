import numpy as np
import os
import tensorflow as tf
import urllib2
##Matplotlib chooses Xwindows backend by default. You need to set matplotlib do not use Xwindows backend. Uncomment next 2 lines if you are running this on a server which doesn't have a display
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

slim = tf.contrib.slim

image_size = inception.inception_v3.default_image_size

with tf.Graph().as_default():
    url = sys.argv[1]
    image_string = urllib2.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
    checkpoints_dir='slim_pretrained' 
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
        slim.get_model_variables('InceptionV3'))
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
    names = imagenet.create_readable_names_for_imagenet_labels()
    result_text=''
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (100*probabilities[index], names[index]))
    result_text+=str(names[sorted_inds[0]])+'=>'+str( "{0:.2f}".format(100*probabilities[sorted_inds[0]]))+'%\n' 
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.text(225,225,result_text,horizontalalignment='center', verticalalignment='center',fontsize=21,color='blue')
    plt.axis('off')
    plt.show()


