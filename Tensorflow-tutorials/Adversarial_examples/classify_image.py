import argparse
import os.path
import re
import sys
import tarfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None
NUM_CLASSES = 1008
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def save_image(image,original_shape,name):
  decoded_image = image[0]/2 + 0.5
  decoded_image = decoded_image*255
  img = cv2.resize(decoded_image,(original_shape[1],original_shape[0]))
  cv2.imwrite("adver_images/"+name,img[:,:,[2,1,0]])

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()
  original_shape = cv2.imread(image).shape
  # Creates graph from saved GraphDef.
  create_graph()
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    image_tensor = sess.graph.get_tensor_by_name('Mul:0')
    image = sess.run(image_tensor,{'DecodeJpeg/contents:0': image_data})
    predictions = sess.run(softmax_tensor,
                           {'Mul:0': image})
    predictions = np.squeeze(predictions)
    print("Generating Adversial Example...\n\n")
    target_class = tf.reshape(tf.one_hot(174,NUM_CLASSES),[1,NUM_CLASSES])
    adv_image_tensor,noise = step_targeted_attack(image_tensor, 0.007, target_class, softmax_tensor)
    #Uncomment the appropriate method that you want to try.
    #adv_image_tensor,noise = step_ll_adversarial_images(image_tensor, 0.007, softmax_tensor)
    #adv_image_tensor,noise = step_fgsm(image_tensor, 0.007, softmax_tensor)
    #adv_image = sess.run(adv_image_tensor,{'DecodeJpeg/contents:0': image_data})
    adv_image = image
    adv_noise = np.zeros(image.shape)
    for i in range(10):
      print("Iteration "+str(i))
      adv_image,a = sess.run((adv_image_tensor,noise),{'Mul:0': adv_image})
      adv_noise = adv_noise + a

    plt.imshow(image[0]/2 + 0.5)
    #plt.show()
    save_image(image,original_shape,"original.jpg")
    plt.imshow(adv_image[0]/2 + 0.5)
    #plt.show()
    save_image(adv_image,original_shape,"adv_image.jpg")
    plt.imshow(adv_noise[0]/2 + 0.5)
    #plt.show()
    save_image(adv_noise,original_shape,"adv_noise.jpg")
    
    adv_predictions = sess.run(softmax_tensor, {'Mul:0' : adv_image})
    adv_predictions = np.squeeze(adv_predictions)
    
    noise_predictions = sess.run(softmax_tensor, {'Mul:0' : adv_noise})
    noise_predictions = np.squeeze(noise_predictions)
    
    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    print("\nNormal Image ...\n")
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    print("\nAdversial Image ...\n")
    top_k = adv_predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      #print(node_id)
      human_string = node_lookup.id_to_string(node_id)
      score = adv_predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    print("\nAdversial Noise ...\n")
    top_k = noise_predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      #print(node_id)
      human_string = node_lookup.id_to_string(node_id)
      score = noise_predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

def step_targeted_attack(x, eps, one_hot_target_class, logits):
  #one_hot_target_class = tf.one_hot(target, NUM_CLASSES)
  #print(one_hot_target_class,"\n\n")
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                  logits,
                                                  label_smoothing=0.1,
                                                  weights=1.0)
  x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
  x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
  return tf.stop_gradient(x_adv),eps * tf.sign(tf.gradients(cross_entropy, x)[0])

def step_fgsm(x, eps, logits):
  label = tf.argmax(logits,1)
  one_hot_label = tf.one_hot(label, NUM_CLASSES)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot_label,
                                                  logits,
                                                  label_smoothing=0.1,
                                                  weights=1.0)
  x_adv = x + eps*tf.sign(tf.gradients(cross_entropy,x)[0])
  x_adv = tf.clip_by_value(x_adv,-1.0,1.0)
  return tf.stop_gradient(x_adv),eps * tf.sign(tf.gradients(cross_entropy, x)[0])

def step_ll_adversarial_images(x, eps, logits):
  least_likely_class = tf.argmin(logits, 1)
  #print(least_likely_class)
  one_hot_ll_class = tf.one_hot(least_likely_class, NUM_CLASSES)
  one_hot_ll_class = tf.reshape(one_hot_ll_class,[1,NUM_CLASSES])
  return step_targeted_attack(x, eps, one_hot_ll_class,logits)
  #return step_targeted_attack(x, eps, least_likely_class,logits)

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_dir',
      type=str,
      default='imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
