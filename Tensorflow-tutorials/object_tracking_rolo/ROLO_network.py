import sys
import ROLO_utils as utils
import tensorflow as tf
import cv2

import numpy as np
import os
import time
import random
class ROLO():
    disp_console = True
    restore_weights = True
    num_steps = 3
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    num_gt = 4
    num_input = num_feat + num_predict # data input: 4096+6= 5002
    rolo_weights_file = os.path.join(os.getcwd(), 'checkpoint', 'demo3.ckpt')

    batch_size = 1
    display_step = 1

    def __init__(self, path1, video = True):
        
        print("Initialising ROLO")
        self.x = tf.placeholder("float32", [None, self.num_steps, self.num_input])
        self.y = tf.placeholder("float32", [None, self.num_gt])
        if video:
            self.path = os.path.dirname(path1)
            self.load_config(path1, video = True)
        else:
            self.path = os.path.split(path1)[0]
            self.load_config(path1, video = False)
        self.rolo_utils = utils.ROLO_utils()
        self.output_path = os.path.join(self.path, 'rolo_out_test')
        utils.createFolder(self.output_path)
        self.build_networks()

    def run_net(self):
        start_time = time.time()
        self.testing(os.path.join(self.path, 'yolo_out'))
        elapsed_time = time.time() - start_time
        print('ROLO network executed in {:.0f} minutes {:.0f} seconds'.format(elapsed_time//60,elapsed_time%60))
        return self.w_img, self.h_img, self.num_steps

    def load_config(self, path, video):
        if video:
            data = cv2.VideoCapture(path)
            self.testing_iters = int(data.get(cv2.CAP_PROP_FRAME_COUNT))
            _,img = data.read()
            self.h_img, self.w_img, _ = img.shape
        else:
            temp = next(os.walk(path))[2]
            self.testing_iters = len(temp)
            img = cv2.imread(os.path.join(path, temp[2]))
            self.h_img, self.w_img, _ = img.shape


    def LSTM_single(self,_X):

        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, self.num_steps, 0) # n_steps * (batch_size, num_input)
        cell = tf.nn.rnn_cell.LSTMCell(self.num_input, name='basic_lstm_cell')
        state = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.static_rnn(cell, _X, initial_state=state, dtype=tf.float32)
        tf.get_variable_scope().reuse_variables()
        return outputs


    def build_networks(self):
        if self.disp_console : print("Building ROLO graph...")

        # Build rolo layers
        self.lstm_module = self.LSTM_single(self.x)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.disp_console : print("Loading complete!" + '\n')


    def testing(self, x_path):

        print("TESTING ROLO...")
        # Use rolo_input for LSTM training
        pred = self.LSTM_single(self.x)
        print("pred: ", pred)
        self.pred_location = pred[-1][:, 4097:4101]
        print("pred_location: ", self.pred_location)
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:

            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print("Loading complete!" + '\n')
            else:
                sess.run(init)


            id = 0
            total_time = 0.0


            # Keep training until reach max iterations
            while id < self.testing_iters - self.num_steps:
                ti = time.time()
                # Load training data & ground truth
                batch_xs = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps, id)

                # Reshape data to get 3 seq of 5002 elements
                batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])

                pred_location= sess.run(self.pred_location,feed_dict={self.x: batch_xs})
                print('Image_no{}'.format(id+1))
                print("ROLO Pred: ", pred_location)
                print("ROLO Pred in pixel: ", pred_location[0][0]*self.w_img, pred_location[0][1]*self.h_img, pred_location[0][2]*self.w_img, pred_location[0][3]*self.h_img)

                # Save pred_location to file
                utils.save_rolo_output_test(self.output_path, pred_location, id, self.num_steps, self.batch_size)

                if id % self.display_step == 0:
                    cycle_time = time.time()-ti
                    total_time += cycle_time
                id += 1
                print(cycle_time)

            print("ROLO network executed")
            print(total_time)
        return None

