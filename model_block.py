import os
import tensorflow as tf
import numpy as np
#import ops
import time
# import os
# import cv2
# import progressbar
# import sys

import importlib
ops = importlib.import_module("self-adapting-confidence.ops")

class OTBBlock(object):

    def __init__(self, output_path, image_height, image_width, save_epoch_freq=10, initial_learning_rate=0.0001, p=['t','a','u'], q=['t'], verbose=False):                        
        #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        #self.sess = tf.compat.v1.Session(config=session_conf)
        self.sess = tf.compat.v1.Session()
        self.disposed = False
        #self.mode = 'otb-online'
        self.output_path = output_path
        self.steps = 0
        self.save_epoch_freq = save_epoch_freq

        hpad = (16 - image_height%16)%16
        wpad = (16 - image_width%16)%16

        self.image_height = image_height+hpad       
        self.image_width = image_width+wpad

        self.initial_learning_rate = initial_learning_rate
        self.p = p
        self.q = q
        self.logName = "OTB Block"
        self.verbose = verbose

        self.build_model()

    def dispose(self):
        if not self.disposed:
            self.sess.close()
            tf.compat.v1.reset_default_graph()
            self.disposed = True

    def log(self, x):
        if self.verbose:
            print(f"{self.logName}: {x}")

    def build_model(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(" [*] Building model...")

        self.placeholders = {'left':tf.compat.v1.placeholder(tf.float32, [1, self.image_height, self.image_width, 3], name='left'),
                        'right':tf.compat.v1.placeholder(tf.float32, [1, self.image_height, self.image_width, 3], name='right'),
                        'disp':tf.compat.v1.placeholder(tf.float32, [1, self.image_height, self.image_width, 1], name='disparity')}
        
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])        
        
        self.ConfNet_v2() 
        self.build_losses()

    def ConfNet_v2(self):
        self.log(" [*] Building ConfNet model...")

        kernel_size = 3
        filters = 32
        disp = self.placeholders['disp']

        with tf.compat.v1.variable_scope('ConfNet'):
                                    
            with tf.compat.v1.variable_scope('disparity'):  
                with tf.compat.v1.variable_scope("conv1"):
                    self.conv1_disparity = ops.conv2d(disp, [kernel_size, kernel_size, 1, filters], 1, True, padding='SAME')
            
            model_input = self.conv1_disparity

            self.net1, self.scale1 = ops.encoding_unit('1', model_input, filters)
            self.net2, self.scale2 = ops.encoding_unit('2', self.net1,   filters * 4)
            self.net3, self.scale3 = ops.encoding_unit('3', self.net2,   filters * 8)
            self.net4, self.scale4 = ops.encoding_unit('4', self.net3,   filters * 16)
            
            self.net5 = ops.decoding_unit('4', self.net4, num_outputs=filters * 8, forwards=self.scale4)
            self.net6 = ops.decoding_unit('3', self.net5, num_outputs=filters * 4, forwards=self.scale3)
            self.net7 = ops.decoding_unit('2', self.net6, num_outputs=filters * 2,  forwards=self.scale2)
            self.net8 = ops.decoding_unit('1', self.net7, num_outputs=filters, forwards=self.conv1_disparity)
                        
            self.prediction = ops.conv2d(self.net8, [kernel_size, kernel_size, filters, 1], 1, False, padding='SAME')


    def build_losses(self):
        with tf.compat.v1.variable_scope('loss'):

            # prepare validity mask
            self.valid = tf.ones_like(self.placeholders['disp'])

            # texture mask
            self.warped = ops.generate_image_left(self.placeholders['right'], self.placeholders['disp'])
            self.reprojection = tf.reduce_sum(0.85*ops.SSIM(self.warped, self.placeholders['left']) + 0.15*tf.abs(self.warped - self.placeholders['left']), -1, keepdims=True)
            self.identity = tf.reduce_sum(0.85*ops.SSIM(self.placeholders['left'], self.placeholders['right']) + 0.15*tf.abs(self.placeholders['left'] - self.placeholders['right']), -1, keepdims=True)                        
            self.t = tf.cast(self.identity > self.reprojection, tf.float32)
            
            # agreement mask
            self.a = tf.cast(tf.py_func(ops.agreement, [self.placeholders['disp'], 2], tf.float32) > (5**2-1)*0.5, tf.float32)

            # uniqueness mask
            self.u = tf.py_func(ops.uniqueness, [self.placeholders['disp']], tf.float32)

            # initializing inliers and outliers masks
            self.P = tf.ones_like(self.placeholders['disp'])
            self.Q = tf.ones_like(self.placeholders['disp'])

            if 't' in self.p:
                self.P *= self.t
            if 'a' in self.p:
                self.P *= self.a
            if 'u' in self.p:
                self.P *= self.u

            if 't' in self.q:
                self.Q *= (1-self.t)
            if 'a' in self.q:
                self.Q *= (1-self.a)
            if 'u' in self.q:
                self.Q *= (1-self.u)

            # quick implementation for MBCE
            self.proxysignal = self.P * (1 - self.Q)
            self.valid = self.valid * (self.P + self.Q) 
            self.loss = tf.compat.v1.losses.sigmoid_cross_entropy(self.proxysignal, self.prediction, self.valid)

    def load(self, checkpoint_path):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log("Loading....")

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        
        self.saver = tf.compat.v1.train.Saver()
        if checkpoint_path:
            self.saver.restore(self.sess, checkpoint_path)
            self.steps = 0
            self.log(" [*] Load model: SUCCESS")
        else:
            self.log(" [*] Load failed...neglected")
            self.log(" [*] End Testing...")
            raise ValueError('args.checkpoint_path is None')

        self.net_output = tf.nn.sigmoid(self.prediction)

        self.log(' [*] Online Adaptation')
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, var_list=tf.compat.v1.global_variables())
        

    def run(self, left, right, disp):
        self.log(" [*] Running....")

        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)

        self.log(" [*] Start Testing...")

        val_disp, hpad, wpad = ops.mypad(disp)
        val_left, _, _ = ops.mypad(left)
        val_right, _, _ = ops.mypad(right)

        start = time.time()
        _, loss, confidence = self.sess.run([self.optimizer, self.loss, self.net_output], feed_dict={self.placeholders['disp']: val_disp, self.placeholders['left']: val_left, self.placeholders['right']: val_right, self.learning_rate: self.initial_learning_rate})
        confidence = ops.depad(confidence, hpad, wpad)
        current = time.time()

        c = confidence[0]
        c = (c - np.min(c)) / (np.max(c) - np.min(c))
        myConfidence = (c).astype('float32')

        if self.steps % self.save_epoch_freq == 0:
            self.saver.save(self.sess, os.path.join(self.output_path, "model"), global_step=self.steps)

        self.steps += 1

        self.log(" [*] Inference time:" + str(current - start) + "s")
        self.log(f" [*] Loss {loss}")
        
        #coord.request_stop()
        #coord.join(threads)

        return myConfidence
