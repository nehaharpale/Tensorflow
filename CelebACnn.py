#Code with Selective Learning
import tensorflow as tf
# import shutil
# from tensorflow.python.saved_model import tag_constants
import numpy as np
import pandas as pd
import os, time, sys
import matplotlib.image as mpimg
import random
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from natsort import natsorted
import re

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
img_width = 178
img_height = 218
train_data = 0
test_data = 2
eval_data = 1
image_size = 178
batch_size = 128  # 100
epoch = 22
train_sim = 162770
eval_sim = 19866
test_sim = 19963
crop_width = 178
crop_height = 178
col_dim = 3
attr_num = 40
learning_rate = 0.2
model_path = "/home/nharpale/Desktop/Model/"
summary_path = "/home/nharpale/Desktop/Summary/"

data_dir = "/home/nharpale/Desktop/Celeba/img_align_celeba/"
label_dir = "/home/nharpale/Desktop/Celeba/list_attr_celeba.txt"

class mymodel():
    def __init__(self, sess, img_width, img_height, crop_width, crop_height, batch_size, train_sim, eval_sim,
                 test_sim, epoch, col_dim, learning_rate, data_dir, label_dir, model_path, summary_path, attr_num):
        self.sess = sess
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.crop_width = int(crop_width)
        self.crop_height = int(crop_height)
        self.batch_size = int(batch_size)
        self.train_sim = int(train_sim)
        self.eval_sim = int(eval_sim)
        self.test_sim = int(test_sim)
        self.epoch = int(epoch)
        self.col_dim = int(col_dim)
        self.learning_rate = float(learning_rate)
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.model_path = model_path
        self.summary_path = summary_path
        self.attr_num = int(attr_num)
        self.model()

    def model(self):
            self.X = tf.placeholder(tf.float32, shape=(self.batch_size, self.crop_height, self.crop_width,
                                                 self.col_dim))
            self.Y = tf.placeholder(tf.float32, shape=(self.batch_size, self.attr_num))
            self.SL_back = tf.placeholder(tf.float32, shape=(self.batch_size, self.attr_num))
            self.net_logits = self.CNN(self.X, isTraining=True, dropout=0.5)
            self.eval_logits = self.CNN(self.X, isTraining=False, dropout=0.0)

            self.loss =tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net_logits,labels=self.Y)  #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net_logits,labels=self.Y))
            # tp = tf.metrics.true_positives(Y, net_logits)
            # tn = tf.metrics.true_negatives(Y, net_logits)
            self.weighted_loss = tf.reduce_mean(self.SL_back * self.loss)
            self.trainprob = tf.nn.sigmoid(self.eval_logits)
            self.trainacc = tf.equal(tf.round(self.trainprob), self.Y)
            # tf.summary.histogram("Accuracy", trainacc)
            # tr, trl, ev, evl, te, tel = all_data_pyarray()
            self.train_vars = tf.trainable_variables()
            self.saver = tf.train.Saver(max_to_keep=23)
            self.load_dataset_names()
            self.load_labels()
            self.train_data()

            self.loss_summary = tf.summary.scalar("Loss", self.loss)
            self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
            self. merged_summary = tf.summary.merge_all()

    def train_data(self):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,var_list=self.train_vars)
            try:
                self.sess.run(tf.global_variables_initializer())
            except:
                self.sess.run(tf.initialize_all_variables())
            train_iter = self.train_sim // self.batch_size
            s = time.time()
            #seed = 11
            for epo in range(self.epoch):
                total = 0.0
		with open('./output.txt', 'a') as f:
                    print >> f, 'Epoch Number:', epo
                print("\nEpoch Number = %d \n" % epo)
                #np.random.seed(seed + epo)
                #np.random.shuffle(self.train_image_names)
                #np.random.seed(seed + epo)
                #np.random.shuffle(self.train_lab)
                for iteration in range(train_iter):
                    x_batch = self.ready_batch(iteration, train_data)
                    x_batch = x_batch - np.mean(x_batch)
                    x_batch = self.random_crop(x_batch)
                    y_batch = self.train_lab[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                    back_propogation = self.bpropogation(y_batch)
                    _, curr_loss = self.sess.run([train_step, self.weighted_loss], feed_dict={self.X: x_batch, self.Y: y_batch, self.SL_back: back_propogation})

                    if iteration % 100 == 0:
                        print('Iteration %d\t%f' % (iteration, curr_loss))
		    	with open('./output.txt', 'a') as f:
				print >> f, 'Iteration' , iteration
				print >> f, '' , curr_loss
                ttrain_iter = self.train_sim // self.batch_size
                total = 0.0
                for i in range(ttrain_iter):
                    x_batch = self.ready_batch(i, train_data)
                    x_batch = self.random_crop(x_batch)
                    y_batch = self.train_lab[i * self.batch_size: (i + 1) * self.batch_size]
                    acc = self.sess.run([self.trainacc], feed_dict={self.X: x_batch, self.Y: y_batch})
                    total += np.mean(acc)
                final_acc = total / ttrain_iter
                print('Accuracy on Training set: %f' % final_acc)
                with open('./output.txt', 'a') as f:
                    print >> f, 'Accuracy Training:', final_acc


                eeval_iter = self.eval_sim // self.batch_size
                total = 0.0
                for i in range(eeval_iter):
                    x_batch = self.ready_batch(i, eval_data)
                    x_batch = self.random_crop(x_batch)
                    y_batch = self.val_lab[i * self.batch_size: (i + 1) * self.batch_size]
                    acc1 = self.sess.run([self.trainacc], feed_dict={self.X: x_batch, self.Y: y_batch})
                    total += np.mean(acc1)
                final_acc1 = total / eeval_iter
                print('Accuracy on Validation set: %f' % final_acc1)
                with open('./output.txt', 'a') as f:
                    print >> f, 'Accuracy Validation:', final_acc1
                self.saver.save(self.sess, self.model_path, global_step=epo)
            with open('./output.txt', 'a') as f:
                print('Total time: %f' % (time.time() - s))

    def CNN(self, inputs, isTraining, dropout):
        with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(inputs, 75, [7, 7], strides=(4, 4), padding='same', name='conv1')
            lrelu1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=isTraining))
            pool1 = tf.layers.max_pooling2d(lrelu1, [3, 3], strides=(2, 2), padding='valid', name='pool1')
            norm1 = tf.nn.local_response_normalization(pool1)

            conv2 = tf.layers.conv2d(norm1, 200, [5, 5], strides=(2, 2), padding='same', name='conv2')
            lrelu2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=isTraining))
            pool2 = tf.layers.max_pooling2d(lrelu2, [3, 3], strides=(2, 2), padding='valid', name='pool2')
            norm2 = tf.nn.local_response_normalization(pool2)

            conv3 = tf.layers.conv2d(norm2, 300, [3, 3], strides=(1, 1), padding='same', name='conv3')
            lrelu3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=isTraining))
            pool3 = tf.layers.max_pooling2d(lrelu3, [5, 5], strides=(2, 2), padding='valid', name='pool3')
            norm3 = tf.nn.local_response_normalization(pool3)

            fc1 = tf.layers.dense(norm3, units=512, name='fc1')
            lrelu4 = tf.nn.relu(tf.layers.batch_normalization(fc1, training=isTraining))
            drop1 = tf.layers.dropout(lrelu4, rate=dropout, training=isTraining)

            fc2 = tf.layers.dense(drop1, units=512, name='fc2')
            lrelu5 = tf.nn.relu(tf.layers.batch_normalization(fc2, training=isTraining))
            drop2 = tf.layers.dropout(lrelu5, rate=dropout, training=isTraining)

            flattened = tf.contrib.layers.flatten(drop2)

            fc3 = tf.layers.dense(flattened, units=self.attr_num, name='fc3')

            return fc3

    def load_dataset_names(self):
        eval_start = self.train_sim + 1
        test_start = eval_start + self.eval_sim
        test_end = test_start + self.test_sim
        self.train_image_names = np.arange(1, eval_start)
        self.val_image_names = np.arange(eval_start, test_start)
        self.test_image_names = np.arange(test_start, test_end)

    def load_labels(self):
        labs = pd.read_csv(self.label_dir, sep='\s+', skiprows=1, usecols=[n for n in range(1, 41)])
        labs = np.array(labs)
        labs[labs < 1] = 0
        self.train_lab = labs[0:self.train_sim]
        self.val_lab = labs[self.train_sim: self.train_sim + self.eval_sim]
        self.test_lab = labs[self.train_sim + self.eval_sim: len(labs)]

    def random_crop(self, batch_imgs):
        x = np.random.randint(0, (img_width - crop_width + 1))
        y = np.random.randint(0, (img_height - crop_height + 1))
        return batch_imgs[:, y:y + crop_height, x:x + crop_width, :]

    def ready_batch(self, iter, split_num):
        images = []
        if split_num == 0:
            img_nums = self.train_image_names[iter * self.batch_size: (iter + 1) * self.batch_size]
        elif split_num == 1:
            img_nums = self.val_image_names[iter * self.batch_size: (iter + 1) * self.batch_size]
        else:
            img_nums = self.test_image_names[iter * self.batch_size: (iter + 1) * self.batch_size]

        for i in img_nums:
            read_name = "%06d.jpg" % i
            img = mpimg.imread(os.path.join(self.data_dir, read_name))
            images.append(img)
        return np.asarray(images)

    def bpropogation(self, y_batch):
        final_count = [0] * 40  ##initialize list
        pos_dict = defaultdict(list)
        neg_dict = defaultdict(list)
        dist1 = 100
        dist2 = 50
        pos_weight = 0
        neg_weight = 0
        back_propogation = np.ones(shape=(100, 40))
        for i in range(self.batch_size - 1):
            for j in range(self.attr_num - 1):
                if y_batch[i][j] == 1:  # get all attributes for an image
                    final_count[j] = final_count[j] + 1  #total count for all attributes in a batch
                    pos_dict[j].append(i)  ##image_number for attribute being present
                else:
                    neg_dict[j].append(i)  ##image number for attribute not being present
        for n in range(self.attr_num - 1):
            neg_list = []
            pos_list = []
            if (final_count[n] < 50): #positive samples less
                neg_list = random.sample(neg_dict[n], dist2)
                for k in range(self.batch_size - 1):
                    if k in neg_list:
                        back_propogation[k][n] = 1
                    elif k in neg_dict[n] and k not in neg_list:
                        back_propogation[k][n] = 0
                    else:
                        pass
                    if final_count[n] != 0:
                        pos_weight = dist2 / final_count[n]
                    else:
                        pos_weight = 0
                    for l in range(self.batch_size - 1):
                        if l in pos_dict[n]:
                            back_propogation[l][n] = pos_weight
            else:
                remain = (dist1 - final_count[n]) #positive samples more
                pos_list = random.sample(pos_dict[n], dist2)
                for k in range(self.batch_size - 1):
                    if k in pos_list:
                        back_propogation[k][n] = 1
                    elif k in pos_dict[n] and k not in pos_list:
                        back_propogation[k][n] = 0
                    else:
                        pass
                    if remain != 0:
                        neg_weight = dist2 / remain
                    else:
                        neg_weight = 0
                    for l in range(self.batch_size - 1):
                        if l in neg_dict[n]:
                            back_propogation[l][n] = neg_weight
        del neg_list[:]
        del pos_list[:]
        pos_dict.clear()
        neg_dict.clear()
        return back_propogation


if __name__ == '__main__':
    mymodel(tf.Session(), 178, 218, 178, 178, 100, 162770, 19866, 19963, 22, 3, 0.001,
            "/home/nharpale/Desktop/Celeba/img_align_celeba/", "/home/nharpale/Desktop/Celeba/list_attr_celeba.txt",
            "/home/nharpale/PycharmProjects/CNN/Model/", "/home/nharpale/PycharmProjects/CNN/Summary/", 40)
