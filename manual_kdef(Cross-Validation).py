import shutil
import os

import tensorflow as tf
# import shutil
# from tensorflow.python.saved_model import tag_constants
from sklearn.model_selection import KFold
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
train_data = 0
test_data = 2
eval_data = 1
split_size = 10


src ="/home/nharpale/Desktop/kdef/kdefinside/"
dest = "/home/nharpale/Desktop/kdef/kdefinside/"
counter = 1
files = os.listdir(src)



class mymodel():
    def __init__(self, sess, img_width, img_height, crop_width, crop_height, batch_size, train_sim,  #eval_sim,test_sim,
                  epoch, col_dim, learning_rate, data_dir, label_dir, model_path, summary_path, attr_num):
        self.sess = sess
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.crop_width = int(crop_width)
        self.crop_height = int(crop_height)
        self.batch_size = int(batch_size)
        self.train_sim = int(train_sim)
        #self.eval_sim = int(eval_sim)
        #self.test_sim = int(test_sim)
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

            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net_logits,labels=self.Y)  #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net_logits,labels=self.Y))
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
            #self.train_data()
            self.cross_validate()

            self.loss_summary = tf.summary.scalar("Loss", self.loss)
            self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
            self. merged_summary = tf.summary.merge_all()

    def load_label(self, filename, fp):
        lines = fp.split("\n")
        for line in lines[0:-1]:
            values = line.strip().split()
            if filename == values[0]:
                labels = [float(v) for v in values[1:]]
        return labels


    def cross_validate(self):
        train_x = []
        train_y = []
        val_x = []
        val_y = []
        counter = 1
        with tf.gfile.Open(self.label_dir) as f:
            fp = f.read()
        for f in files:
            if counter != 1:
                dest_name = src + "%s/" % (counter)
                file_dest = os.listdir(dest_name)
                for t in file_dest:
                    train_x.append(t)
                    label = self.load_label(t, fp)
                    train_y.append(label)
            else:
                dest_name1 = src + "%s/" % (counter)
                file_dest = os.listdir(dest_name1)
                for t1 in file_dest:
                    val_x.append(t1)
                    label1 = self.load_label(t1, fp)
                    val_y.append(label1)
            counter = counter + 1
            if counter == 11:
                break
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_y[train_y < 3.5] = 0
        train_y[train_y >= 3.5] = 1

        val_x = np.array(val_x)
        val_y = np.array(val_y)
        val_y[val_y < 3.5] = 0
        val_y[val_y >= 3.5] = 1

        self.train_data(train_x, train_y, val_x, val_y)


    def train_data(self, train_x, train_y, val_x, val_y):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,var_list=self.train_vars)
            try:
                self.sess.run(tf.global_variables_initializer())
            except:
                self.sess.run(tf.initialize_all_variables())
            train_iter = train_x.shape[0] // self.batch_size
            s = time.time()
            split_data = 0.0
            total = 0.0
            split_n = 10
            with open('./outputkdef.txt', 'a') as f:
                print >> f, 'Split Number:', split_n
            for epo in range(self.epoch):
                total = 0.0
                print("\nEpoch Number = %d \n" % epo)
                for iteration in range(train_iter):
                    x_batch = self.ready_batch(iteration,train_data, train_x)
                    #x_batch = x_batch - np.mean(x_batch)
                    #x_batch = self.random_crop(x_batch)
                    # total = total + x_batch.shape[0]
                    # print(total)
                    y_batch = train_y[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                    back_propogation = self.bpropogation(y_batch)
                    _, curr_loss = self.sess.run([train_step, self.loss], feed_dict={self.X: x_batch, self.Y: y_batch,self.SL_back: back_propogation})

                    # if iteration % 4 == 0:
                    #     print('Iteration %d\t%f' % (iteration, curr_loss))

            val_iter = val_x.shape[0] // self.batch_size
            for itera in range(val_iter):
                x_batch = self.ready_batch(itera, eval_data, val_x)
                #x_batch = self.random_crop(x_batch)
                y_batch = val_y[itera * self.batch_size: (itera + 1) * self.batch_size]
                acc = self.sess.run([self.trainacc], feed_dict={self.X: x_batch, self.Y: y_batch})
                acc = np.mean(acc, axis=1)
                split_data += acc
            final_acc = split_data / val_iter
            print("Accuracy")
            print(final_acc)
            self.saver.save(self.sess, self.model_path, global_step=epo)
            with open('./outputkdef.txt', 'a') as f:
             	print >> f, 'Accuracy on Validation Set:', final_acc
            #self.saver.save(self.sess, self.model_path, global_step=epo)


    def CNN(self, inputs, isTraining, dropout):
        with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(inputs, 45, [7, 7], strides=(4, 4), padding='same', name='conv1')
            lrelu1 = tf.nn.relu(tf.layers.batch_normalization(conv1, training=isTraining))
            pool1 = tf.layers.max_pooling2d(lrelu1, [3, 3], strides=(2, 2), padding='valid', name='pool1')
            norm1 = tf.nn.local_response_normalization(pool1)

            conv2 = tf.layers.conv2d(norm1, 85, [5, 5], strides=(2, 2), padding='same', name='conv2')
            lrelu2 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=isTraining))
            pool2 = tf.layers.max_pooling2d(lrelu2, [3, 3], strides=(2, 2), padding='valid', name='pool2')
            norm2 = tf.nn.local_response_normalization(pool2)

            conv3 = tf.layers.conv2d(norm2, 125, [3, 3], strides=(1, 1), padding='same', name='conv3')
            lrelu3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=isTraining))
            pool3 = tf.layers.max_pooling2d(lrelu3, [5, 5], strides=(2, 2), padding='valid', name='pool3')
            norm3 = tf.nn.local_response_normalization(pool3)

            fc1 = tf.layers.dense(norm3, units=195, name='fc1')
            lrelu4 = tf.nn.relu(tf.layers.batch_normalization(fc1, training=isTraining))
            drop1 = tf.layers.dropout(lrelu4, rate=dropout, training=isTraining)

            #fc2 = tf.layers.dense(drop1, units=256, name='fc2')
            #lrelu5 = tf.nn.relu(tf.layers.batch_normalization(fc2, training=isTraining))
            #drop2 = tf.layers.dropout(lrelu5, rate=dropout, training=isTraining)

            flattened = tf.contrib.layers.flatten(drop1)

            fc3 = tf.layers.dense(flattened, units=self.attr_num, name='fc3')

            return fc3

    def load_dataset_names(self):
        eval_start = self.train_sim + 1
        self.train_image_names = np.arange(1, eval_start)

    def load_labels(self):
        labs = pd.read_csv(self.label_dir, sep='\s+', usecols=[n for n in range(1, 4)])
        labs = np.array(labs)
        labs[labs < 3.5] = 0
        labs[labs >= 3.5] = 1
        self.train_lab = labs[0:self.train_sim]
        #self.val_lab = labs[self.train_sim: self.train_sim + self.eval_sim]
        #self.test_lab = labs[self.train_sim + self.eval_sim: len(labs)]

    def random_crop(self, batch_imgs):
        x = np.random.randint(0, (self.img_width - self.crop_width + 1))
        y = np.random.randint(0, (self.img_height - self.crop_height + 1))
        return batch_imgs[:, y:y + self.crop_height, x:x + self.crop_width, :]

    def ready_batch(self, iter, split_num, data):
        images = []
        if split_num == 0:
            img_nums = data[iter * self.batch_size: (iter + 1) * self.batch_size]
        #elif split_num == 1:
            #img_nums = self.val_image_names[iter * self.batch_size: (iter + 1) * self.batch_size]
        else:
            img_nums = data[iter * self.batch_size: (iter + 1) * self.batch_size]
        for i in img_nums: ##get the image name and add to imread
            img = mpimg.imread(os.path.join(self.data_dir, i))
            images.append(img)
        return np.asarray(images)

    def bpropogation(self, y_batch):
        final_count = [0] * 3  ##initialize list
        pos_dict = defaultdict(list)
        neg_dict = defaultdict(list)
        dist1 = 26
        dist2 = 13
        pos_weight = 0
        neg_weight = 0
        back_propogation = np.ones(shape=(26, 3))
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
            if (final_count[n] < 13): #positive samples less
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
    mymodel(tf.Session(), 178, 178, 178, 178, 26, 1152, 22, 3, 0.003,
            "/home/nharpale/Desktop/KDEF/178_Resized_Images/", "/home/nharpale/Desktop/KDEF/Labels/labelskdef.txt",
            "/home/nharpale/PycharmProjects/kdef/Modelkdef", "/home/nharpale/PycharmProjects/kdef/Summarykdef", 3)
