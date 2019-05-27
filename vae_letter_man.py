# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import struct
from scipy.stats import norm
from letter_man import read_data_sets

letter = read_data_sets('.\\letter_man\\letter')
man = read_data_sets('.\\letter_man\\man')

letter_edge_groundtruth_path = '.\\letter_man\\letter\\letter_edge_groundtruth-images.idx3-ubyte'
f1 = open(letter_edge_groundtruth_path , 'rb')
buf1 = f1.read()
image_index = 0
image_index += struct.calcsize('>IIII')
letter_edge_groundtruth = struct.unpack_from('>16384B', buf1, image_index)
letter_edge_groundtruth = np.reshape(letter_edge_groundtruth,(1,16384))

man_edge_groundtruth_path = '.\\letter_man\\man\\man_edge_groundtruth-images.idx3-ubyte'
f1 = open(man_edge_groundtruth_path , 'rb')
buf1 = f1.read()
image_index = 0
image_index += struct.calcsize('>IIII')
man_edge_groundtruth = struct.unpack_from('>16384B', buf1, image_index)
man_edge_groundtruth = np.reshape(man_edge_groundtruth,(1,16384))

abc_edge_groundtruth_path = '.\\letter_man\\abc\\abc_edge_groundtruth-images.idx3-ubyte'
f1 = open(abc_edge_groundtruth_path , 'rb')
buf1 = f1.read()
image_index = 0
image_index += struct.calcsize('>IIII')
abc_edge_groundtruth = struct.unpack_from('>16384B', buf1, image_index)
abc_edge_groundtruth = np.reshape(abc_edge_groundtruth,(1,16384))
#############################################

n_input = 16384
n_hidden_1 = 256
n_hidden_2 = 2

x = tf.placeholder(tf.float32, [None, n_input])
zinput = tf.placeholder(tf.float32, [None, n_hidden_2])
groundtruth = tf.placeholder(tf.float32, [None, n_input])

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.001)),
    'b1': tf.Variable(tf.zeros([n_hidden_1])),

    'mean_w1': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.001)),
    'mean_b1': tf.Variable(tf.zeros([n_hidden_2])),

    'log_sigma_w1': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.001)),
    'log_sigma_b1': tf.Variable(tf.zeros([n_hidden_2])),
  
    'w2': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.001)),
    'b2': tf.Variable(tf.zeros([n_hidden_1])),

    'w3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.001)),
    'b3': tf.Variable(tf.zeros([n_input])), 
}

h1=tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), weights['b1']))
z_mean = tf.add(tf.matmul(h1, weights['mean_w1']), weights['mean_b1'])
z_log_sigma_sq = tf.add(tf.matmul(h1, weights['log_sigma_w1']), weights['log_sigma_b1'])

#tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值。
eps = tf.random_normal(tf.stack([tf.shape(h1)[0], n_hidden_2]), 0, 1, dtype = tf.float32)
z =tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) #采样变量z

h2=tf.nn.relu(tf.matmul(z, weights['w2'])+ weights['b2'])
reconstruction = tf.matmul(h2, weights['w3'])+ weights['b3']

h2out=tf.nn.relu(tf.matmul(zinput, weights['w2'])+ weights['b2'])
reconstructionout = tf.matmul(h2out, weights['w3'])+ weights['b3']

reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction, groundtruth), 2.0))
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

training_epochs = 1
batch_size = 16
display_step = 3

model_path = ".\\saved_model\\vae_letter_man.ckpt"

if not os.path.exists(model_path):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(letter.train.num_examples/batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = letter.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost], feed_dict={x: batch_xs, groundtruth: letter_edge_groundtruth})

            for i in range(total_batch):
                batch_xs, batch_ys = man.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost], feed_dict={x: batch_xs, groundtruth: man_edge_groundtruth})

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
        print("完成!")
        save_path = saver.save(sess, model_path) # 保存模型
        print("Model saved in file: %s" % save_path)
        writer = tf.summary.FileWriter(".\\graph\\", sess.graph) #第一个参数指定生成文件的目录。

print("Starting 2nd session...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 初始化变量
    saver.restore(sess, model_path) # 恢复模型变量

    f, a = plt.subplots(2, 3, figsize=(4, 3)) #dpi=80
    pred_letter = sess.run(reconstruction, feed_dict={x: np.reshape(letter.test.images[2], (1, 16384))})
    pred_man = sess.run(reconstruction, feed_dict={x: np.reshape(man.test.images[2], (1, 16384))})
    pred_abc = sess.run(reconstruction, feed_dict={x: abc_edge_groundtruth})
    a[0][0].imshow(np.reshape(letter.test.images[2], (128, 128)))
    a[0][1].imshow(np.reshape(man.test.images[2], (128, 128)))
    a[0][2].imshow(np.reshape(abc_edge_groundtruth, (128, 128)))
    a[1][0].imshow(np.reshape(pred_letter, (128, 128)))
    a[1][1].imshow(np.reshape(pred_man, (128, 128)))
    a[1][2].imshow(np.reshape(pred_abc, (128, 128)))
    plt.savefig('.\\sample1.png')

    n = 15
    digit_size = 128
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) #希望正态分布取样
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = sess.run(reconstructionout,feed_dict={zinput:z_sample})
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(25, 25))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig('.\\sample2.png')
    plt.show()