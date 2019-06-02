import random
import numpy as np
import tensorflow as tf
import argparse
import os
import glob
from PIL import Image


from random import randint
from model import Model

BATCH_SIZE = 8
WIDTH = 975
HEIGHT = 1400
EPOCH = 10000

#load data in memory
raws = []
cleans = []
diffs = []
weights = []
databasepath = "varric/train/"
for i in range(1, 10):
    raw_paths = glob.glob("varric/train/" + str(i) + "r/*.png")
    clean_paths = [p.replace('r/', 'c/') for p in raw_paths]
    for i in range(len(raw_paths)):
        if os.path.exists(clean_paths[i]):
            raw_data = (np.asarray(Image.open(raw_paths[i])) / 255.0) - 0.5
            clean_data = (np.asarray(Image.open(clean_paths[i])) / 255.0) - 0.5
            diff_data = np.abs(raw_data - clean_data)
            weight_data = np.clip(np.abs(raw_data - clean_data) * 5, 0.1, 1)
            raws.append(raw_data[:,:,None])
            cleans.append(clean_data[:,:,None])
            diffs.append(diff_data[:,:,None])
            weights.append(weight_data[:,:,None])

raws = np.array(raws)
cleans = np.array(cleans)
diffs = np.array(diffs)
weights = np.array(weights)
avail_idxs = list(range(len(raws)))

tf.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.Session(config=tf.config) as sess:
    model = Model(WIDTH, HEIGHT)
    sess.run(tf.global_variables_initializer())
    summary_recorder = tf.summary.FileWriter("summary", sess.graph)
    for epoch in range(EPOCH):
        samp_idxes = random.sample(avail_idxs, BATCH_SIZE)
        batch_raws = raws[samp_idxes]
        batch_cleans = cleans[samp_idxes]
        batch_weights = weights[samp_idxes]
        batch_diffs = diffs[samp_idxes]
        error, summary = model.train(sess, batch_raws, batch_cleans, batch_weights, batch_diffs)
        summary_recorder.add_summary(summary, epoch)
        print(epoch, error)
