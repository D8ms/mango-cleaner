import random
import numpy as np
import tensorflow as tf
import argparse
import os
import glob
import cv2
from PIL import Image


from random import randint
from model import Model

BATCH_SIZE = 8
WIDTH = 975
HEIGHT = 1400
EPOCH = 10000

SWIDTH = WIDTH // 2
SHEIGHT = HEIGHT // 2

#load data in memory
raws = []
cleans = []
diffs = []
databasepath = "varric/train/"

#norm, resize, reshape
def normrere(img):
    renormed = (img / 255.0) - 0.5
    return cv2.resize(renormed, (SWIDTH, SHEIGHT), interpolation=cv2.INTER_AREA)[:,:,None]

for i in range(1, 10):
    raw_paths = glob.glob("varric/train/" + str(i) + "r/*.png")
    clean_paths = [p.replace('r/', 'c/') for p in raw_paths]
    for i in range(len(raw_paths)):
        if os.path.exists(clean_paths[i]):
            raw_data = normrere(np.asarray(Image.open(raw_paths[i])))
            clean_data = normrere(np.asarray(Image.open(clean_paths[i])))
            diff_data = np.abs(raw_data - clean_data)
            raws.append(raw_data)
            cleans.append(clean_data)
            diffs.append(diff_data)

raws = np.array(raws)
cleans = np.array(cleans)
diffs = np.array(diffs)
avail_idxs = list(range(len(raws)))

tf.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.Session(config=tf.config) as sess:
    model = Model(SWIDTH, SHEIGHT)
    sess.run(tf.global_variables_initializer())
    summary_recorder = tf.summary.FileWriter("summary", sess.graph)
    for epoch in range(EPOCH):
        samp_idxes = random.sample(avail_idxs, BATCH_SIZE)
        batch_raws = raws[samp_idxes]
        batch_cleans = cleans[samp_idxes]
        batch_diffs = diffs[samp_idxes]
        error, summary = model.train(sess, batch_raws, batch_cleans, batch_diffs)
        summary_recorder.add_summary(summary, epoch)
        print(epoch, error)
