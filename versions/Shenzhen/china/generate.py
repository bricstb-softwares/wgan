#!/usr/bin/env python3
import argparse
import tensorflow as tf
tf.config.run_functions_eagerly(False)
from rxwgan.models import Generator_v2
import os, glob, cv2, json
from tqdm import tqdm
import numpy as np



tests = 10
sorts = 9
model_path  = 'user.jodafons.task.Shenzhen.wgan.v2_tb.r1/job.test_%d.sort_%d/generator_trained.h5'
output_path = 'user.jodafons.images.Shenzhen.wgan.v2_tb.r1.sample_3k/job.test_%d.sort_%d'
label       = 'tb'
tf.random.set_seed(512)
batch_size  = 64
epoch       = 50

for test in range(tests):
    for sort in range(sorts):

      print('Generate images to test %d sort %d'%(test,sort))
      if not os.path.exists(output_path): os.makedirs(output_path%(test,sort))
      model = Generator_v2(model_path%(test,sort)).model

      img_idx = 0

      for idx in tqdm( range(epoch) ):
        z = tf.random.normal( (batch_size,100) )
        img = model( z ).numpy()
        
        for im in img:
          imgpath = output_path%(test,sort) + '/fake.%s.test_%s.sort_%d.%06d.png'%(label, test,sort, img_idx)
          cv2.imwrite(imgpath, im*255)
          img_idx +=1




