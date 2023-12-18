
import argparse
import tensorflow as tf
tf.config.run_functions_eagerly(False)
from rxwgan.models import Generator_v2
import os, glob, cv2, json
from tqdm import tqdm
import numpy as np
import pandas as pd


basepath      = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/fake_images'
tuning_path   = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/models/user.joao.pinto.task.SantaCasa.imageamento_anonimizado_valid.wgan_v2_notb.r4'
dataset_name  = 'user.joao.pinto.task.SantaCasa.imageamento_anonimizado_valid.wgan_v2_notb.r4.sample'
image_format  = 'imageamento_anonimizado_valid.test_{test}.sort_{sort}.ID{project_id}.png'

tf.random.set_seed(512)
batch_size       = 64
epoch            = 50
tests            = 10
sorts            = 9
target           = False
date_acquisition = '2023-02-28'


for test in range(tests):
    for sort in range(sorts):

      folder = f'job.test_{test}.sort_{sort}'
      os.makedirs(dataset_name+'/'+folder, exist_ok=True)

      print(f'Generate images to test {test} sort {sort}')

      gen_path = tuning_path + f'/job.test_{test}.sort_{sort}/generator_trained.h5'
      model = Generator_v2(gen_path).model

      project_id = 0

      for idx in tqdm( range(epoch) ):
        z = tf.random.normal( (batch_size,100) )
        img = model( z ).numpy()
        for im in img:
          image_path = image_format.format(test=test, sort=sort, project_id='%06d'%project_id)
          cv2.imwrite(dataset_name+'/'+folder+'/'+image_path, im*255)
         

