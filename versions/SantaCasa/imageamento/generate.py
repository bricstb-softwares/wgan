
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
dataset_name  = 'user.joao.pinto.task.SantaCasa.imageamento_anonimizado_valid.wgan_v2_notb.r4.sample_short'
image_format  = 'imageamento_anonimizado_valid.test_{test}.sort_{sort}.ID{project_id}.png'

tf.random.set_seed(512)
batch_size       = 64
epoch            = 50
tests            = 10
sorts            = 9
target           = False
date_acquisition = '2023-02-28'

d =  {
      'dataset_name': [],
      'dataset_type': [],
      'project_id': [],
      'image_path':[],
      'metadata': [],
      'test':[],
      'sort':[],
      'type':[],
      }


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
          d['sort'].append(sort)
          d['test'].append(test)
          d['type'].append('test')
          d['metadata'].append({'has_tb':target, 'data_aquisition':date_acquisition})
          d['dataset_name'].append(dataset_name)
          d['dataset_type'].append('synthetic')
          d['project_id'].append(project_id)
          d['image_path'].append(basepath+'/'+dataset_name+'/'+folder+'/'+image_path)
          project_id +=1


table = pd.DataFrame(d)
table.to_csv(dataset_name+'.csv')

#dd0d3688ff707de5b3d9f6aaf488067a7a9bd4682e79267c048141df8d8b753e  imageamento_anonimizado_valid.test_00.sort00.ID000000.png
