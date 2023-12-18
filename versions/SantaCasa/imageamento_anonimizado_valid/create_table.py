
import os, glob,json
import pandas as pd


basepath      = '/home/brics/public/brics_data/SantaCasa/imageamento_anonimizado_valid/fake_images'
dataset_name  = 'user.joao.pinto.task.SantaCasa.imageamento_anonimizado_valid.wgan_v2_notb.r4.samples'
 


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

      print(f'Generate images to test {test} sort {sort}')

      path = basepath + '/' + dataset_name + '/' + folder
      for f in sorted(glob.glob(path + '/*.png')):
          project_id = f.split('/')[-1].replace('.png','')
          d['sort'].append(sort)
          d['test'].append(test)
          d['type'].append('test')
          d['metadata'].append({'has_tb':target, 'data_aquisition':date_acquisition})
          d['dataset_name'].append(dataset_name)
          d['dataset_type'].append('synthetic')
          d['project_id'].append(project_id)
          d['image_path'].append(f)


table = pd.DataFrame(d)
table.to_csv(dataset_name+'.csv')

#dd0d3688ff707de5b3d9f6aaf488067a7a9bd4682e79267c048141df8d8b753e  imageamento_anonimizado_valid.test_00.sort00.ID000000.png
