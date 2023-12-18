#!/usr/bin/env python3

import os
import tensorflow as tf 


import pandas as pd
import numpy as np
import argparse, traceback, json, os, pickle
from rxwgan.models import *
from rxwgan.wgangp import wgangp_optimizer
from colorama import Back, Fore
#from rxwgan import stratified_train_val_test_splits
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from loguru import logger



#
# Input args (mandatory!)
#
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-v','--volume', action='store', 
    dest='volume', required = False, default=os.getcwd(),
    help = "volume path")

parser.add_argument('-j','--job', action='store', 
    dest='job', required = True, default = None, 
    help = "job configuration.")

parser.add_argument('--disable_wandb', action='store_true', 
    dest='disable_wandb', required = False, 
    help = "Disable wandb report")

parser.add_argument('--username_wandb', action='store', 
    dest='username_wandb', required = False, default='jodafons',
    help = "wanddb username.")

parser.add_argument('--wandb_taskname', action='store', 
    dest='wandb_taskname', required = False, default='task',
    help = "wandb task name")

parser.add_argument('--is_tb', action='store_false', help = "Is TB?", default=False)



import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()




# getting parameters from the server
device       = int(os.environ['CUDA_VISIBLE_DEVICES'])
workarea     = os.environ['JOB_WORKAREA']
job_id       = os.environ['JOB_ID']
run_id       = os.environ['MLFLOW_RUN_ID']
tracking_url = os.environ['MLFLOW_URL']
dry_run      = os.environ['JOB_DRY_RUN'] == 'true'


if device>=0:
    logger.info(f"running in gpu device: {device}")
    tf.config.experimental.set_memory_growth(device, True)



try:

    #
    # Start your job here
    #

    job          = json.load(open(args.job, 'r'))
    sort         = job['sort']
    test         = job['test']
    dataset_path = job['dataset']
    splits_path  = job['splits']
    seed         = job['seed']
    epochs       = 1 if dry_run else job['epochs']
    batch_size   = job['batch_size']
    target       = args.is_tb

    logger.info( 'Is TB: ' + 'Yes' if args.is_tb else 'No')




    #
    # Check if we need to recover something...
    #
    if os.path.exists(args.volume+'/checkpoint.json'):
        logger.info('reading from last checkpoint...')
        checkpoint = json.load(open(args.volume+'/checkpoint.json', 'r'))
        history = json.load(open(checkpoint['history'], 'r'))
        critic = tf.keras.models.load_model(checkpoint['critic'])
        generator = tf.keras.models.load_model(checkpoint['generator'])
        start_from_epoch = checkpoint['epoch'] + 1
        logger.info('starts from %d epoch...'%start_from_epoch)
    else:
        start_from_epoch= 0
        # create models
        critic = Critic_v2().model
        generator = Generator_v2().model
        history = None

    height = critic.layers[0].input_shape[0][1]
    width  = critic.layers[0].input_shape[0][2]

    # Read dataframe
    dataframe = pd.read_csv(dataset_path)
    #splits = stratified_train_val_test_splits(dataframe,10,seed)[test]
    splits = pickle.load(open(splits_path, 'rb'))
    splits = splits[test]

    
    training_data   = dataframe.iloc[splits[sort][0]]
    validation_data = dataframe.iloc[splits[sort][1]]
    training_data   = training_data.loc[training_data.target==target]
    validation_data = validation_data.loc[validation_data.target==target]


    logger.info(training_data.shape)
    logger.info(validation_data.shape)

    extra_d = {'sort' : sort, 'test':test, 'target':target, 'seed':seed}

        
    datagen = ImageDataGenerator( rescale=1./255 )

    train_generator = datagen.flow_from_dataframe(training_data, directory = None,
                                                  x_col = 'image_path', 
                                                  y_col = 'target',
                                                  batch_size = batch_size,
                                                  target_size = (height,width), 
                                                  class_mode = 'raw', 
                                                  shuffle = True,
                                                  color_mode = 'grayscale')

    val_generator   = datagen.flow_from_dataframe(validation_data, directory = None,
                                                  x_col = 'image_path', 
                                                  y_col = 'target',
                                                  batch_size = batch_size,
                                                  class_mode = 'raw',
                                                  target_size = (height,width),
                                                  shuffle = True,
                                                  color_mode = 'grayscale',
                                                  )

    #
    # Create optimizer
    #
    optimizer = wgangp_optimizer( critic, generator, 
                                  n_discr = 0,
                                  history = history,
                                  start_from_epoch = start_from_epoch,
                                  max_epochs = epochs, 
                                  volume = args.volume,
                                  disp_for_each = 10, 
                                  #use_gradient_penalty = False,
                                  save_for_each = 200,
                                  run_id = run_id,
                                  tracking_url = tracking_url)

    
    try:
        if args.disable_wandb or is_test:
            wandb=None
        else:
            import wandb
            name = 'test_%d_sort_%d'%(test,sort)
            wandb.init(project=args.wandb_taskname,
                       name=name,
                       id=name,
                       entity=args.username_wandb)

    except:
        wandb=None


    if is_test:
        sys.exit(0)


    # Run!
    history = optimizer.fit( train_generator , val_generator, extra_d=extra_d, wandb=wandb )

    # in the end, save all by hand
    critic.save(args.volume + '/critic_trained.h5')
    generator.save(args.volume + '/generator_trained.h5')
    with open(args.volume+'/history.json', 'w') as handle:
      json.dump(history, handle,indent=4)

    #
    # End your job here
    #

    sys.exit(0)

except  Exception as e:
    traceback.print_exc()
    sys.exit(1)
