#!/usr/bin/env python


import tensorflow as tf 
import pandas as pd
import numpy as np
import argparse, traceback, json, os, pickle, sys

from wgan.models import *
from wgan.core   import prepare_data
from wgan import wgangp_optimizer, prepare_data, generate_samples
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from loguru import logger



#
# train 
#
def run():


    #
    # Input args (mandatory!)
    #
    parser = argparse.ArgumentParser(description = '', add_help = False)
    parser = argparse.ArgumentParser()

  
    parser.add_argument('-c','--card', action='store', 
        dest='card', required = True,
        help = "train card")

    parser.add_argument('-j','--job', action='store', 
        dest='job', required = True, 
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

    parser.add_argument('--dry_run', action='store_true', 
        dest='dry_run', required = False, 
        help = "Dry run it!")

    if len(sys.argv)==1:
      parser.print_help()
      sys.exit(1)

    args = parser.parse_args()

    # getting parameters from the server
    device       = int(os.environ.get('CUDA_VISIBLE_DEVICES','-1'))
    workarea     = os.environ.get('JOB_WORKAREA'    , os.getcwd())
    job_id       = os.environ.get('JOB_ID'          , -1)
    run_id       = os.environ.get('MLFLOW_RUN_ID'   ,'' )
    tracking_url = os.environ.get('MLFLOW_URL'      , '')
    dry_run      = os.environ.get('JOB_DRY_RUN'     ,'false') == 'true'
    dry_run      = dry_run or args.dry_run

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
    
        params       = json.load(open(args.card, 'rb'))
        target       = params['label'] == "tb"
        data_info    = params['dataset_info']
        sample_size  = params['sample_size']
        batch_size   = params['batch_size']
        epochs       = params['epochs'] if not dry_run else 1

        #
        # Check if we need to recover something...
        #
        if os.path.exists(workarea+'/checkpoint.json'):
            logger.info('reading from last checkpoint...')
            checkpoint = json.load(open(workarea+'/checkpoint.json', 'r'))
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


   

        logger.info('prepare data...')
        data            = prepare_data( data_info )
        training_data   = data.loc[(data.test==test)&(data.sort==sort)&(data.set=='train')]
        validation_data = data.loc[(data.test==test)&(data.sort==sort)&(data.set=='val')  ]
        training_data   = training_data.loc[training_data.target==target]
        validation_data = validation_data.loc[validation_data.target==target]

        logger.info(training_data.shape)
        logger.info(validation_data.shape)


        extra_d = {'sort' : sort, 'test':test, 'target':target, 'params' : params}

        logger.info('prepare dataset...')
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
        logger.info("prepare optimizer...")
        #
        # Create optimizer
        #
        optimizer = wgangp_optimizer( critic, generator, 
                                      n_discr = 0,
                                      history = history,
                                      start_from_epoch = start_from_epoch,
                                      max_epochs = epochs, 
                                      volume = workarea,
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


       

        logger.info('fit...')
        # Run!
        history = optimizer.fit( train_generator , val_generator, extra_d=extra_d, wandb=wandb )


        logger.info('saving...')
        # in the end, save all by hand
        critic.save(workarea + '/critic_trained.h5')
        generator.save(workarea + '/generator_trained.h5')
        with open(workarea+'/history.json', 'w') as handle:
          json.dump(history, handle,indent=4)


        #
        # generate samples
        #
        if sample_size > 0:
          logger.info('prepare sample generation...')
          os.makedirs(workarea+'/samples', exist_ok=True)
          nblocks = int(sample_size/batch_size) if not dry_run else 1
          image_output = f'test_{test}.sort_{sort}'
          generate_samples( generator ,workarea+'/samples', nblocks = nblocks )


        #
        # End your job here
        #

        sys.exit(0)

    except  Exception as e:
        traceback.print_exc()
        sys.exit(1)




if __name__ == "__main__":
    run()