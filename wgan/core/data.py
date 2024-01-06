
__all__ = ["prepare_real", "stratified_train_val_test_splits"]

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from itertools import compress

import pandas as pd
import numpy as np
import pickle
import os


def prepare_real( data_info ) -> pd.DataFrame:

    dataset    = data_info['dataset']
    tag        = data_info['tag']
    table_name = data_info['csv']
    split_name = data_info['splits']

    # load raw data
    data       = pd.read_csv( f"{DATA_DIR}/{dataset}/{tag}/raw/{table_name}" )

    # load crossval splits
    with open( f"{DATA_DIR}/{dataset}/{tag}/raw/{split_name}", 'rb') as f:
        splits = pickle.load(f) 

    # append path
    def _append_basepath(row):
        return f"{DATA_DIR}/{dataset}/{tag}/raw/{row.path}"


    data['image_path'] = data.apply(_append_basepath, axis='columns')
    data["name"]       = dataset
    data["type"]       = "real"
    data_list = []

    for test in range(10):
        for sort in range(9):
            trn_idx = splits[test][sort][0]
            val_idx = splits[test][sort][1]
            tst_idx = splits[test][sort][2]
            train = data.loc[trn_idx]
            train["set"] = "train"
            train["test"] = test
            train["sort"] = sort
            data_list.append(train)
            valid = data.loc[val_idx]
            valid["set"] = "val"
            valid["test"] = test
            valid["sort"] = sort
            data_list.append(valid)
            test = data.loc[tst_idx]
            test["set"] = "test"
            test["test"] = test
            test["sort"] = sort
            data_list.append(test)

    return pd.concat(data_list)





def stratified_train_val_test_splits(df, n_folds,seed=512):
    cv_index = {'train_val': [], 'test': []}
    cv_train_test = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    sorts_train_test = []
    for train_val_idx, test_idx in cv_train_test.split(df.values, df.target.values):
        cv_index['test'].append(test_idx)
        cv_index['train_val'].append(train_val_idx)
    fold_idx = set(np.arange(0, n_folds, 1))
    for fold in fold_idx:
        val_idx = list(fold_idx - set([fold]))
        print('bins selected for val: ' + str(val_idx))
        sorts = []
        for i in tqdm(val_idx):
            l_val_idx = cv_index['test'][i]
            flt = ~pd.Series(cv_index['train_val'][fold]).isin(l_val_idx).values
            l_train_idx = np.array(list(compress(cv_index['train_val'][fold], flt)))
            sorts.append((l_train_idx, l_val_idx, cv_index['test'][fold]))
        sorts_train_test.append(sorts)
    return sorts_train_test




def generate_samples( model_generator, output_path, image_format, nblocks=50, batch_size = 64, seed=512):
    tf.random.set_seed(seed)
    image_id = 0
    for idx in tqdm( range(nblocks) , desc="generating..."):
        z = tf.random.normal( (batch_size,100) )
        img = model( z ).numpy()
        for im in img:
          image_path = image_format.format(image_id='%06d'%image_id)
          cv2.imwrite(output_path'/'+image_path, im*255)