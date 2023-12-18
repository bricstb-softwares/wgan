
__all__ = ['stratified_train_val_test_splits']

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from itertools import compress

import pandas as pd
import numpy as np

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

