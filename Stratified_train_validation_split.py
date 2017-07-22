"""
Split train data into development and validation data sets by stratifying the y variable. Here y is a continuous variable.
"""

import numpy as np
import pandas as pd
import random
import pickle

def stratified_kfold(data, nsplit = 5):
    num_per_fold = data.shape[0]//nsplit
    num_folds_1less = data.shape[0]%nsplit
    rand_delta_lst = list(range(nsplit))
    
    lst_kfold = [None]*nsplit
    for i in range(nsplit):
        lst_kfold[i] = []
    
    for i in range(num_per_fold):
        random.shuffle(rand_delta_lst)
        for j in range(nsplit):
            lst_kfold[j].append(int(data.iloc[(i*nsplit + rand_delta_lst[j]), 2]))
    
    rand_delta_lst = list(range(num_folds_1less))
    random.shuffle(rand_delta_lst)
    for k in range(num_folds_1less):
        lst_kfold[k].append(int(data.iloc[(num_per_fold*nsplit + rand_delta_lst[k]), 2]))
        
    val_ind = [None]*nsplit
    dev_ind = [None]*nsplit
    for i in range(nsplit):
        val_ind[i] = []
        dev_ind[i] = []
    for l in range(nsplit):
        val_ind[l] = lst_kfold[l]
        for p in range(nsplit):
            if p != l:
                dev_ind[l] += lst_kfold[p]
                
    return dev_ind, val_ind

def n_stratified_k_fold(data, nsplit, ntimes):
    skf_result = [None] * ntimes
    for i in range(ntimes):
        dev_ind, val_ind = stratified_kfold(data, nsplit)
        skf_result[i] = [dev_ind, val_ind]
    return skf_result

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    IDny = train.loc[:, ['ID', 'y']]
    IDny['ind'] = train.index.values
    IDny.sort_values('y', axis=0,  inplace=True)
    
    skf_result = n_stratified_k_fold(IDny.iloc[:, :], 10, 20)
    with open('dev_val_ind_10f_20_times.pickle', 'wb') as handle:
        pickle.dump(skf_result, handle, protocol=pickle.HIGHEST_PROTOCOL)