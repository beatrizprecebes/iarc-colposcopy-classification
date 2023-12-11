import numpy as np
import random

def rename_directories_dataframe(dataframe):
    cases_dir_names = []

    for i in range(len(dataframe['Case Number'].values)):
        if dataframe['Case Number'].values[i] < 10:
            cases_dir_names.append('Case_00'+str(dataframe['Case Number'].values[i]))
        elif dataframe['Case Number'].values[i] > 99:
            cases_dir_names.append('Case_'+str(dataframe['Case Number'].values[i]))
        else:
            cases_dir_names.append('Case_0'+str(dataframe['Case Number'].values[i]))
    return cases_dir_names

def split_two(array_numbers, train_ratio=0.5, val_ratio=0.5, seed=0):
    random.Random(seed).shuffle(array_numbers)
    
    indices_for_splittin = [int(len(array_numbers) * train_ratio)]
    train_indices, val_indices = np.split(array_numbers, indices_for_splittin)
    
    return train_indices, val_indices