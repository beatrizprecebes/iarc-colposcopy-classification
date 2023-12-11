import numpy as np
import random

def rename_directories_dataframe(dataframe, type='colpo'):
    cases_dir_names = []

    if type == 'colpo':
        column_name = 'Case Number'
    elif type == 'via':
        column_name = 'CaseNumber'

    for i in range(len(dataframe[column_name].values)):
        if dataframe[column_name].values[i] < 10:
            cases_dir_names.append('Case_00'+str(dataframe[column_name].values[i]))
        elif dataframe[column_name].values[i] > 99:
            cases_dir_names.append('Case_'+str(dataframe[column_name].values[i]))
        else:
            cases_dir_names.append('Case_0'+str(dataframe[column_name].values[i]))
    return cases_dir_names

def split_two(array_numbers, train_ratio=0.5, val_ratio=0.5, seed=0):
    random.Random(seed).shuffle(array_numbers)
    
    indices_for_splittin = [int(len(array_numbers) * train_ratio)]
    train_indices, val_indices = np.split(array_numbers, indices_for_splittin)
    
    return train_indices, val_indices


def export_txt_file(path, dataframe):
    #export DataFrame to text file
    with open(path, 'a') as f:
        df_string = dataframe.to_string(header=False, index=True)
        f.write(df_string)
    return df_string