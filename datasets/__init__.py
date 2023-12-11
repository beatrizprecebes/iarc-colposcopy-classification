from .iarc_dataset import IARCGeneralDataset

def get_trainval_datasets(tag, mode):
    if tag == 'iarc_general':
        train_set_path = '/workspace/experiments/vit/datasets/prepared_data/iarc_multiclass_general_train.csv'
        val_set_path = '/workspace/experiments/vit/datasets/prepared_data/iarc_multiclass_general_val.csv'
        return IARCGeneralDataset(train_set_path, mode=mode), IARCGeneralDataset(val_set_path, mode=mode)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))

def get_test_dataset(tag, mode):
    if tag == 'iarc_general':
        test_set_path = '/workspace/experiments/vit/datasets/prepared_data/iarc_multiclass_general_test.csv'
        return IARCGeneralDataset(test_set_path, mode=mode)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))