import scipy.io as sio
import scipy.sparse as sis
import numpy as np
from pathlib import Path
import os

# TODO: New Dataset directories and documentation!
DATASETS_DIR = 'datasets'
DATASETS_OUTPUT_DIR = 'datasets_csv'

def to_full(mat):
    if isinstance(mat, sis.csc_matrix):
        return mat.toarray()
    else:
        return mat

def load_file(file):
    data = sio.loadmat(file)
    if 'Xs' in data:
        Xs = np.array(to_full(data['Xs']))
        Xt = np.array(to_full(data['Xt']))
        Ys = np.array(to_full(data['Ys']))
        Yt = np.array(to_full(data['Yt']))
        return Xs, Xt, Ys, Yt
    elif 'fts' in data:
        Xs = np.array(data['fts']).transpose()
        Xt = None
        Ys = np.array(data['labels'])
        Yt = None
        return Xs, Xt, Ys, Yt
    else:
        raise Exception('unimplemented')

def process_data(Xs, Xt, Ys, Yt, output_path):
    Xs = Xs.transpose()
    Xt = Xt.transpose() if Xt is not None else None
    #print(str.format('Xs: {} Ys: {}', Xs.shape, Ys.shape))
    s = np.concatenate((Ys, Xs), axis=1)
    t = np.concatenate((Yt, Xt), axis=1) if Yt is not None else None
    Ys = None # these are needed to reduce the used memory
    Xs = None
    Yt = None
    Xt = None
    result = np.concatenate((s, t), axis=0) if t is not None else s
    s = None
    t = None
    target_file_name = output_path
    target_dir = os.path.dirname(target_file_name)
    print('saving: ' + target_file_name)
    os.makedirs(target_dir, exist_ok=True)
    np.savetxt(target_file_name, result, delimiter=',', fmt='%f')


def to_output_path(file):
    target_file_name = str(file)
    target_file_name = target_file_name.replace(DATASETS_DIR, DATASETS_OUTPUT_DIR)
    target_file_name = target_file_name.replace('.mat', '.csv')
    return target_file_name

def process_file(file):
    print('processing: ' + str(file))

    target_file_name = to_output_path(file)

    if target_file_name == str(file):
        raise Exception('filename ill formed')

    Xs, Xt, Ys, Yt = load_file(file)
    process_data(Xs, Xt, Ys, Yt, target_file_name)


def convert_subdir_to_csv(path):
    files = list(Path(path).rglob('*.mat'))
    for file in files:
        process_file(file)

def generate_filename_cartesian(file1, file2):
    file1 = str(file1).replace('.mat', '')
    return file1 + '+' + file2

def convert_subdir_to_csv_cartesian(path):
    files = list(Path(path).rglob('*.mat'))
    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            if j <= i: continue
            outputpath = to_output_path(os.path.dirname(file1) + '/' + generate_filename_cartesian(os.path.basename(file1), os.path.basename(file2)))
            Xs, _, Ys, _ = load_file(file1)
            Xt, _, Yt, _ = load_file(file2)
            process_data(Xs, Xt, Ys, Yt, outputpath)


def convert_dir_to_csv(path):
    convert_subdir_to_csv_cartesian(path + '/OfficeCaltech/')
    convert_subdir_to_csv(path + '/Reuters')
    convert_subdir_to_csv(path + '/20Newsgroup')



if __name__ == '__main__':
    convert_dir_to_csv('./' + DATASETS_DIR)


