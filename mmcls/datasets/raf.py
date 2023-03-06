import os
import random
from typing import Any, Dict

import numpy as np
from mmcls.models.losses import accuracy, f1_score, precision, recall
from mmcls.models.losses.eval_metrics import class_accuracy

from .base_dataset import BaseDataset
from .builder import DATASETS


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples

#                  0       1         2          3       4               5           6           7
FER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral', 'Contempt']

def convert2coarse_label(i:int):
    """The first four are negative"""
    if i <= 3:
        return 0
    return i - 3


def gen_class_map(dataset_class):
    """
    generate the convert map from DATASET_CLASSES to FER_CLASSES
    """
    convert_map = []
    for i in dataset_class:
        convert_map.append(FER_CLASSES.index(i))
    assert sum(convert_map) == sum([i for i in range(len(dataset_class))])
    return convert_map

@DATASETS.register_module()
class RAF(BaseDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    DATASET_CLASSES = [
        'Surprise',
        'Fear',
        'Disgust',
        'Happiness',
        'Sadness',
        'Anger',
        'Neutral'
    ]
    CLASSES = FER_CLASSES[:7]

    @staticmethod
    def convert_gt_label(i:int):
        """# dataset -> FER_CLASSES"""
        convert_table = (5, 2, 1, 4, 3, 0, 6)
        assert sum(convert_table) == sum([i for i in range(7)])
        return convert_table[i]

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().split(' ') for x in f.readlines()]
            samples = [[i[0].replace('_aligned', ''), i[1]] for i in samples]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            gt_label = int(gt_label) - 1
            gt_label = self.convert_gt_label(gt_label)
            coarse_label = convert2coarse_label(gt_label)
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['coarse_label'] = np.array(coarse_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

