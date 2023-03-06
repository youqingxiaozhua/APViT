import copy
from abc import ABCMeta, abstractmethod
import os
from shutil import copyfile

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmcls.models.losses import accuracy, f1_score, precision, recall
from mmcls.models.losses.eval_metrics import class_accuracy
from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations()
        # self.data_infos = self.load_exp_and_au_annotations()
        self.CLASSES = self.get_classes(classes)
        if os.environ.get('DEBUG_MODE', '0') == '1':
            self.data_infos = self.data_infos[:30]

    @abstractmethod
    def load_annotations(self):
        pass

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels
    
    def get_coarse_labels(self):
        coarse_labels = np.array([data['coarse_label'] for data in self.data_infos])
        return coarse_labels

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return (int(self.data_infos[idx]['gt_label'].astype(np.int)), )

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def evaluate(self,
                results,
                metric='accuracy',
                metric_options={'topk': (1, 2)},
                logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'class_accuracy']
        eval_results = {}
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs, f'{len(gt_labels)}, {num_imgs}'
            # convert gt_labels to another dataset format for cross-dataset test
            # raf2affect_dict = [6, 3, 4, 0, 1, 2, 5] # RAF -> AffectNet实验需要将RAF标签修改为AffectNet的标签
            # affect2raf_dict = [3, 4, 5, 1, 2, 6, 0]
            # raf2ferplus_dict = [6,3,0,4,5,2,1,None]
            # gt_labels = [raf2ferplus_dict[i] for i in gt_labels]
            # # specific process for RAF -> FERPlus
            # results = [results[i] for i in range(len(results)) if gt_labels[i] is not None]
            # gt_labels = [gt_labels[i] for i in range(len(gt_labels)) if gt_labels[i] is not None]
            # results = np.array(results)
            # gt_labels = np.array(gt_labels)

            if metric == 'accuracy':
                topk = metric_options.get('topk')
                acc = accuracy(results, gt_labels, topk)
                eval_result = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            elif metric == 'precision':
                precision_value = precision(results, gt_labels)
                eval_result = {'precision': precision_value}
            elif metric == 'recall':
                recall_value = recall(results, gt_labels)
                eval_result = {'recall': recall_value}
            elif metric == 'f1_score':
                f1_score_value = f1_score(results, gt_labels)
                eval_result = {'f1_score': f1_score_value}
            elif metric == 'class_accuracy':
                class_accuracy_value = class_accuracy(results, gt_labels, self.CLASSES)
                print('\n')
                for name, val in zip(self.CLASSES, class_accuracy_value):
                    print(f'{name}: \t{val}')
                print('\n')
                eval_result = dict()
            eval_results.update(eval_result)
        return eval_results
