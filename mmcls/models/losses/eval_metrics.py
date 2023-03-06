import numpy as np
import torch


def calculate_confusion_matrix(pred, target):
    if isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)
    elif not (isinstance(pred, torch.Tensor)
              and isinstance(target, torch.Tensor)):
        raise TypeError('pred and target should both be'
                        'torch.Tensor or np.ndarray')
    _, pred_label = pred.topk(1, dim=1)
    num_classes = pred.size(1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t, p in zip(target_label, pred_label):
            confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def class_accuracy(pred, target, classes=None):
    confusion_matrix = calculate_confusion_matrix(pred, target)
    # plot_confusion_matrix(confusion_matrix.type(torch.long), classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues)
    with torch.no_grad():
        result = []
        for i in range(confusion_matrix.shape[0]):
            acc = confusion_matrix[i][i] / confusion_matrix[i].sum()
            result.append(acc.item() * 100)
    return result


def precision(pred, target):
    """Calculate macro-averaged precision according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as precision.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(0), min=1)
        res = res.mean().item() * 100
    return res


def recall(pred, target):
    """Calculate macro-averaged recall according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as recall.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(1), min=1)
        res = res.mean().item() * 100
    return res


def f1_score(pred, target):
    """Calculate macro-averaged F1 score according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction.

    Returns:
        float: The function will return a single float as F1 score.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        precision = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(1), min=1)
        recall = confusion_matrix.diag() / torch.clamp(
            confusion_matrix.sum(0), min=1)
        res = 2 * precision * recall / torch.clamp(
            precision + recall, min=1e-20)
        res = torch.where(torch.isnan(res), torch.full_like(res, 0), res)
        res = res.mean().item() * 100
    return res


import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.type(torch.float32)
        sum_row = cm.sum(dim=1)
        cm = cm / sum_row.unsqueeze(1)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig('output/confusion_matrix_FERPlus.jpg')
