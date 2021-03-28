import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0], "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """MSE存在量纲的问题，如房产价格，房产价格的平方除以常数，量纲还是房产价格万元的平方"""
    """计算y_true和y_predict之间MSE"""
    assert len(y_predict) == len(y_true), "the size of y_predict must be equal to the size of y_true"

    return np.sum((y_predict - y_true) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """对应的量纲和y对应的量纲相同，如房产价格预测中，预测误差为4.9万美元左右"""
    """计算y_true和y_predict之间RMSE"""

    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """对应的量纲和y对应的量纲相同，如房产价格预测中，预测误差为3.5万美元左右"""
    """计算y_true和y_predict之间MAE"""
    assert len(y_predict) == len(y_true), "the size of y_predict must be equal to the size of y_true"

    return np.sum(np.absolute(y_predict - y_true)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间R Squared"""

    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)

def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true==0) & (y_predict==0))

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true==0) & (y_predict==1))

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true==1) & (y_predict==0))

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true==1) & (y_predict==1))

def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])

def precison_score(y_true, y_predict):
    try:
        return TP(y_true, y_predict) / (TP(y_true, y_predict) + FP(y_true, y_predict))
    except:
        return 0.0

def recall_score(y_true, y_predict):
    try:
        return TP(y_true, y_predict) / (TP(y_true, y_predict) + FN(y_true, y_predict))
    except:
        return 0.0

def f1_score(y_true, y_predict):
    try:
        return 2 * precison_score(y_true, y_predict) * recall_score(y_true, y_predict) / (precison_score(y_true, y_predict) + recall_score(y_true, y_predict))
    except:
        return 0.0

def TPR(y_true, y_predict):
    try:
        return TP(y_true, y_predict) / (TP(y_true, y_predict) + FN(y_true, y_predict))
    except:
        return 0.0

def FPR(y_true, y_predict):
    try:
        return FP(y_true, y_predict) / (FP(y_true, y_predict) + TN(y_true, y_predict))
    except:
        return 0.0
