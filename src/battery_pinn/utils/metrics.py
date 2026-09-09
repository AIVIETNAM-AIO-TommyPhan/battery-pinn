import numpy as np


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def mape(pred, true):
    return float(np.mean(np.abs((pred - true) / true))) * 100


def r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def full_metrics(pred, true):
    return dict(rmse=rmse(pred, true), mae=mae(pred, true), mape=mape(pred, true), r2=r2(pred, true))
