import math
import time
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import util as distribution_util

import sys
import numpy as np
from math import pi
import pprint

import pandas as pd

base_dir = './data/'

def load_uci_dataset(dataset, i,ratio = 0.9):
    # We load the data

    if dataset == 'boston':
        datapath = './data/boston/boston_housing'
        data = np.loadtxt(datapath)
        data_num = data.shape[0]
        ind = np.arange(data_num)
        np.random.seed(i)
        np.random.shuffle(ind)
        train_num = int(data_num*ratio)
        X_train = data[:train_num,:-1]
        y_train = data[:train_num,-1]
        X_test = data[train_num:,:-1]
        y_test = data[train_num:,-1]
    elif dataset == 'year':
        datapath = './data/year/YearPredictionMSD.txt'
        data = np.loadtxt(datapath, delimiter=',')
        data_num = data.shape[0]
        ind = np.arange(data_num)
        np.random.seed(i)
        np.random.shuffle(ind)
        train_num = int(data_num*ratio)
        X_train = data[:train_num,1:]
        y_train = data[:train_num,0]
        X_test = data[train_num:,1:]
        y_test = data[train_num:,0]
    elif dataset == 'combined':
        datapath = './data/combined/Folds5x2_pp.xlsx'
        df = pd.read_excel(datapath)
        data = df.to_numpy()
        data_num = data.shape[0]
        ind = np.arange(data_num)
        np.random.seed(i)
        np.random.shuffle(ind)
        train_num = int(data_num*ratio)
        X_train = data[:train_num,:-1]
        y_train = data[:train_num,-1]
        X_test = data[train_num:,:-1]
        y_test = data[train_num:,-1]
    elif dataset == 'wine':
        datapath = './data/wine/winequality-white.csv'
        data = np.genfromtxt(datapath, delimiter=';')
        data = data[1:,:]
        data_num = data.shape[0]
        ind = np.arange(data_num)
        np.random.seed(i)
        np.random.shuffle(ind)
        train_num = int(data_num*ratio)
        X_train = data[:train_num,:-1]
        y_train = data[:train_num,-1]
        X_test = data[train_num:,:-1]
        y_test = data[train_num:,-1]
    elif dataset == 'wine':
        datapath = './data/wine/winequality-white.csv'
        data = np.genfromtxt(datapath, delimiter=',')
        data = data[1:,:]
        data_num = data.shape[0]
        ind = np.arange(data_num)
        np.random.seed(i)
        np.random.shuffle(ind)
        train_num = int(data_num*ratio)
        X_train = data[:train_num,:-1]
        y_train = data[:train_num,-1]
        X_test = data[train_num:,:-1]
        y_test = data[train_num:,-1]
    elif dataset == 'kin8nm':
        datapath = './data/kin8nm/dataset_2175_kin8nm.csv'
        data = np.genfromtxt(datapath, delimiter=',')
        data = data[1:,:]
        data_num = data.shape[0]
        ind = np.arange(data_num)
        np.random.seed(i)
        np.random.shuffle(ind)
        train_num = int(data_num*ratio)
        X_train = data[:train_num,:-1]
        y_train = data[:train_num,-1]
        X_test = data[train_num:,:-1]
        y_test = data[train_num:,-1]
    elif dataset == 'concrete':
        datapath = './data/concrete/Concrete_Data.xls'
        df = pd.read_excel(datapath)
        data = df.to_numpy()
        data_num = data.shape[0]
        ind = np.arange(data_num)
        np.random.seed(i)
        np.random.shuffle(ind)
        train_num = int(data_num*ratio)
        X_train = data[:train_num,:-1]
        y_train = data[:train_num,-1]
        X_test = data[train_num:,:-1]
        y_test = data[train_num:,-1]


    ### add the bias
    #X_train = np.concatenate((X_train, np.ones((len(X_train), 1))), axis=1)
    #X_test = np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)

    # We normalize the features
    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    mean_X_train = np.mean(X_train, 0)
    X_train = (X_train - mean_X_train) / std_X_train
    X_test = (X_test - mean_X_train) / std_X_train
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)
    y_train = (y_train - mean_y_train) / std_y_train

    y_train = np.array(y_train, ndmin = 2).reshape((-1, 1))
    y_test = np.array(y_test, ndmin = 2).reshape((-1, 1))

    return X_train, X_test, np.squeeze(y_train), np.squeeze(y_test), mean_y_train, std_y_train


