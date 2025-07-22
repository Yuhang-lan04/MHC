from __future__ import print_function, division

import argparse
import math
import os
import pickle
import random
import time

import torch
from sklearn.preprocessing import StandardScaler

from measure import *
from test import test_MHC
from train import train_MHC


def set_seed(seed=2000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 添加命令行参数
    parser.add_argument('--lr', type=float, default=3e-5,help="Learning Rate")
    parser.add_argument('--maxiter', default=100, type=int,help="Maximum iterations")
    parser.add_argument('--batch_size', default=128, type=int,help="Batch size")
    parser.add_argument('--dataset', type=str, default='LandUse21',help="Dataset name")
    parser.add_argument('--dataPath', type=str, default='C:/Users/dell/Desktop/multi _view_ classification/re/MHC/data',help="Data path")
    parser.add_argument('--n_z', default=128, type=int,help="Embedding dimension")
    parser.add_argument('--TraindataRatio', type=float, default=0.8,help="Training data ratio")
    parser.add_argument('--temperature', default=0.2, type=float,help="Temperature parameter")
    parser.add_argument('--xi', default=0.05, type=float,help="BIH Boundary Value")
    parser.add_argument('--varphi', type=float, default=0.05,help="BCM Threshold")
    parser.add_argument('--weight_supervised_cross_view_contrastive', type=float, default=0.0001,
                        help='Weight for supervised cross-view contrastive')
    parser.add_argument('--weight_boundary_aware_independent_hash', type=float, default=1,
                        help='Weight for boundary-aware independent hash')
    parser.add_argument('--gpu', default='0', type=str, help='GPU device idx to use.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    file_path = './train_sup-' + args.dataset + '_TR_' + str(args.TraindataRatio) + '-(0610-1(X)).txt'

    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    device = torch.device("cuda" if args.cuda else "cpu")
    Pre_fnum = 5

    seed = [2000, 2001, 2002, 2003, 2004]

    with open(args.dataPath + '/' + args.dataset + '/X.pkl', 'rb') as f:
        data = pickle.load(f)
    X = [data[i] for i in range(len(data))]
    view_num = len(X)
    with open(args.dataPath + '/' + args.dataset + '/Y.pkl', 'rb') as f:
        label = pickle.load(f)
    first_nonzero_indices = [np.argmax(row != 0) for row in label]
    label = np.array(first_nonzero_indices)

    expanded_labels_file = args.dataPath + '/' + args.dataset + '_expanded_labels_' + '.npy'
    expanded_labels = np.load(expanded_labels_file, allow_pickle=True)

    print(args)
    acc = np.zeros(Pre_fnum)
    train_time = np.zeros(Pre_fnum)
    test_time = np.zeros(Pre_fnum)
    for fnum in range(Pre_fnum):
        set_seed(seed[fnum])
        mul_X = [None] * view_num
        Ndata = label.shape[0]
        indexperm = np.arange(Ndata)
        np.random.shuffle(indexperm)
        train_num = math.ceil(Ndata * args.TraindataRatio)
        rtest_num = Ndata - train_num
        print('train_num', train_num)
        print('rtest_num', rtest_num)

        train_index = indexperm[0:train_num]
        rtest_index = indexperm[train_num:]

        for iv in range(view_num):
            mul_X[iv] = np.copy(X[iv])
            mul_X[iv] = mul_X[iv].astype(np.float32)
            mul_X[iv] = StandardScaler().fit_transform(mul_X[iv])
            mul_X[iv] = torch.Tensor(np.nan_to_num(np.copy(mul_X[iv])))
        mul_X_rtest = [xiv[rtest_index] for xiv in mul_X]
        mul_X_train = [xiv[train_index] for xiv in mul_X]

        train_label = expanded_labels[train_index]
        original_train_labels = label[train_index]
        yrt_label = label[rtest_index]
        original_train_labels = torch.tensor(original_train_labels, device=device)
        st = time.time()
        model, mul_X_hashes = train_MHC(mul_X_train, train_label, original_train_labels, mul_X_rtest, yrt_label, device,
                                        args)
        train_time[fnum] = time.time() - st
        yy_pred, test_time[fnum] = test_MHC(model, mul_X_rtest, mul_X_hashes, original_train_labels, device, args)

        value_result = do_metric(yy_pred, yrt_label)

        print("final:acc")
        print(value_result)
        print("inference time:")
        print(test_time[fnum])

        acc[fnum] = value_result

    print('mean_acc: {}'.format(round(acc.mean(), 4)))
    print('std_acc: {}'.format(round(acc.std(), 4)))
    result_file_path = os.path.join(results_dir, os.path.basename(file_path))
    file_handle = open(result_file_path, mode='a')
    if os.path.getsize(result_file_path) == 0:
        file_handle.write('Data Record\n')

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_handle.write('mean_acc: {}\n'.format(round(acc.mean(), 4)))
    file_handle.write('std_acc: {}\n'.format(round(acc.std(), 4)))
    file_handle.write('ava_train_time: {}\n'.format(round(train_time.mean(), 4)))
    file_handle.write('ava_test_time: {}\n'.format(round(test_time.mean(), 9)))
    file_handle.write('lr: {}\n'.format(args.lr))
    file_handle.write('maxiter: {}\n'.format(args.maxiter))
    file_handle.write('xi:{}\n'.format(args.xi))
    file_handle.write('temperature:{}\n'.format(args.temperature))
    file_handle.write('varphi: {}\n'.format(args.varphi))
    file_handle.write('n_z: {}\n'.format(args.n_z))
    file_handle.write('weight_supervised_cross_view_contrastive: {}\n'.format(args.weight_supervised_cross_view_contrastive))
    file_handle.write('weight_boundary_aware_independent_hash: {}\n'.format(args.weight_boundary_aware_independent_hash))
    file_handle.write('current_time: {}\n'.format(current_time))
    file_handle.write('--------------------\n')
    file_handle.close()
