import argparse
import sys
import cPickle as pickle
import datetime, math, sys, time

from sklearn.datasets import fetch_mldata
import numpy as np
import cupy as cp

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import FunctionSet, Variable, optimizers, cuda, serializers

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help = 'which gpu device to use', default = 1)
parser.add_argument('--dataset', type=str, default = 'mnist')

args = parser.parse_args()

chainer.cuda.get_device(args.gpu).use()

if args.dataset == 'mnist':
    sys.path.append('mnist')
    from load_mnist import *
    whole = load_mnist_whole(PATH = 'mnist/', scale=1.0/128.0, shift=-1.0)
else:
    print 'The dataset is not supported.'
    exit(-1)

data = cuda.to_gpu(whole.data)


num_data = [10]

print num_data

dist_accum = 0
dist_list = [[] for i in range(len(num_data))]

for i in range(len(data)):
	if i % 1000 == 0:
		print i
	dist = cp.sqrt(cp.sum((data - data[i])**2, axis = 1))
	dist[i] = 1000
	sorted_dist = np.sort(cuda.to_cpu(dist))
	for j in range(len(num_data)):
		dist_list[j].append(sorted_dist[num_data[j]])

for i in range(len(num_data)):
	np.savetxt(args.dataset + '/' + str(num_data[i]) + 'th_neighbor.txt', np.array(dist_list[i]))

