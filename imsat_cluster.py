import argparse, sys
import numpy as np
import chainer
import chainer.functions as F
from chainer import FunctionSet, Variable, optimizers, cuda, serializers
from munkres import Munkres, print_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help='which gpu device to use', default=1)
parser.add_argument('--lam', type=float, help='trade-off parameter for mutual information and smooth regularization',
                    default=0.1)
parser.add_argument('--mu', type=float, help='trade-off parameter for entropy minimization and entropy maximization',
                    default=4)
parser.add_argument('--prop_eps', type=float, help='epsilon', default=0.25)
parser.add_argument('--dataset', type=str, help='which dataset to use', default='mnist')
parser.add_argument('--hidden_list', type=str, help='hidden size list', default='1200-1200')

args = parser.parse_args()

if args.dataset == 'mnist':
    sys.path.append('mnist')
    from load_mnist import *

    whole = load_mnist_whole(PATH='mnist/', scale=1.0 / 128.0, shift=-1.0)
else:
    print 'The dataset is not supported.'
    raise NotImplementedError

n_data = len(whole.data)
n_class = np.max(whole.label) + 1
print n_class
dim = whole.data.shape[1]

print 'use gpu'
chainer.cuda.get_device(args.gpu).use()
xp = cuda.cupy
hidden_list = map(int, args.hidden_list.split('-'))


def call_bn(bn, x, test=False, update_batch_stats=True):
    if not update_batch_stats:
        return F.batch_normalization(x, bn.gamma, bn.beta, use_cudnn=False)
    if test:
        return F.fixed_batch_normalization(x, bn.gamma, bn.beta, bn.avg_mean, bn.avg_var, use_cudnn=False)
    else:
        return bn(x)


def kl(p, q):
    return F.sum(p * F.log((p + 1e-8) / (q + 1e-8))) / float(len(p.data))


def distance(y0, y1):
    return kl(F.softmax(y0), F.softmax(y1))


def entropy(p):
    if p.data.ndim == 2:
        return - F.sum(p * F.log(p + 1e-8)) / float(len(p.data))
    elif p.data.ndim == 1:
        return - F.sum(p * F.log(p + 1e-8))
    else:
        raise NotImplementedError


def vat(forward, distance, x, eps_list, xi=10, Ip=1):
    y = forward(Variable(x))
    y.unchain_backward()

    d = xp.random.normal(size=x.shape, dtype=np.float32)
    d = d / xp.sqrt(xp.sum(d ** 2, axis=1)).reshape((x.shape[0], 1))
    for ip in range(Ip):
        d_var = Variable(d.astype(np.float32))
        y2 = forward(x + xi * d_var)
        kl_loss = distance(y, y2)
        kl_loss.backward()
        d = d_var.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=1)).reshape((x.shape[0], 1))
    d_var = Variable(d.astype(np.float32))

    eps = args.prop_eps * eps_list
    y2 = forward(x + F.transpose(eps * F.transpose(d_var)))
    return distance(y, y2)


class Encoder(chainer.Chain):
    def __init__(self):
        super(Encoder, self).__init__(
            l1=F.Linear(dim, hidden_list[0], wscale=0.1),
            l2=F.Linear(hidden_list[0], hidden_list[1], wscale=0.1),
            l3=F.Linear(hidden_list[1], n_class, wscale=0.0001),
            bn1=F.BatchNormalization(hidden_list[0]),
            bn2=F.BatchNormalization(hidden_list[1])
        )

    def __call__(self, x, test=False, update_batch_stats=True):
        h = F.relu(call_bn(self.bn1, self.l1(x), test=test, update_batch_stats=update_batch_stats))
        h = F.relu(call_bn(self.bn2, self.l2(h), test=test, update_batch_stats=update_batch_stats))
        y = self.l3(h)
        return y


def enc_aux_noubs(x):
    return enc(x, test=False, update_batch_stats=False)


def loss_unlabeled(x, eps_list):
    L = vat(enc_aux_noubs, distance, x.data, eps_list)
    return L


def loss_test(x, t):
    prob = F.softmax(enc(x, test=True)).data
    pmarg = cuda.to_cpu(xp.sum(prob, axis=0) / len(prob))
    ent = np.sum(-pmarg * np.log(pmarg + 1e-8))
    pred = cuda.to_cpu(np.argmax(prob, axis=1))
    tt = cuda.to_cpu(t.data)

    m = Munkres()
    mat = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            mat[i][j] = np.sum(np.logical_and(pred == i, tt == j))
    indexes = m.compute(-mat)

    corresp = []
    for i in range(n_class):
        corresp.append(indexes[i][1])

    pred_corresp = [corresp[int(predicted)] for predicted in pred]
    acc = np.sum(pred_corresp == tt) / float(len(tt))
    return acc, ent


def loss_equal(enc, x):
    p_logit = enc(x)
    p = F.softmax(p_logit)
    p_ave = F.sum(p, axis=0) / x.data.shape[0]
    ent = entropy(p)
    return ent, -F.sum(p_ave * F.log(p_ave + 1e-8))


enc = Encoder()
enc.to_gpu()

o_enc = optimizers.Adam(alpha=0.002, beta1=0.9)
o_enc.setup(enc)

batchsize_ul = 250

n_epoch = 50

nearest_dist = np.loadtxt(args.dataset + '/10th_neighbor.txt').astype(np.float32)

for epoch in range(n_epoch):
    print epoch

    sum_loss_entmax = 0
    sum_loss_entmin = 0
    vatt = 0
    for it in range(n_data / batchsize_ul):
        x_u, _, ind = whole.get(batchsize_ul, need_index=True)
        loss_eq1, loss_eq2 = loss_equal(enc, Variable(x_u))

        loss_eq = loss_eq1 - args.mu * loss_eq2

        sum_loss_entmin += loss_eq1.data
        sum_loss_entmax += loss_eq2.data

        loss_ul = loss_unlabeled(Variable(x_u), cuda.to_gpu(nearest_dist[ind]))
        o_enc.zero_grads()
        (loss_ul + args.lam * loss_eq).backward()
        o_enc.update()

        vatt += loss_ul.data

        loss_ul.unchain_backward()

    print 'entmax ', sum_loss_entmax / (n_data / batchsize_ul)
    print 'entmin ', sum_loss_entmin / (n_data / batchsize_ul)
    print 'vatt ', vatt / (n_data / batchsize_ul)

    x_ul, t_ul = cuda.to_gpu(whole.data), cuda.to_gpu(whole.label)
    acc, ment = loss_test(Variable(x_ul, volatile=True), Variable(t_ul, volatile=True))
    print "ment: ", ment
    print "accuracy: ", acc
    sys.stdout.flush()
