import argparse, sys
import numpy as np
import chainer
import chainer.functions as F
from chainer import FunctionSet, Variable, optimizers, cuda, serializers
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help='which gpu device to use', default=0)
parser.add_argument('--lam', type=float, help='trade-off parameter for mutual information and smooth regularization',
                    default=0.1)
parser.add_argument('--prop_eps', type=float, help='epsilon', default=0.25)
parser.add_argument('--n_bit', type=int, help='number of bits', default=16)
parser.add_argument('--hidden_list', type=str, help='hidden size list', default='400-400')
parser.add_argument('--seed', type=int, help='seed for random variable', default=0)
parser.add_argument('--dataset', type=str, default='mnist')

args = parser.parse_args()

lam = args.lam
n_bit = args.n_bit

N_query = 1000

args = parser.parse_args()

if args.dataset == 'mnist':
    sys.path.append('mnist')
    from load_mnist import *

    whole = load_mnist_whole(PATH='mnist/', scale=1.0 / 128.0, shift=-1.0)
else:
    print 'The dataset is not supported.'
    raise NotImplementedError

n_class = np.max(whole.label) + 1
print n_class
dim = whole.data.shape[1]

data = whole.data
target = whole.label

np.random.seed(args.seed)
perm = np.random.permutation(len(target))

cnt_query = [0] * 10
ind_query = []
ind_gallary = []

for i in range(len(target)):
    l = target[perm[i]]
    if cnt_query[l] < 100:
        ind_query.append(perm[i])
        cnt_query[l] += 1
    else:
        ind_gallary.append(perm[i])

x_query = data[ind_query]
x_gallary = data[ind_gallary]
y_query = target[ind_query]
y_gallary = target[ind_gallary]

print x_query.shape
print x_gallary.shape

query = Data(x_query, y_query)
gallary = Data(x_gallary, y_gallary)

print 'use gpu'
chainer.cuda.get_device(args.gpu).use()
print 'query data: ' + str(N_query)
xp = cuda.cupy
hidden_list = map(int, args.hidden_list.split('-'))


def call_bn(bn, x, test=False, update_batch_stats=True):
    if not update_batch_stats:
        return F.batch_normalization(x, bn.gamma, bn.beta, use_cudnn=False)
    if test:
        return F.fixed_batch_normalization(x, bn.gamma, bn.beta, bn.avg_mean, bn.avg_var, use_cudnn=False)
    else:
        return bn(x)


def distance(y0, y1):
    p0 = F.sigmoid(y0)
    p1 = F.sigmoid(y1)
    return F.sum(p0 * F.log((p0 + 1e-8) / (p1 + 1e-8)) + (1 - p0) * F.log((1 - p0 + 1e-8) / (1 - p1 + 1e-8))) / \
           p0.data.shape[0]


def vat(forward, distance, x, eps_list, xi=10, Ip=1):
    y = forward(Variable(x))
    y.unchain_backward()
    # calc adversarial direction
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
            l3=F.Linear(hidden_list[1], n_bit, wscale=0.0001),
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


def enc_test(x):
    return enc(x, test=True)


def loss_unlabeled(x, eps_list):
    L = vat(enc_aux_noubs, distance, x.data, eps_list)
    return L


def loss_test(x_query, t_query, x_gallary, t_gallary):
    query_hash = F.sigmoid(enc(x_query, test=True)).data > 0.5
    gallary_hash = F.sigmoid(enc(x_gallary, test=True)).data > 0.5

    t_query = cuda.to_cpu(t_query.data)
    t_gallary = cuda.to_cpu(t_gallary.data)

    withinN_precision_label = 0
    withinR_precision_label = 0

    mAP = 0

    for i in range(N_query):
        hamming_distance = cuda.to_cpu(xp.sum((1 - query_hash[i]) == gallary_hash, axis=1))
        mAP += metrics.average_precision_score(t_gallary == t_query[i], 1. / (1 + hamming_distance))

        nearestN_index = np.argsort(hamming_distance)[:500]
        withinN_precision_label += float(np.sum(t_gallary[nearestN_index] == t_query[i])) / 500

        withinR_label = t_gallary[hamming_distance < 3]
        num_withinR = len(withinR_label)
        if not num_withinR == 0:
            withinR_precision_label += np.sum(withinR_label == t_query[i]) / float(num_withinR)

    return mAP / N_query, withinN_precision_label / N_query, withinR_precision_label / N_query


def loss_information(enc, x):
    p_logit = enc(x)
    p = F.sigmoid(p_logit)
    p_ave = F.sum(p, axis=0) / x.data.shape[0]

    cond_ent = F.sum(- p * F.log(p + 1e-8) - (1 - p) * F.log(1 - p + 1e-8)) / p.data.shape[0]
    marg_ent = F.sum(- p_ave * F.log(p_ave + 1e-8) - (1 - p_ave) * F.log(1 - p_ave + 1e-8))

    p_ave = F.reshape(p_ave, (1, len(p_ave.data)))

    p_ave_separated = F.separate(p_ave, axis=1)
    p_separated = F.separate(F.expand_dims(p, axis=2), axis=1)

    p_ave_list_i = []
    p_ave_list_j = []

    p_list_i = []
    p_list_j = []

    for i in range(n_bit - 1):
        p_ave_list_i.extend(list(p_ave_separated[i + 1:]))
        p_list_i.extend(list(p_separated[i + 1:]))

        p_ave_list_j.extend([p_ave_separated[i] for n in range(n_bit - i - 1)])
        p_list_j.extend([p_separated[i] for n in range(n_bit - i - 1)])

    p_ave_pair_i = F.expand_dims(F.concat(tuple(p_ave_list_i), axis=0), axis=1)
    p_ave_pair_j = F.expand_dims(F.concat(tuple(p_ave_list_j), axis=0), axis=1)

    p_pair_i = F.expand_dims(F.concat(tuple(p_list_i), axis=1), axis=2)
    p_pair_j = F.expand_dims(F.concat(tuple(p_list_j), axis=1), axis=2)

    p_pair_stacked_i = F.concat((p_pair_i, 1 - p_pair_i, p_pair_i, 1 - p_pair_i), axis=2)
    p_pair_stacked_j = F.concat((p_pair_j, p_pair_j, 1 - p_pair_j, 1 - p_pair_j), axis=2)

    p_ave_pair_stacked_i = F.concat((p_ave_pair_i, 1 - p_ave_pair_i, p_ave_pair_i, 1 - p_ave_pair_i), axis=1)
    p_ave_pair_stacked_j = F.concat((p_ave_pair_j, p_ave_pair_j, 1 - p_ave_pair_j, 1 - p_ave_pair_j), axis=1)

    p_product = F.sum(p_pair_stacked_i * p_pair_stacked_j, axis=0) / len(p.data)
    p_ave_product = p_ave_pair_stacked_i * p_ave_pair_stacked_j
    pairwise_mi = 2 * F.sum(p_product * F.log((p_product + 1e-8) / (p_ave_product + 1e-8)))

    return cond_ent, marg_ent, pairwise_mi


enc = Encoder()
enc.to_gpu()

o_enc = optimizers.Adam(alpha=0.002, beta1=0.9)
o_enc.setup(enc)

batchsize = 250
N_gallary = len(gallary.data)

nearest_dist = np.loadtxt(args.dataset + '/10th_neighbor.txt').astype(np.float32)

x_query, t_query = cuda.to_gpu(query.data), cuda.to_gpu(query.label)
x_gallary, t_gallary = cuda.to_gpu(gallary.data), cuda.to_gpu(gallary.label)
n_epoch = 50
for epoch in range(n_epoch):
    print epoch
    sum_cond_ent = 0
    sum_marg_ent = 0
    sum_pairwise_mi = 0
    sum_vat = 0
    for it in range(N_gallary / batchsize):
        x, _, ind = whole.get(batchsize, need_index=True)
        cond_ent, marg_ent, pairwise_mi = loss_information(enc, Variable(x))

        sum_cond_ent += cond_ent.data
        sum_marg_ent += marg_ent.data
        sum_pairwise_mi += pairwise_mi.data

        loss_info = cond_ent - marg_ent + pairwise_mi
        loss_ul = loss_unlabeled(Variable(x), cuda.to_gpu(nearest_dist[ind]))
        sum_vat += loss_ul.data

        o_enc.zero_grads()
        (loss_ul + lam * loss_info).backward()
        o_enc.update()

        loss_ul.unchain_backward()
        loss_info.unchain_backward()

    condent = sum_cond_ent / (N_gallary / batchsize)
    margent = sum_marg_ent / (N_gallary / batchsize)
    pairwise = sum_pairwise_mi / (N_gallary / batchsize)
    print 'conditional entropy: ' + str(condent)
    print 'marginal entropy: ' + str(margent)
    print 'pairwise mi: ' + str(pairwise)
    print 'vat loss: ' + str(sum_vat / (N_gallary / batchsize))

    sys.stdout.flush()

mAP, withNpreclabel, withRpreclabel = loss_test(Variable(x_query, volatile=True), Variable(t_query, volatile=True),
                                                Variable(x_gallary, volatile=True), Variable(t_gallary, volatile=True))

print 'mAP: ', mAP
print 'withNpreclabel: ', withNpreclabel
print 'withRpreclabel: ', withRpreclabel
