from sklearn.metrics import r2_score
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from comutils.jax_node_icnn_cann import *
import sys
from jax.example_libraries import optimizers
import jax.numpy as np
from jax.nn import softplus
import matplotlib
import numpy
import numpy as onp
from jax import random, grad, jit, vmap, config, value_and_grad
from functools import partial

# key_value = numpy.random.randint(low=numpy.iinfo(np.int32).min, high=numpy.iinfo(np.int32).max, dtype=numpy.int32)
# key = random.PRNGKey(key_value)
key = random.PRNGKey(0)


sys.path.append('..')

# matplotlib.use('Qt5Agg')

config.update("jax_enable_x64", True)
plt.rcParams.update({'font.size': 12})


class CANN_model():
    def __init__(self, params_I1, params_I2, normalization):
        self.params_I1 = params_I1
        self.params_I2 = params_I2
        self.normalization = normalization

    # Psi1
    def Psi1norm(self, I1norm):
        # Note: I1norm = (I1-3)/normalization
        return CANN_dpsidInorm(I1norm, self.params_I1)[:, 0] / normalization[0]

    # Psi2
    def Psi2norm(self, I2norm):
        # Note: I2norm = (I2-3)/normalization
        return CANN_dpsidInorm(I2norm, self.params_I2)[:, 0] / normalization[1]


# play with ICNN a bit, how do we get that one to work
class ICNN_model():
    def __init__(self, params_I1, params_I2, normalization):
        self.params_I1 = params_I1
        self.params_I2 = params_I2
        self.normalization = normalization

    # Psi1
    # note: for ICNN the prediction is the function not the gradient
    # but the P11_UT, P11_ET, etc expects the model to output the derivative
    def Psi1norm(self, I1norm):
        # Note: I1norm = (I1-3)/normalization
        def f1(x): return icnn_forwardpass(x, self.params_I1)[0]
        # normalization = [I1_factor,I2_factor,Psi1_factor,Psi2_factor ]
        df1 = grad(f1)
        return vmap(df1)(I1norm[:, None])[:, 0] / self.normalization[0]

    # Psi2
    # note: for ICNN the prediction is the function not the gradient
    def Psi2norm(self, I2norm):
        # Note: I2norm = (I2-3)/normalization
        def f2(x): return icnn_forwardpass(x, self.params_I2)[0]
        # normalization = [I1_factor,I2_factor,Psi1_factor,Psi2_factor ]
        df2 = grad(f2)
        return vmap(df2)(I2norm[:, None])[:, 0] / self.normalization[1]

## NODE model outputs normalized strain energy given normalized invariants
class NODE_model():
    def __init__(self, params_I1, params_I2):
        self.params_I1 = params_I1
        self.params_I2 = params_I2
    def Psi1norm(self, I1norm):
        # Note: I1norm = (I1-3)/normalization
        return NODE_posb_vmap(I1norm, self.params_I1)
    def Psi2norm(self, I2norm):
        # Note: I2norm = (I2-3)/normalization
        return NODE_posb_vmap(I2norm, self.params_I2)

def plotstresses(x_gt, y_gt, x_pr, y_pr):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    labels = ['UT', 'ET', 'PS']
    for axi, x_gti, y_gti, x_pri, y_pri, label in zip(ax, x_gt, y_gt, x_pr, y_pr, labels):
        axi.plot(x_gti, y_gti, 'k.')
        axi.plot(x_pri, y_pri)
        axi.set_title(label)
        axi.set_xlabel(r'Stretch $\lambda [-]$')
        axi.set_ylabel(r'Nominal stress $P_{11} [MPa]$')
    return fig, ax

# Maximum I1 and I2 for different cases: UT: 60, 15 ET: 40, 392 PS: 26, 26
# I1_factor = 30
# Psi1_factor = 0.3
# I2_factor = 250
# Psi2_factor = 0.001


def P11_UT(lamb, model, normalization):
    I1_factor = normalization[0]
    I2_factor = normalization[1]
    Psi1_factor = normalization[2]
    Psi2_factor = normalization[3]
    I1 = lamb ** 2 + 2 / lamb
    I2 = 2 * lamb + 1 / lamb ** 2
    I1norm = (I1 - 3) / I1_factor
    I2norm = (I2 - 3) / I2_factor
    Psi1 = model.Psi1norm(I1norm) * Psi1_factor
    Psi2 = model.Psi2norm(I2norm) * Psi2_factor
    return 2 * (Psi1 + Psi2 / lamb) * (lamb - 1 / lamb ** 2)


def P11_ET(lamb, model, normalization):
    I1_factor = normalization[0]
    I2_factor = normalization[1]
    Psi1_factor = normalization[2]
    Psi2_factor = normalization[3]
    I1 = 2 * lamb ** 2 + 1 / lamb ** 4
    I2 = lamb ** 4 + 2 / lamb ** 2
    I1norm = (I1 - 3) / I1_factor
    I2norm = (I2 - 3) / I2_factor
    Psi1 = model.Psi1norm(I1norm) * Psi1_factor
    Psi2 = model.Psi2norm(I2norm) * Psi2_factor
    return 2 * (Psi1 + Psi2 * lamb ** 2) * (lamb - 1 / lamb ** 5)


def P11_PS(lamb, model, normalization):
    I1_factor = normalization[0]
    I2_factor = normalization[1]
    Psi1_factor = normalization[2]
    Psi2_factor = normalization[3]
    I1 = lamb ** 2 + 1 / lamb ** 2 + 1
    I2 = lamb ** 2 + 1 / lamb ** 2 + 1
    I1norm = (I1 - 3) / I1_factor
    I2norm = (I2 - 3) / I2_factor
    Psi1 = model.Psi1norm(I1norm) * Psi1_factor
    Psi2 = model.Psi2norm(I2norm) * Psi2_factor
    return 2 * (Psi1 + Psi2) * (lamb - 1 / lamb ** 3)


# Eval invariants for the three types of deformation
def evalI1_UT(lam):
    return lam ** 2 + (2 / lam)


def evalI2_UT(lam):
    return 2 * lam + (1 / lam ** 2)


def evalI1_ET(lam):
    return 2 * lam ** 2 + 1 / lam ** 4


def evalI2_ET(lam):
    return lam ** 4 + 2 / lam ** 2


def evalI1_PS(lam):
    return lam ** 2 + 1 / lam ** 2 + 1


def evalI2_PS(lam):
    return lam ** 2 + 1 / lam ** 2 + 1


# read the data
UTdata = pd.read_csv('../Data/UT20.csv')
ETdata = pd.read_csv('../Data/ET20.csv')
PSdata = pd.read_csv('../Data/PS20.csv')

# stack into single array
P11_data = np.hstack([UTdata['P11'].to_numpy(), ETdata['P11'].to_numpy(), PSdata['P11'].to_numpy()])
F11_data = np.hstack([UTdata['F11'].to_numpy(), ETdata['F11'].to_numpy(), PSdata['F11'].to_numpy()])
# indices for the three data sets
indET = len(UTdata['P11'])
indPS = indET + len(ETdata['P11'])
# Maximum I1 and I2 for different cases: UT: 60, 15 ET: 40, 392 PS: 26, 26

I1_factor = 30
Psi1_factor = 0.3
I2_factor = 250
Psi2_factor = 0.001
normalization = [I1_factor, I2_factor, Psi1_factor, Psi2_factor]


# @partial(jit, static_argnums=(2,))
def loss_P11_all(params, F11_data, mdlnumber):
    lamUT = F11_data[0:indET]
    lamET = F11_data[indET:indPS]
    lamPS = F11_data[indPS:]
    params_I1 = params[0]
    params_I2 = params[1]
    if mdlnumber == 1:
        model = CANN_model(params_I1, params_I2, normalization)
    elif mdlnumber == 2:
        model = ICNN_model(params_I1, params_I2, normalization)
    else:
        model = NODE_model(params_I1, params_I2)

    P11UT_pr = P11_UT(lamUT, model, normalization)
    P11ET_pr = P11_ET(lamET, model, normalization)
    P11PS_pr = P11_PS(lamPS, model, normalization)
    return np.mean((P11UT_pr - P11_data[0:indET]) ** 2) + np.mean((P11ET_pr - P11_data[indET:indPS]) ** 2) + np.mean(
        (P11PS_pr - P11_data[indPS:]) ** 2)


@partial(jit, static_argnums=(2,))
def loss_P11_UT(params, F11_data, mdlnumber):
    lamUT = F11_data[0:indET]
    params_I1 = params[0]
    params_I2 = params[1]
    if mdlnumber == 1:
        model = CANN_model(params_I1, params_I2, normalization)
    elif mdlnumber == 2:
        model = ICNN_model(params_I1, params_I2, normalization)
    else:
        model = NODE_model(params_I1, params_I2)
    P11UT_pr = P11_UT(lamUT, model, normalization)
    return np.mean((P11UT_pr - P11_data[0:indET]) ** 2)


@partial(jit, static_argnums=(2,))
def loss_P11_ET(params, F11_data, mdlnumber):
    lamET = F11_data[indET:indPS]
    params_I1 = params[0]
    params_I2 = params[1]
    if mdlnumber == 1:
        model = CANN_model(params_I1, params_I2, normalization)
    elif mdlnumber == 2:
        model = ICNN_model(params_I1, params_I2, normalization)
    else:
        model = NODE_model(params_I1, params_I2)
    P11ET_pr = P11_ET(lamET, model, normalization)
    return np.mean((P11ET_pr - P11_data[indET:indPS]) ** 2)


@partial(jit, static_argnums=(2,))
def loss_P11_PS(params, F11_data, mdlnumber):
    lamPS = F11_data[indPS:]
    params_I1 = params[0]
    params_I2 = params[1]
    if mdlnumber == 1:
        model = CANN_model(params_I1, params_I2, normalization)
    elif mdlnumber == 2:
        model = ICNN_model(params_I1, params_I2, normalization)
    else:
        model = NODE_model(params_I1, params_I2)
    P11PS_pr = P11_PS(lamPS, model, normalization)
    return np.mean((P11PS_pr - P11_data[indPS:]) ** 2)


@partial(jit, static_argnums=(0, 1,))
def step_jp(loss, mdlnumber, i, opt_state, X_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, mdlnumber)
    # loss_value, g = value_and_grad(loss)(params, X_batch, mdlnumber)
    return opt_update(i, g, opt_state)


def train_jp(loss, mdlnumber, X, opt_state, key, nIter=10000, print_freq=1000):
    train_loss = []
    val_loss = []
    print('start training...')
    for it in range(nIter):
        opt_state = step_jp(loss, mdlnumber, it, opt_state, X)

        if (it + 1) % print_freq == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, mdlnumber)
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it + 1, train_loss_value)
            print(to_print)

    return get_params(opt_state), train_loss, val_loss


# Using layers = [1,2,3,1] for node and [1,4,1] for icnn
# n_params_cann = 212 = 24
# n_params_icnn = 2(4 + 8) = 24
# n_params_node = 2*(2 + 6 + 3) = 22

def init_icnn(key, layers=[1, 3, 4, 1]):
    params_I1 = init_params_icnn(layers, key)
    params_I2 = init_params_icnn(layers, key)
    return [params_I1, params_I2]


def init_cann(key, layers=None):
    params_I1 = init_params_cann(key)
    params_I2 = init_params_cann(key)
    return [params_I1, params_I2]

def init_node(key, layers = [1,5,5,1]):
    params_I1 = init_params_posb(layers, key)
    params_I2 = init_params_posb(layers, key)
    return [params_I1,params_I2]


# ---------------------------------------------------------------------------
#    Train ICNN model
# ---------------------------------------------------------------------------
# params_icnn_all = init_icnn(key)
# opt_init, opt_update, get_params = optimizers.adam(2.e-4)  # Original: 1.e-4
# opt_state = opt_init(params_icnn_all)
# mdlnumber = 2
# params_icnn_all, train_loss, val_loss = train_jp(
#     loss_P11_all, mdlnumber, F11_data, opt_state, key, nIter=100000)  # Original 100000
# lamUT_vec = np.linspace(1, UTdata['F11'].iloc[-1], 100)
# lamET_vec = np.linspace(1, ETdata['F11'].iloc[-1], 100)
# lamPS_vec = np.linspace(1, PSdata['F11'].iloc[-1], 100)
#
# normalization = [I1_factor, I2_factor, Psi1_factor, Psi2_factor]
# model = ICNN_model(params_icnn_all[0], params_icnn_all[1], normalization)
# P11_NN_UT_p = P11_UT(lamUT_vec, model, normalization)
# P11_NN_ET_p = P11_ET(lamET_vec, model, normalization)
# P11_NN_PS_p = P11_PS(lamPS_vec, model, normalization)
#
# np.save('../outputs/rubber/benchmark_data/rubber_icnn_ut.npy', P11_NN_UT_p)
# np.save('../outputs/rubber/benchmark_data/rubber_icnn_et.npy', P11_NN_ET_p)
# np.save('../outputs/rubber/benchmark_data/rubber_icnn_ps.npy', P11_NN_PS_p)
#
# fig, ax = plotstresses([UTdata['F11'], ETdata['F11'], PSdata['F11']],
#                        [UTdata['P11'], ETdata['P11'], PSdata['P11']],
#                        [lamUT_vec, lamET_vec, lamPS_vec],
#                        [P11_NN_UT_p, P11_NN_ET_p, P11_NN_PS_p])
# plt.show()


# -------------------------------------------------------------------------------------------------------------------
#    Train CANN model
# -------------------------------------------------------------------------------------------------------------------
# key, subkey = random.split(key)
# params_cann_all = init_cann(key)
# opt_init, opt_update, get_params = optimizers.adam(2.e-4)  # Original: 1.e-4
# opt_state = opt_init(params_cann_all)
# mdlnumber = 1
# params_cann_all, train_loss, val_loss = train_jp(loss=loss_P11_all, mdlnumber=mdlnumber, X=F11_data,
#                                                  opt_state=opt_state, key=key, nIter=100000)  # Original 100000
#
# lamUT_vec = np.linspace(1, UTdata['F11'].iloc[-1], 100)
# lamET_vec = np.linspace(1, ETdata['F11'].iloc[-1], 100)
# lamPS_vec = np.linspace(1, PSdata['F11'].iloc[-1], 100)
#
# normalization = [I1_factor, I2_factor, Psi1_factor, Psi2_factor]
# model = CANN_model(params_cann_all[0], params_cann_all[1], normalization)
# P11_NN_UT_p = P11_UT(lamUT_vec, model, normalization)
# P11_NN_ET_p = P11_ET(lamET_vec, model, normalization)
# P11_NN_PS_p = P11_PS(lamPS_vec, model, normalization)
#
# np.save('../outputs/rubber/benchmark_data/rubber_cann_ut.npy', P11_NN_UT_p)
# np.save('../outputs/rubber/benchmark_data/rubber_cann_et.npy', P11_NN_ET_p)
# np.save('../outputs/rubber/benchmark_data/rubber_cann_ps.npy', P11_NN_PS_p)
#
#
# fig, ax = plotstresses([UTdata['F11'], ETdata['F11'], PSdata['F11']],
#                        [UTdata['P11'], ETdata['P11'], PSdata['P11']],
#                        [lamUT_vec, lamET_vec, lamPS_vec],
#                        [P11_NN_UT_p, P11_NN_ET_p, P11_NN_PS_p])
# plt.show()




# -------------------------------------------------------------------------------------------------------------------
#    Train NODE model
# -------------------------------------------------------------------------------------------------------------------
params_all = init_node(key)
opt_init, opt_update, get_params = optimizers.adam(2.e-4) #Original: 2.e-4
opt_state = opt_init(params_all)

mdlnumber = 3 #NODE
params_all, train_loss, val_loss = train_jp(loss_P11_all, mdlnumber, F11_data, opt_state, key, nIter = 100000) #Original 100000

lamUT_vec = np.linspace(1,UTdata['F11'].iloc[-1],100)
lamET_vec = np.linspace(1,ETdata['F11'].iloc[-1],100)
lamPS_vec = np.linspace(1,PSdata['F11'].iloc[-1],100)

normalization = [I1_factor,I2_factor,Psi1_factor,Psi2_factor ]
model = NODE_model(params_all[0], params_all[1])
P11_NN_UT_p = P11_UT(lamUT_vec, model, normalization)
P11_NN_ET_p = P11_ET(lamET_vec, model, normalization)
P11_NN_PS_p = P11_PS(lamPS_vec, model, normalization)

fig, ax = plotstresses([UTdata['F11'], ETdata['F11'], PSdata['F11']],
                       [UTdata['P11'], ETdata['P11'], PSdata['P11']],
                       [lamUT_vec, lamET_vec, lamPS_vec],
                       [P11_NN_UT_p, P11_NN_ET_p, P11_NN_PS_p])

np.save('../outputs/rubber/benchmark_data/rubber_node_ut.npy', P11_NN_UT_p)
np.save('../outputs/rubber/benchmark_data/rubber_node_et.npy', P11_NN_ET_p)
np.save('../outputs/rubber/benchmark_data/rubber_node_ps.npy', P11_NN_PS_p)

plt.show()






# # --------------------------------------------------------------
# #    PLOT ALL RESULTS
# # --------------------------------------------------------------
# I1_factor = 30
# Psi1_factor = 0.3
# I2_factor = 250
# Psi2_factor = 0.001
# normalization = [I1_factor,I2_factor,Psi1_factor,Psi2_factor ]
# UTdata = pd.read_csv('Data/UT20.csv')
# ETdata = pd.read_csv('Data/ET20.csv')
# PSdata = pd.read_csv('Data/PS20.csv')
#
# # ALL
# with open('savednet/NODE_params_all.npy', 'rb') as f:
#     params_node_all = pickle.load(f)
# with open('savednet/ICNN_params_all.npy', 'rb') as f:
#     params_icnn_all = pickle.load(f)
# with open('savednet/CANN_params_all.npy', 'rb') as f:
#     params_cann_all = pickle.load(f)
#
# # UT
# with open('savednet/NODE_params_UT.npy', 'rb') as f:
#     params_node_UT = pickle.load(f)
# with open('savednet/ICNN_params_UT.npy', 'rb') as f:
#     params_icnn_UT = pickle.load(f)
# with open('savednet/CANN_params_UT.npy', 'rb') as f:
#     params_cann_UT = pickle.load(f)
#
# # ET
# with open('savednet/NODE_params_ET.npy', 'rb') as f:
#     params_node_ET = pickle.load(f)
# with open('savednet/ICNN_params_ET.npy', 'rb') as f:
#     params_icnn_ET = pickle.load(f)
# with open('savednet/CANN_params_ET.npy', 'rb') as f:
#     params_cann_ET = pickle.load(f)
#
# # PS
# with open('savednet/NODE_params_PS.npy', 'rb') as f:
#     params_node_PS = pickle.load(f)
# with open('savednet/ICNN_params_PS.npy', 'rb') as f:
#     params_icnn_PS = pickle.load(f)
# with open('savednet/CANN_params_PS.npy', 'rb') as f:
#     params_cann_PS = pickle.load(f)
#
# params_UT = [params_cann_UT, params_icnn_UT, params_node_UT]
# params_ET = [params_cann_ET, params_icnn_ET, params_node_ET]
# params_PS = [params_cann_PS, params_icnn_PS, params_node_PS]
# params_all = [params_cann_all, params_icnn_all, params_node_all]
# params = [params_UT, params_ET, params_PS, params_all]
