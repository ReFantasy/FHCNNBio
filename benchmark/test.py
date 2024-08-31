import matplotlib.pyplot as plt
from comutils.jax_node_icnn_cann import *
import sys
from jax.example_libraries import optimizers
import jax.numpy as np
from jax import random, grad, jit, vmap, config
import time
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

key = random.PRNGKey(2810)

sys.path.append('..')

config.update("jax_enable_x64", True)


plt.rcParams.update({'font.size': 12})

with open('../Data/P12AC1_bsxsy.npy', 'rb') as f:
    lamb, sigma = np.load(f, allow_pickle=True)
lamb = lamb.astype(np.float64)
sigma = sigma.astype(np.float64)
ind_sx = 81
ind_sy = 182
lamb_sigma = np.hstack([lamb, sigma])


# Doing with an object 'model' I still like that
def eval_Cauchy(lambx, lamby, model, theta):
    I1 = lambx ** 2 + lamby ** 2 + (1. / (lambx * lamby) ** 2)
    I2 = lambx ** 2 * lamby ** 2 + lambx ** 2 * \
        (1. / (lambx * lamby) ** 2) + lamby ** 2 * (1. / (lambx * lamby) ** 2)

    theta_a = theta[0]
    theta_s = theta[1]

    # * jnp.sin(theta_a)**2
    I4a = lambx ** 2 * jnp.sin(jnp.pi/2*jnp.tanh(theta_a))**2
    # * jnp.cos(theta_s)**2
    I4s = lamby ** 2 * jnp.cos(jnp.pi/2*jnp.tanh(theta_s))**2

    I1 = (I1 - 3)
    I2 = (I2 - 3)
    I4a = (I4a - 1)
    I4s = (I4s - 1)

    Psi1 = model.Psi1(I1)
    Psi2 = model.Psi2(I2)
    Psi_1_4a = model.Psi_1_4a(I1, I4a)
    Psi_4a_1 = model.Psi_4a_1(I1, I4a)
    Psi_1_4s = model.Psi_1_4s(I1, I4s)
    Psi_4s_1 = model.Psi_4s_1(I1, I4s)
    Psi_4a_4s = model.Psi_4a_4s(I4a, I4s)
    Psi_4s_4a = model.Psi_4s_4a(I4a, I4s)

    Psi1 = Psi1 + 0.0 + Psi_4a_1 + Psi_4s_1
    Psi2 = Psi2 + 0.0 + 0.0 + 0.0
    Psi4a = 0.0 + Psi_1_4a + 0.0 + Psi_4s_4a
    Psi4s = 0.0 + Psi_1_4s + 0.0 + Psi_4a_4s

    lambz = 1. / (lambx * lamby)
    p = 2 * Psi1 * lambz ** 2 + 2 * Psi2 * (I1 * lambz ** 2 - lambz ** 4)
    sigx = 2 * Psi1 * lambx ** 2 + 2 * Psi2 * \
        (I1 * lambx ** 2 - lambx ** 4) + 2 * Psi4a * lambx ** 2 - p
    sigy = 2 * Psi1 * lamby ** 2 + 2 * Psi2 * \
        (I1 * lamby ** 2 - lamby ** 4) + 2 * Psi4s * lamby ** 2 - p
    return sigx, sigy


def plotstresses(x_gt, sgmx_gt, sgmy_gt, x_pr, sgmx_pr, sgmy_pr):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    labels = ['SX', 'EB', 'SY']
    for axi, x_gti, sgmx_gti, sgmy_gti, x_pri, sgmx_pri, sgmy_pri, label in zip(ax, x_gt, sgmx_gt, sgmy_gt, x_pr,
                                                                                sgmx_pr, sgmy_pr, labels):
        axi.plot(x_gti, sgmx_gti, 'k.')
        axi.plot(x_pri, sgmx_pri, 'k-')

        axi.plot(x_gti, sgmy_gti, 'b.')
        axi.plot(x_pri, sgmy_pri, 'b-')

        axi.set_title(label)
        axi.set_xlabel(r'Stretch $\lambda [-]$')
        axi.set_ylabel(r'Cauchy stress $\sigma [MPa]$')
    return fig, ax


# play with ICNN a bit, how do we get that one to work
class ICNN_model():
    def __init__(self, params_I1, params_I2, params_I1_I4a, params_I1_I4s, params_I4a_I4s, theta):
        self.params_I1 = params_I1
        self.params_I2 = params_I2
        self.params_I1_I4a = params_I1_I4a
        self.params_I1_I4s = params_I1_I4s
        self.params_I4a_I4s = params_I4a_I4s

    # Psi1
    # note: for ICNN the prediction is the function not the gradient
    # but the sigma functions expect the gradient so taking derivative
    def Psi1(self, I1):
        # Note: I1norm = (I1-3)/normalization
        def f1(x): return icnn_forwardpass(x, self.params_I1)[0]
        # normalization = [I1_factor,I2_factor,I4a_factor,I4s_factor,Psi1_factor,Psi2_factor,... ]
        df1 = grad(f1)
        return vmap(df1)(I1[:, None])[:, 0]

    # Psi2
    # note: for ICNN the prediction is the function not the gradient
    def Psi2(self, I2):
        # Note: I2norm = (I2-3)/normalization
        def f2(x): return icnn_forwardpass(x, self.params_I2)[0]
        # normalization = [I1_factor,I2_factor,I4a_factor,I4s_factor,Psi1_factor,Psi2_factor,... ]
        df2 = grad(f2)
        return vmap(df2)(I2[:, None])[:, 0]

    # mixed term with I4a and I1
    # output is derivative wrt I1
    def Psi_4a_1(self, I1, I4a):
        alpha = self.params_I1_I4a[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I1) + (1 - alpha) * (I4a)
        def f_1_4a(x): return icnn_forwardpass(x, self.params_I1_I4a[:-1])[0]
        df_1_4a = grad(f_1_4a)
        # normalization = [I1_factor,I2_factor,I4a_factor,I4s_factor,Psi1_factor,Psi2_factor,... ]
        return vmap(df_1_4a)(K[:, None])[:, 0] * alpha

    # mixed term with I4a and I1
    # output is derivative wrt I4a
    def Psi_1_4a(self, I1, I4a):
        alpha = self.params_I1_I4a[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I1) + (1 - alpha) * (I4a)
        def f_1_4a(x): return icnn_forwardpass(x, self.params_I1_I4a[:-1])[0]
        df_1_4a = grad(f_1_4a)
        # normalization = [I1_factor,I2_factor,I4a_factor,I4s_factor,Psi1_factor,Psi2_factor,... ]
        return vmap(df_1_4a)(K[:, None])[:, 0] * (1 - alpha)

    # mixed term with I4a and I1
    # output is derivative wrt I1
    def Psi_4s_1(self, I1, I4s):
        alpha = self.params_I1_I4s[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I1) + (1 - alpha) * (I4s)
        def f_1_4s(x): return icnn_forwardpass(x, self.params_I1_I4s[:-1])[0]
        df_1_4s = grad(f_1_4s)
        # normalization = [I1_factor,I2_factor,I4a_factor,I4s_factor,Psi1_factor,Psi2_factor,... ]
        return vmap(df_1_4s)(K[:, None])[:, 0] * alpha

    # mixed term with I4s and I1
    # output is derivative wrt I4s
    def Psi_1_4s(self, I1, I4s):
        alpha = self.params_I1_I4s[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I1) + (1 - alpha) * (I4s)
        def f_1_4s(x): return icnn_forwardpass(x, self.params_I1_I4s[:-1])[0]
        df_1_4s = grad(f_1_4s)
        # normalization = [I1_factor,I2_factor,I4a_factor,I4s_factor,Psi1_factor,Psi2_factor,... ]
        return vmap(df_1_4s)(K[:, None])[:, 0] * (1 - alpha)

    # mixed term with I4a and I4s
    # output is derivative wrt I4a
    def Psi_4s_4a(self, I4a, I4s):
        alpha = self.params_I4a_I4s[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I4a) + (1 - alpha) * (I4s)
        def f_4a_4s(x): return icnn_forwardpass(x, self.params_I4a_I4s[:-1])[0]
        df_4a_4s = grad(f_4a_4s)
        # normalization = [I1_factor,I2_factor,I4a_factor,I4s_factor,Psi1_factor,Psi2_factor,... ]
        return vmap(df_4a_4s)(K[:, None])[:, 0] * alpha

    # mixed term with I4a and I4s
    # output is derivative wrt I4s
    def Psi_4a_4s(self, I4a, I4s):
        alpha = self.params_I4a_I4s[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I4a) + (1 - alpha) * (I4s)
        def f_4a_4s(x): return icnn_forwardpass(x, self.params_I4a_I4s[:-1])[0]
        df_4a_4s = grad(f_4a_4s)
        # normalization = [I1_factor,I2_factor,I4a_factor,I4s_factor,Psi1_factor,Psi2_factor,... ]
        return vmap(df_4a_4s)(K[:, None])[:, 0] * (1 - alpha)


class CANN_model():
    def __init__(self, params_I1, params_I2, params_I1_I4a, params_I1_I4s, params_I4a_I4s, theta):
        self.params_I1 = params_I1
        self.params_I2 = params_I2
        self.params_I1_I4a = params_I1_I4a
        self.params_I1_I4s = params_I1_I4s
        self.params_I4a_I4s = params_I4a_I4s

    # Psi1
    def Psi1(self, I1):
        # Note: I1norm = (I1-3)/normalization
        return CANN_dpsidInorm(I1, self.params_I1)[:, 0]

    # Psi2
    def Psi2(self, I2):
        # Note: I2norm = (I2-3)/normalization
        return CANN_dpsidInorm(I2, self.params_I2)[:, 0]

    # mixed term with I4a and I1
    # output is derivative wrt I1
    def Psi_4a_1(self, I1, I4a):
        alpha = self.params_I1_I4a[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I1) + (1 - alpha) * (I4a)
        return CANN_dpsidInorm(K, self.params_I1_I4a[:-1])[:, 0] * alpha

    # mixed term with I4a and I1
    # output is derivative wrt I4a
    def Psi_1_4a(self, I1, I4a):
        alpha = self.params_I1_I4a[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I1) + (1 - alpha) * (I4a)
        return CANN_dpsidInorm(K, self.params_I1_I4a[:-1])[:, 0] * (1 - alpha)

    # mixed term with I4s and I1
    # output is derivative wrt I1
    def Psi_4s_1(self, I1, I4s):
        alpha = self.params_I1_I4s[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I1) + (1 - alpha) * (I4s)
        return CANN_dpsidInorm(K, self.params_I1_I4s[:-1])[:, 0] * alpha

    # mixed term with I4a and I1
    # output is derivative wrt I4a
    def Psi_1_4s(self, I1, I4s):
        alpha = self.params_I1_I4s[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I1) + (1 - alpha) * (I4s)
        return CANN_dpsidInorm(K, self.params_I1_I4s[:-1])[:, 0] * (1 - alpha)

    # mixed term with I4s and I4a
    # output is derivative wrt I4a
    def Psi_4s_4a(self, I4a, I4s):
        alpha = self.params_I4a_I4s[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I4a) + (1 - alpha) * (I4s)
        return CANN_dpsidInorm(K, self.params_I4a_I4s[:-1])[:, 0] * alpha

    # mixed term with I4a and I4s
    # output is derivative wrt I4s
    def Psi_4a_4s(self, I4a, I4s):
        alpha = self.params_I4a_I4s[-1]
        alpha = 0.5 * (np.tanh(alpha) + 1.0)
        K = alpha * (I4a) + (1 - alpha) * (I4s)
        return CANN_dpsidInorm(K, self.params_I4a_I4s[:-1])[:, 0] * (1 - alpha)


@partial(jit, static_argnums=(2,))
def loss_sig_all(params, lamb_sigma, mdlnumber):
    if mdlnumber == 1:
        model = CANN_model(*params)
    elif mdlnumber == 2:
        model = ICNN_model(*params)
    else:
        model = NODE_model(*params)

    theta = params[-1]
    #jax.debug.print('{theta}', theta=theta)

    lambx = lamb_sigma[:, 0]
    lamby = lamb_sigma[:, 1]
    sigmax = lamb_sigma[:, 2]
    sigmay = lamb_sigma[:, 3]
    sigx, sigy = eval_Cauchy(lambx, lamby, model, theta)
    return np.mean((sigx - sigmax) ** 2 + (sigy - sigmay) ** 2)


@partial(jit, static_argnums=(2,))
def loss_sig_e(params, lamb_sigma, mdlnumber):
    if mdlnumber == 1:
        model = CANN_model(*params)
    elif mdlnumber == 2:
        model = ICNN_model(*params)
    else:
        model = NODE_model(*params)

    theta = params[-1]

    lambx = lamb_sigma[:ind_sx, 0]
    lamby = lamb_sigma[:ind_sx, 1]
    sigmax = lamb_sigma[:ind_sx, 2]
    sigmay = lamb_sigma[:ind_sx, 3]
    sigx, sigy = eval_Cauchy(lambx, lamby, model, theta)
    return np.mean((sigx - sigmax) ** 2 + (sigy - sigmay) ** 2)


@partial(jit, static_argnums=(2,))
def loss_sig_sx(params, lamb_sigma, mdlnumber):
    if mdlnumber == 1:
        model = CANN_model(*params)
    elif mdlnumber == 2:
        model = ICNN_model(*params)
    else:
        model = NODE_model(*params)
    lambx = lamb_sigma[ind_sx:ind_sy, 0]
    lamby = lamb_sigma[ind_sx:ind_sy, 1]
    sigmax = lamb_sigma[ind_sx:ind_sy, 2]
    sigmay = lamb_sigma[ind_sx:ind_sy, 3]
    sigx, sigy = eval_Cauchy(lambx, lamby, model)
    return np.mean((sigx - sigmax) ** 2 + (sigy - sigmay) ** 2)


@partial(jit, static_argnums=(2,))
def loss_sig_sy(params, lamb_sigma, mdlnumber):
    if mdlnumber == 1:
        model = CANN_model(*params)
    elif mdlnumber == 2:
        model = ICNN_model(*params)
    else:
        model = NODE_model(*params)
    lambx = lamb_sigma[ind_sy:, 0]
    lamby = lamb_sigma[ind_sy:, 1]
    sigmax = lamb_sigma[ind_sy:, 2]
    sigmay = lamb_sigma[ind_sy:, 3]
    sigx, sigy = eval_Cauchy(lambx, lamby, model)
    return np.mean((sigx - sigmax) ** 2 + (sigy - sigmay) ** 2)


@partial(jit, static_argnums=(0, 1,))
def step_jp(loss, mdlnumber, i, opt_state, X_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, mdlnumber)
    return opt_update(i, g, opt_state)


writer = SummaryWriter(
    './log/' + time.strftime('%m-%d %H:%M:%S', time.localtime()))


def train_jp(loss, mdlnumber, X, opt_state, key, nIter=10000, print_freq=1000):
    train_loss = []
    val_loss = []
    for it in range(nIter):
        opt_state = step_jp(loss, mdlnumber, it, opt_state, X)
        if (it + 1) % print_freq == 0 or it == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, mdlnumber)
            theta = params[-1]
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it + 1, train_loss_value)
            print(to_print)
            print(np.array(theta))
            writer.add_scalar(
                tag='loss', scalar_value=train_loss_value.tolist(), global_step=it)
    return get_params(opt_state), train_loss, val_loss


def init_icnn(key, layers=[1, 4, 3, 1]):
    key, subkey = random.split(key)
    params_I1 = init_params_icnn(layers, key)
    params_I2 = init_params_icnn(layers, key)
    params_I1_I4a = init_params_icnn(layers, key)
    params_I1_I4a.append(0.5)
    params_I1_I4s = init_params_icnn(layers, key)
    params_I1_I4s.append(0.5)
    params_I4a_I4s = init_params_icnn(layers, key)
    params_I4a_I4s.append(0.5)
    theta = [jnp.pi/2, jnp.pi/2]

    return [params_I1, params_I2, params_I1_I4a, params_I1_I4s, params_I4a_I4s, theta], key


def init_cann(key, layers=None):  # n params = 63
    key, subkey = random.split(key)
    params_I1 = init_params_cann(key)
    params_I2 = init_params_cann(key)
    params_I1_I4a = init_params_cann(key)
    params_I1_I4a.append(0.5)
    params_I1_I4s = init_params_cann(key)
    params_I1_I4s.append(0.5)
    params_I4a_I4s = init_params_cann(key)
    params_I4a_I4s.append(0.5)
    theta = [jnp.pi / 2, jnp.pi / 2]

    return [params_I1, params_I2, params_I1_I4a, params_I1_I4s, params_I4a_I4s, theta], key


params_icnn_all, key = init_icnn(key)
opt_init, opt_update, get_params = optimizers.adam(1.e-3)  # Original: 1.e-4
opt_state = opt_init(params_icnn_all)
mdlnumber = 2

params_icnn_all, train_loss, val_loss = train_jp(
    loss_sig_all, mdlnumber, lamb_sigma, opt_state, key, nIter=100000)  # Original 100000
model = ICNN_model(*params_icnn_all)
lambx = lamb_sigma[:, 0]
lamby = lamb_sigma[:, 1]
sigmax = lamb_sigma[:, 2]
sigmay = lamb_sigma[:, 3]
theta = params_icnn_all[-1]
sigx, sigy = eval_Cauchy(lambx, lamby, model, theta)

fig, ax = plotstresses([lambx[ind_sx:ind_sy], lambx[0:ind_sx], lamby[ind_sy:]],
                       [sigmax[ind_sx:ind_sy], sigmax[0:ind_sx], sigmax[ind_sy:]],
                       [sigmay[ind_sx:ind_sy], sigmay[0:ind_sx], sigmay[ind_sy:]],
                       [lambx[ind_sx:ind_sy], lambx[0:ind_sx], lamby[ind_sy:]],
                       [sigx[ind_sx:ind_sy], sigx[0:ind_sx], sigx[ind_sy:]],
                       [sigy[ind_sx:ind_sy], sigy[0:ind_sx], sigy[ind_sy:]])
plt.show()

# params_cann_all, key = init_cann(key)
# opt_init, opt_update, get_params = optimizers.adam(2.e-4) #Original: 1.e-4
# opt_state = opt_init(params_cann_all)
# mdlnumber = 1
# params_cann_all, train_loss, val_loss = train_jp(loss_sig_all, mdlnumber, lamb_sigma, opt_state, key, nIter = 100000)
# model = CANN_model(*params_cann_all)
# lambx = lamb_sigma[:,0]
# lamby = lamb_sigma[:,1]
# sigmax = lamb_sigma[:,2]
# sigmay = lamb_sigma[:,3]
# theta = params_cann_all[-1]
# sigx,sigy = eval_Cauchy(lambx,lamby, model, theta)
#
# fig, ax = plotstresses([lambx[ind_sx:ind_sy], lambx[0:ind_sx], lamby[ind_sy:]],
#                        [sigmax[ind_sx:ind_sy], sigmax[0:ind_sx], sigmax[ind_sy:]],
#                        [sigmay[ind_sx:ind_sy], sigmay[0:ind_sx], sigmay[ind_sy:]],
#                        [lambx[ind_sx:ind_sy], lambx[0:ind_sx], lamby[ind_sy:]],
#                        [sigx[ind_sx:ind_sy], sigx[0:ind_sx], sigx[ind_sy:]],
#                        [sigy[ind_sx:ind_sy], sigy[0:ind_sx], sigy[ind_sy:]])
# plt.show()
