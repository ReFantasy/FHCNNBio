import jax
from jax import config, jit, random, vmap
import jax.numpy as jnp
import numpy as np
from functools import partial
import torch
import torch.nn.functional as F
from jax.nn import softplus
from jax.lax import scan

##------------------------------------##
# Common functions
# parameter initialization
# forward pass with and without biases
# train and step functions
##------------------------------------##

def init_params(layers, key):
    '''
    初始化每一层的权重并放入列表中
    :param layers: e.g. [2,3,4]
    :param key:
    :return: matrix: [[2,3], [3,4]]
    '''
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = jnp.sqrt(2 / (layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(
            subkey, (layers[i], layers[i + 1])) * std_glorot)
    return Ws


@jit
def forward_pass(H, Ws):
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i])
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1])
    return Y


def init_params_posb(layers, key):
    '''
    带有偏置项的 init_params, 偏置项只有最后的输出层
    :param layers:
    :param key:
    :return:
    '''
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = jnp.sqrt(2 / (layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(
            subkey, (layers[i], layers[i + 1])) * std_glorot)
    b = jnp.zeros(layers[i + 1])
    return Ws, b


def forward_pass_posb(H, params):
    Ws, b = params
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i])
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1]) + jnp.exp(b)  # We want a positive bias
    return Y


def init_params_b(layers, key):
    '''
    每一层都带有偏置项的权重初始化函数
    :param layers:
    :param key:
    :return:
    '''
    Ws = []
    bs = []
    for i in range(len(layers) - 1):
        std_glorot = jnp.sqrt(2 / (layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(
            subkey, (layers[i], layers[i + 1])) * std_glorot)
        bs.append(jnp.zeros(layers[i + 1]))
    return (Ws, bs)


@jit
def forward_pass_b(H, params):
    Ws = params[0]
    bs = params[1]
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i]) + bs[i]
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1]) + bs[-1]
    return Y


@partial(jit, static_argnums=(0,))
def step(loss, i, opt_state, X_batch, Y_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, Y_batch)
    return opt_update(i, g, opt_state)


def train(loss, X, Y, opt_state, key, nIter=10000, batch_size=10):
    train_loss = []
    val_loss = []
    for it in range(nIter):
        key, subkey = random.split(key)
        idx_batch = random.choice(
            subkey, X.shape[0], shape=(batch_size,), replace=False)
        opt_state = step(loss, it, opt_state, X[idx_batch, :], Y[idx_batch, :])
        # opt_state = step(loss, it, opt_state, X, Y)
        if it % 100 == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, Y)
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss, val_loss


##------------------------------------##
# ICNN functions
# ininialization for icnn
# special forward pass
##------------------------------------##

def init_params_icnn(layers, key):
    Wz = []
    Wy = []
    bs = []

    std_glorot = jnp.sqrt(2 / (layers[0] + layers[1]))
    key, subkey = random.split(key)
    Wy.append(random.normal(subkey, (layers[0], layers[1])) * std_glorot)
    bs.append(jnp.zeros(layers[1]))

    for i in range(1, len(layers) - 1):
        std_glorot = jnp.sqrt(2 / (layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Wz.append(-3 + random.normal(subkey,
                  (layers[i], layers[i + 1])) * std_glorot)
        Wy.append(random.normal(
            subkey, (layers[0], layers[i + 1])) * std_glorot)
        bs.append(jnp.zeros(layers[i + 1]))
    return [Wz, Wy, bs]


@jit
def icnn_forwardpass(Y, params):
    Wz, Wy, bs = params
    N_layers = len(Wy)
    Z = softplus(jnp.matmul(Y, jnp.exp(Wy[0])) + bs[0]) ** 2
    for i in range(1, N_layers - 1):
        Z = jnp.matmul(Z, jnp.exp(Wz[i - 1])) + \
            jnp.matmul(Y, jnp.exp(Wy[i])) + bs[i]
        Z = softplus(Z) ** 2
    Z = jnp.matmul(Z, jnp.exp(Wz[-1])) + \
        jnp.matmul(Y, jnp.exp(Wy[-1])) + bs[-1]
    return Z


##----------------------------------------------##
# CANN functions
# initialize weights
# eval energy for isotropic material
# eval derivatives of energy isotropic material
##----------------------------------------------##

def init_params_cann(key):
    Ws = []
    std_glorot = jnp.sqrt(1. / 2.)
    for i in range(4):
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (3, 1)) * std_glorot)
    return Ws


def CANN_psi(Inorm, Ws):
    # Inorm assume normalized scalar, e.g. = (I1-3)/normalization
    # powers are not fully connected, defined operation
    # so dont see how to do general, rather, 3 different input
    YIp = jnp.array([Inorm, Inorm ** 2, Inorm ** 3])
    # for each of the powers, multiply by positive weights
    # then pass through activation functions
    # 1. Identity activation
    # note: element-wise operation
    Yh1Ip = YIp * Ws[0] ** 2
    # 2. Exponential activation
    # note: element-wise operations
    Yh2Ip = jnp.exp(YIp * Ws[1] ** 2) - jnp.ones(YIp.shape)
    # multiply by the next set of weights and add to output
    # note: here yes dot products and adding to scalar
    Z = jnp.dot(Yh1Ip.transpose(), Ws[2] ** 2) + \
        jnp.dot(Yh2Ip.transpose(), Ws[3] ** 2)
    return Z


def CANN_dpsidInorm(Inorm, Ws):
    # Inorm assume normalized scalar, e.g. = (I1-3)/normalization
    # powers are not fully connected, defined operation
    # so dont see how to do general, rather, 3 different input
    YIp = jnp.array([Inorm, Inorm ** 2, Inorm ** 3])
    dYIpdI = jnp.array([jnp.ones(Inorm.shape), 2. * Inorm, 3. * Inorm ** 2])
    # for each of the powers, multiply by positive weights
    # then pass through activation functions, dot and not matmul
    # 1. Identity activation
    # note: element-wise operations
    Yh1Ip = YIp * Ws[0] ** 2
    dYh1IpdI = dYIpdI * Ws[0] ** 2
    # 2. Exponential activation
    # note: element-wise operation
    Yh2Ip = jnp.exp(YIp * Ws[1] ** 2) - jnp.ones(YIp.shape)
    dYh2IpdI = jnp.exp(YIp * Ws[1] ** 2) * Ws[1] ** 2 * dYIpdI
    # multiply by the next set of weights and add to output
    Z = jnp.dot(Yh1Ip.transpose(), Ws[2] ** 2) + \
        jnp.dot(Yh2Ip.transpose(), Ws[3] ** 2)
    dZdI = jnp.dot(dYh1IpdI.transpose(),
                   Ws[2] ** 2) + jnp.dot(dYh2IpdI.transpose(), Ws[3] ** 2)
    return dZdI


##-------------------------------------##
## NODE functions
# integration of the neural network
# compute PK2 using NODEs for derivatives
# prediction of sigma by push fwd
##-------------------------------------##

# NODE integration with odeint from jax
@jit
def NODE_old(y0, params):
    f = lambda y, t: forward_pass(jnp.array([y]), params)  # fake time argument for ODEint
    return odeint(f, y0, jnp.array([0.0, 1.0]))[-1]  # integrate between 0 and 1 and return the results at 1


# The same function as NODE_old except using Euler integration
@jit
def NODE(y0, params, steps=100):
    t0 = 0.0
    dt = 1.0 / steps
    body_func = lambda y, t: (y + forward_pass(jnp.array([y]), params)[0] * dt, None)
    out, _ = scan(body_func, y0, jnp.linspace(0, 1, steps), length=steps)
    return out


NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)


@jit
def NODE_posb(y0, params, steps=100):
    t0 = 0.0
    dt = 1.0 / steps
    body_func = lambda y, t: (y + forward_pass_posb(jnp.array([y]), params)[0] * dt, None)
    out, _ = scan(body_func, y0, jnp.linspace(0, 1, steps), length=steps)
    return out


NODE_posb_vmap = vmap(NODE_posb, in_axes=(0, None), out_axes=0)


# PK2 stress prediction using NODE.
# this is the main function for an arbitrary deformation C
@jit
def NODE_S(C, params):
    I1_params, I2_params, Iv_params, Iw_params, J1_params, J2_params, J3_params, J4_params, J5_params, J6_params, I_weights, theta, Psi1_bias, Psi2_bias = params
    a = 1 / (1 + jnp.exp(-I_weights))
    v0 = jnp.array([jnp.cos(theta), jnp.sin(theta), 0])
    w0 = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0])
    V0 = jnp.outer(v0, v0)
    W0 = jnp.outer(w0, w0)
    I1 = jnp.trace(C)
    C2 = jnp.einsum('ij,jk->ik', C, C)
    I2 = 0.5 * (I1 ** 2 - jnp.trace(C2))
    Iv = jnp.einsum('ij,ij', C, V0)
    Iw = jnp.einsum('ij,ij', C, W0)
    Cinv = jnp.linalg.inv(C)

    I1 = I1 - 3
    I2 = I2 - 3
    Iv = Iv - 1
    Iw = Iw - 1
    J1 = a[0] * I1 + (1 - a[0]) * I2
    J2 = a[1] * I1 + (1 - a[1]) * Iv
    J3 = a[2] * I1 + (1 - a[2]) * Iw
    J4 = a[3] * I2 + (1 - a[3]) * Iv
    J5 = a[4] * I2 + (1 - a[4]) * Iw
    J6 = a[5] * Iv + (1 - a[5]) * Iw

    Iv, Iw, J1, J2, J3, J4, J5, J6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    Psi1 = NODE(I1, I1_params)
    Psi2 = NODE(I2, I2_params)
    Psiv = NODE(Iv, Iv_params)
    Psiw = NODE(Iw, Iw_params)
    Phi1 = NODE(J1, J1_params)
    Phi2 = NODE(J2, J2_params)
    Phi3 = NODE(J3, J3_params)
    Phi4 = NODE(J4, J4_params)
    Phi5 = NODE(J5, J5_params)
    Phi6 = NODE(J6, J6_params)

    Psiv = jnp.max([Psiv, 0])
    Psiw = jnp.max([Psiw, 0])
    Phi1 = jnp.max([Phi1, 0])
    Phi2 = jnp.max([Phi2, 0])
    Phi3 = jnp.max([Phi3, 0])
    Phi4 = jnp.max([Phi4, 0])
    Phi5 = jnp.max([Phi5, 0])
    Phi6 = jnp.max([Phi6, 0])

    Psi1 = Psi1 + a[0] * Phi1 + a[1] * Phi2 + a[2] * Phi3 + jnp.exp(Psi1_bias)
    Psi2 = Psi2 + (1 - a[0]) * Phi1 + a[3] * Phi4 + a[4] * Phi5 + jnp.exp(Psi2_bias)
    Psiv = Psiv + (1 - a[1]) * Phi2 + (1 - a[3]) * Phi4 + a[5] * Phi6
    Psiw = Psiw + (1 - a[2]) * Phi3 + (1 - a[4]) * Phi5 + (1 - a[5]) * Phi6

    p = -C[2, 2] * (2 * Psi1 + 2 * Psi2 * ((I1 + 3) - C[2, 2]) + 2 * Psiv * V0[2, 2] + 2 * Psiw * W0[2, 2])
    S = p * Cinv + 2 * Psi1 * jnp.eye(3) + 2 * Psi2 * ((I1 + 3) * jnp.eye(3) - C) + 2 * Psiv * V0 + 2 * Psiw * W0
    return S


NODE_S_vmap = vmap(NODE_S, in_axes=0, out_axes=0)


# Prediction of cauchy stress by computing PK2 and push forward
@jit
def NODE_sigma(F, params):
    C = jnp.dot(F.T, F)
    S = NODE_S(C, params)
    return jnp.einsum('ij,jk,kl->il', F, S, F.T)


NODE_sigma_vmap = vmap(NODE_sigma, in_axes=(0, None), out_axes=0)


# Prediction of stress for particular type of deformation gradient
@jit
def NODE_lm2sigma(lamb, params):
    lamb1 = lamb[0]
    lamb2 = lamb[1]
    lamb3 = 1 / (lamb1 * lamb2)
    F = jnp.array([[lamb1, 0, 0],
                   [0, lamb2, 0],
                   [0, 0, lamb3]])
    return NODE_sigma(F, params)[[0, 1], [0, 1]]


NODE_lm2sigma_vmap = vmap(NODE_lm2sigma, in_axes=(0, None), out_axes=0)

# ---------------------------------------------------------
#          torch version
# ---------------------------------------------------------
# ----------------------------------------------
#    CANN 模型
# ----------------------------------------------


class torch_cann_psi(torch.nn.Module):
    def __init__(self, key=None):
        super(torch_cann_psi, self).__init__()

        Ws = []
        std_glorot = jnp.sqrt(1. / 2.)
        for i in range(4):
            key, subkey = random.split(key)
            Ws.append(random.normal(subkey, (3, 1)) * std_glorot)
        w0 = torch.from_numpy(np.array(Ws[0]))
        w1 = torch.from_numpy(np.array(Ws[1]))
        w2 = torch.from_numpy(np.array(Ws[2]))
        w3 = torch.from_numpy(np.array(Ws[3]))

        self.w0 = torch.nn.Parameter(w0.reshape((1, 3)))
        self.w1 = torch.nn.Parameter(w1.reshape((1, 3)))
        self.w2 = torch.nn.Parameter(w2.reshape((3, 1)))
        self.w3 = torch.nn.Parameter(w3.reshape((3, 1)))

        # print(self.w0)
        # print(self.w1)
        # print(self.w2)
        # print(self.w3)

    def forward(self, x):
        x = torch.cat((x, x ** 2, x ** 3), dim=1)
        Yh1Ip = x * self.w0 ** 2
        Yh2Ip = torch.exp(x * self.w1 ** 2) - torch.ones_like(x)
        z = torch.matmul(Yh1Ip, self.w2 ** 2) + \
            torch.matmul(Yh2Ip, self.w3 ** 2)
        return z


# ----------------------------------------------
#    ICNN 模型 1
# ----------------------------------------------
class NonNegativeLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, key=None, std_glorot=None):
        super(NonNegativeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # init weight
        if std_glorot == None:
            std_glorot = jnp.sqrt(2.0 / (in_features + out_features))

        w = random.normal(key, (in_features, out_features)) * std_glorot
        w = jnp.transpose(w)
        self.weight = torch.nn.Parameter(
            torch.from_numpy(np.array(w)).to(torch.float))

        if bias == True:
            self.bs = jnp.zeros(out_features)
            self.bias = torch.nn.Parameter(torch.from_numpy(
                np.array(self.bs)).to(torch.float))
        else:
            self.bs = None
            self.bias = None

    def forward(self, input):
        return F.linear(input, torch.exp(self.weight), self.bias)

    def get_parameters(self):
        return [jnp.transpose(jnp.array(self.weight.data.clone().detach().cpu().numpy())), self.bs]


class torch_icnn_psi(torch.nn.Module):
    def __init__(self, layers=[3, 2, 5, 7, 4], key=None):
        super(torch_icnn_psi, self).__init__()

        # var just for print info
        Wz = []
        Wy = []
        bs = []

        self.layer_k = len(layers) - 1

        layers_modules_y = []
        layers_modules_z = []

        key, subkey = random.split(key)
        net_y = NonNegativeLinear(layers[0], layers[1], bias=True, key=subkey)
        layers_modules_y.append(net_y)
        # print info
        params = net_y.get_parameters()
        Wy.append(params[0])
        bs.append(params[1])

        for i in range(1, len(layers) - 1):
            key, subkey = random.split(key)

            net_y = NonNegativeLinear(layers[0], layers[i + 1], bias=True, key=subkey,
                                      std_glorot=jnp.sqrt(2 / (layers[i] + layers[i + 1])))
            layers_modules_y.append(net_y)
            #  print info
            params = net_y.get_parameters()
            Wy.append(params[0])
            bs.append(params[1])

            net_z = NonNegativeLinear(
                layers[i], layers[i + 1], bias=False, key=subkey)
            new_weight = net_z.weight.data - 3.0
            net_z.weight.data = new_weight
            layers_modules_z.append(net_z)
            #  print info
            params = net_z.get_parameters()
            Wz.append(params[0])

        self.w_y = torch.nn.ModuleList(layers_modules_y)
        self.w_z = torch.nn.ModuleList(layers_modules_z)

        self.params = [Wz, Wy, bs]

    def get_parameters(self):
        return self.params

    def forward(self, x):
        z = F.softplus(self.w_y[0](x)) ** 2

        for i in range(1, self.layer_k - 1):
            out_z = self.w_z[i - 1](z)
            out_x = self.w_y[i](x)
            z = F.softplus(out_z + out_x) ** 2

        last_index = self.layer_k - 1
        out_z = self.w_z[last_index - 1](z)
        out_x = self.w_y[last_index](x)
        z = out_z + out_x

        return z


def icnn_valid():
    I_value = np.random.random()
    I_t = torch.tensor([I_value]).requires_grad_().reshape((-1, 1))
    I_j = jnp.array([I_value])

    # jax random
    key_value = np.random.randint(low=np.iinfo(
        np.int32).min, high=np.iinfo(np.int32).max, dtype=np.int32)
    key = random.PRNGKey(key_value)

    # ----------------------------------------------
    #    验证 CANN 能量
    # ----------------------------------------------
    print('# ----------------------------------------------')
    print('#    验证 CANN 能量')
    print('# ----------------------------------------------')
    torch_psi = torch_cann_psi(key)
    y = torch_psi(I_t)
    print(y)
    Ws = init_params_cann(key)
    z = CANN_psi(I_j, Ws)
    print(z)
    print('\n')

    # ----------------------------------------------
    #    验证 CANN 能量关于 不变量的导数
    # ----------------------------------------------
    print('# ----------------------------------------------')
    print('#    验证 CANN 能量关于 不变量的导数')
    print('# ----------------------------------------------')
    torch_psi = torch_cann_psi(key)
    y = torch_psi(I_t)
    print(gradients(y, I_t))

    Ws = init_params_cann(key)
    z = CANN_dpsidInorm(I_j, Ws)
    print(z)
    print('\n')

    # ----------------------------------------------
    #    验证 ICNN 能量
    # ----------------------------------------------
    print('# ----------------------------------------------')
    print('#    验证 ICNN 能量')
    print('# ----------------------------------------------')
    layers = [1, 3, 4, 1]
    torch_psi = torch_icnn_psi(layers, key)
    # print(torch_psi.get_parameters()[0])
    y = torch_psi(I_t)
    print(y)

    params_I1 = init_params_icnn(layers, key)
    # print(params_I1[0])
    z = icnn_forwardpass(I_j, params_I1)
    print(z)
    print('\n')

    # ----------------------------------------------
    #    验证 ICNN 能量关于 不变量的导数
    # ----------------------------------------------
    print('# ----------------------------------------------')
    print('#    验证 ICNN 能量关于 不变量的导数')
    print('# ----------------------------------------------')
    layers = [1, 3, 4, 1]
    torch_psi = torch_icnn_psi(layers, key)
    y = torch_psi(I_t)
    print(gradients(y, I_t))

    params_I1 = init_params_icnn(layers, key)

    def aux(Y, params):
        z = icnn_forwardpass(Y, params)
        return z[0]

    z = grad(aux)(I_j, params_I1)
    print(z)
    print('\n')

    # ----------------------------------------------
    #    验证 ICNN 初始化参数
    # ----------------------------------------------
    print('# ----------------------------------------------')
    print('#    验证 ICNN 初始化参数')
    print('# ----------------------------------------------')
    layers = [1, 3, 4, 1]
    torch_psi = torch_icnn_psi(layers, key)
    print(torch_psi.get_parameters())
    print('#---------------------------------------')
    params_I1 = init_params_icnn(layers, key)
    print(params_I1)


if __name__ == '__main__':
    key = random.PRNGKey(0)

    # layers = [2, 3, 4]
    # Ws, bs = init_params_b(layers=layers, key=key)
    # for i, w in enumerate(Ws):
    #     print(w)
    # for i, bias in enumerate(bs):
    #    print(bias)

    # key, subkey = random.split(key)
    # H = random.normal(subkey, (5, 2))
    # y = forward_pass_b(H, [Ws, bs])
    # print(y)

    # Ws = init_params_cann(key)
    # Ws[0] = jnp.array([[1.], [2.], [3.]])
    # Ws[1] = jnp.array([[4.], [5.], [6.]])
    # Ws[2] = jnp.array([[7.], [8.], [9.]])
    # Ws[3] = jnp.array([[10.], [11.], [12.]])
    #
    # Inorm = jnp.array([0.9, 0.2, 0.3, 0.6])

    def my_init_params_icnn(layers, key):
        Wz = []
        Wy = []
        bs = []

        std_glorot = jnp.sqrt(2 / (layers[0] + layers[1]))
        Wy.append(random.normal(random.PRNGKey(0),
                  (layers[0], layers[1])) * std_glorot)
        bs.append(jnp.zeros(layers[1]))

        for i in range(1, len(layers) - 1):
            std_glorot = jnp.sqrt(2 / (layers[i] + layers[i + 1]))
            Wz.append(random.normal(
                key, (layers[i], layers[i + 1])) * std_glorot)
            std_glorot = jnp.sqrt(2 / (layers[0] + layers[i + 1]))
            Wy.append(random.normal(
                key, (layers[0], layers[i + 1])) * std_glorot)
            bs.append(jnp.zeros(layers[i + 1]))
        # for wy in Wy:
        #    print(wy)
        return [Wz, Wy, bs]

    def my_icnn_forwardpass(Y, params):
        Wz, Wy, bs = params
        N_layers = len(Wy)
        Z = softplus(jnp.matmul(Y, jnp.exp(Wy[0])) + bs[0]) ** 2
        for i in range(1, N_layers - 1):
            Z = jnp.matmul(Z, jnp.exp(Wz[i - 1])) + \
                jnp.matmul(Y, jnp.exp(Wy[i])) + bs[i]
            Z = softplus(Z) ** 2
        Z = jnp.matmul(Z, jnp.exp(Wz[-1])) + \
            jnp.matmul(Y, jnp.exp(Wy[-1])) + bs[-1]
        return Z

    key = random.PRNGKey(0)
    params_I1 = init_params_icnn([1, 3, 4, 1], key)
    x = jnp.array([2.0])
    y = my_icnn_forwardpass(x, params_I1)
    print(y)
    # print(params_I1[0])

    #x = random.uniform(key, (4, 3))
    #key,subkey = random.split(random.PRNGKey(0))

    # ficnn = FICNN([1, 3, 4, 1], random.PRNGKey(0))
    # x = torch.tensor([2.0]).reshape((-1, 1))
    # y = ficnn(x)
    # print(y)

    # linear = NonNegativeLinear(1,4, True, random.PRNGKey(0))
    # print(linear)

    # test cann
    # Z = CANN_psi(Inorm, Ws)
    # print(Z)
    # dZdI = CANN_dpsidInorm(Inorm, Ws)
    # print(dZdI)
    # def cann_psi_warpper(Inorm, Ws):
    #     Z = CANN_psi(Inorm, Ws)
    #     Z = jnp.squeeze(Z)
    #     return jnp.sum(Z, axis=0)
    # autodiff_dzdi = grad(cann_psi_warpper)(Inorm, Ws)
    # print(autodiff_dzdi)
