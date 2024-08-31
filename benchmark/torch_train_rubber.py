from src.torch_icnn_cann import *
import sys
from jax import random, config
import pandas as pd
import matplotlib.pyplot as plt
# torch.set_printoptions(precision=8)

config.update("jax_enable_x64", True)

sys.path.append('..')

key = random.PRNGKey(0)

I1_factor = 30
Psi1_factor = 0.3
I2_factor = 250
Psi2_factor = 0.001
normalization = [I1_factor, I2_factor, Psi1_factor, Psi2_factor]


class TorchCANN_model():
    def __init__(self, normalization, key):
        self.psi_i1 = torch_cann_psi(key)
        self.psi_i2 = torch_cann_psi(key)
        self.normalization = normalization

    # Psi1
    def Psi1norm(self, I1norm):
        # Note: I1norm = (I1-3)/normalization
        # return CANN_dpsidInorm(I1norm, self.params_I1)[:, 0] / normalization[0]
        psi = self.psi_i1(I1norm)
        dpsidI = gradients(psi, I1norm) / self.normalization[0]
        return dpsidI

    # Psi2
    def Psi2norm(self, I2norm):
        # Note: I2norm = (I2-3)/normalization
        # return CANN_dpsidInorm(I2norm, self.params_I2)[:, 0] / normalization[1]
        psi = self.psi_i2(I2norm)
        dpsidI = gradients(psi, I2norm) / self.normalization[1]
        return dpsidI


class TorchICNN_model():
    def __init__(self, layers, normalization, key):
        self.psi_i1 = torch_icnn_psi(key=key, layers=layers)
        self.psi_i2 = torch_icnn_psi(key=key, layers=layers)
        self.normalization = normalization

    # Psi1
    def Psi1norm(self, I1norm):
        # Note: I1norm = (I1-3)/normalization
        # return CANN_dpsidInorm(I1norm, self.params_I1)[:, 0] / normalization[0]
        psi = self.psi_i1(I1norm)
        dpsidI = gradients(psi, I1norm) / self.normalization[0]
        return dpsidI

    # Psi2
    def Psi2norm(self, I2norm):
        # Note: I2norm = (I2-3)/normalization
        # return CANN_dpsidInorm(I2norm, self.params_I2)[:, 0] / normalization[1]
        psi = self.psi_i2(I2norm)
        dpsidI = gradients(psi, I2norm) / self.normalization[1]
        return dpsidI


def Torch_P11_UT(lamb, model, normalization):
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


def Numpy_P11_UT(lamb, model, normalization):
    lam = torch.from_numpy(lamb).to(
        torch.float32).reshape((-1, 1)).requires_grad_()
    P11 = Torch_P11_UT(lam, model, normalization)
    return P11.detach().numpy()


def Torch_P11_ET(lamb, model, normalization):
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


def Numpy_P11_ET(lamb, model, normalization):
    lam = torch.from_numpy(lamb).to(
        torch.float32).reshape((-1, 1)).requires_grad_()
    P11 = Torch_P11_ET(lam, model, normalization)
    return P11.detach().numpy()


def Torch_P11_PS(lamb, model, normalization):
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


def Numpy_P11_PS(lamb, model, normalization):
    lam = torch.from_numpy(lamb).to(
        torch.float32).reshape((-1, 1)).requires_grad_()
    P11 = Torch_P11_PS(lam, model, normalization)
    return P11.detach().numpy()


def validate_model():
    I1_value = np.random.random()
    I2_value = np.random.random()
    I1_torch = torch.tensor([I1_value]).requires_grad_().reshape((-1, 1))
    I1_jax = jnp.array([I1_value])
    I2_torch = torch.tensor([I2_value]).requires_grad_().reshape((-1, 1))
    I2_jax = jnp.array([I2_value])

    # jax random
    key_value = np.random.randint(low=np.iinfo(
        np.int32).min, high=np.iinfo(np.int32).max, dtype=np.int32)
    key = random.PRNGKey(key_value)

    I1_factor = 30
    Psi1_factor = 0.3
    I2_factor = 250
    Psi2_factor = 0.001
    normalization = [I1_factor, I2_factor, Psi1_factor, Psi2_factor]

    # ----------------------------------------------
    #    验证 CANN 模型
    # ----------------------------------------------
    # print('# ----------------------------------------------')
    # print('#    验证 CANN 模型')
    # print('# ----------------------------------------------')
    key, subkey = random.split(key)
    torch_cann_model = TorchCANN_model(normalization, key)
    print(torch_cann_model.Psi1norm(I1_torch))
    print(torch_cann_model.Psi2norm(I2_torch))

    def init_cann(key, layers=None):
        params_I1 = init_params_cann(key)
        params_I2 = init_params_cann(key)
        return [params_I1, params_I2]

    params_cann_all = init_cann(key)
    jax_cann_model = CANN_model(
        params_cann_all[0], params_cann_all[1], normalization)
    print(jax_cann_model.Psi1norm(I1_jax))
    print(jax_cann_model.Psi2norm(I2_jax))
    print('\n')

    # ----------------------------------------------
    #    验证 ICNN 模型
    # ----------------------------------------------
    # print('# ----------------------------------------------')
    # print('#    验证 ICNN 模型')
    # print('# ----------------------------------------------')
    # layers = [1, 3, 4, 1]
    #
    # torch_icnn_model = TorchICNN_model(layers, normalization, key)
    # print(torch_icnn_model.Psi1norm(I1_torch))
    # print(torch_icnn_model.Psi2norm(I2_torch))
    #
    # params_icnn_all = init_icnn(key)
    # jax_icnn_model = ICNN_model(params_icnn_all[0], params_icnn_all[1], normalization)
    # print(jax_icnn_model.Psi1norm(I1_jax))
    # print(jax_icnn_model.Psi2norm(I2_jax))
    # print('\n')

    # ----------------------------------------------'
    #    其他函数')
    # ----------------------------------------------')
    print('# ----------------------------------------------')
    print('#    其他函数')
    print('# ----------------------------------------------')
    # read the data
    UTdata = pd.read_csv('../Data/UT20.csv')
    ETdata = pd.read_csv('../Data/ET20.csv')
    PSdata = pd.read_csv('../Data/PS20.csv')

    # stack into single array
    P11_data = np.hstack(
        [UTdata['P11'].to_numpy(), ETdata['P11'].to_numpy(), PSdata['P11'].to_numpy()])
    F11_data = np.hstack(
        [UTdata['F11'].to_numpy(), ETdata['F11'].to_numpy(), PSdata['F11'].to_numpy()])
    # indices for the three data sets
    indET = len(UTdata['P11'])
    indPS = indET + len(ETdata['P11'])

    lamUT = F11_data[0:indET]
    lamET = F11_data[indET:indPS]
    lamPS = F11_data[indPS:]
    tensor_lamUT = torch.from_numpy(lamUT).to(
        torch.float32).reshape((-1, 1)).requires_grad_()
    tensor_lamET = torch.from_numpy(lamET).to(
        torch.float32).reshape((-1, 1)).requires_grad_()
    tensor_lamPS = torch.from_numpy(lamPS).to(
        torch.float32).reshape((-1, 1)).requires_grad_()
    torch_P11 = Torch_P11_PS(tensor_lamPS, torch_cann_model, normalization)
    print(torch_P11)

    jax_P11 = P11_PS(lamPS, jax_cann_model, normalization)
    print(jax_P11)


def train_cann_model():
    # read the data
    UTdata = pd.read_csv('../Data/UT20.csv')
    ETdata = pd.read_csv('../Data/ET20.csv')
    PSdata = pd.read_csv('../Data/PS20.csv')

    # stack into single array
    P11_data = np.hstack(
        [UTdata['P11'].to_numpy(), ETdata['P11'].to_numpy(), PSdata['P11'].to_numpy()])
    F11_data = np.hstack(
        [UTdata['F11'].to_numpy(), ETdata['F11'].to_numpy(), PSdata['F11'].to_numpy()])
    # indices for the three data sets
    indET = len(UTdata['P11'])
    indPS = indET + len(ETdata['P11'])

    lamUT = F11_data[0:indET]
    lamET = F11_data[indET:indPS]
    lamPS = F11_data[indPS:]

    tensor_lamUT = torch.from_numpy(lamUT).to(
        torch.float32).reshape((-1, 1)).requires_grad_()
    tensor_lamET = torch.from_numpy(lamET).to(
        torch.float32).reshape((-1, 1)).requires_grad_()
    tensor_lamPS = torch.from_numpy(lamPS).to(
        torch.float32).reshape((-1, 1)).requires_grad_()

    P11UT_exp = torch.from_numpy(P11_data[0:indET]).to(
        torch.float).reshape((-1, 1))
    P11ET_exp = torch.from_numpy(P11_data[indET:indPS]).to(
        torch.float).reshape((-1, 1))
    P11PS_exp = torch.from_numpy(P11_data[indPS:]).to(
        torch.float).reshape((-1, 1))

    key = random.PRNGKey(12)
    key, subkey = random.split(key)
    torch_cann_model = TorchCANN_model(normalization, key)
    parameters = list(torch_cann_model.psi_i1.parameters()) + \
        list(torch_cann_model.psi_i2.parameters())
    optimizer = torch.optim.Adam(parameters, lr=2e-4)

    mse_loss = torch.nn.MSELoss()

    torch_cann_model.psi_i1.train()
    torch_cann_model.psi_i2.train()

    epochs = 20000
    for i in range(epochs):
        optimizer.zero_grad()

        P11UT_pr = Torch_P11_UT(tensor_lamUT, torch_cann_model, normalization)
        P11ET_pr = Torch_P11_ET(tensor_lamET, torch_cann_model, normalization)
        P11PS_pr = Torch_P11_PS(tensor_lamPS, torch_cann_model, normalization)

        loss = mse_loss(P11UT_pr, P11UT_exp) + mse_loss(P11ET_pr,
                                                        P11ET_exp) + mse_loss(P11PS_pr, P11PS_exp)
        loss.backward()
        if (i + 1) % 1000 == 0:
            print(i+1, loss.item())
        #writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=i)
        optimizer.step()

    # predict
    torch_cann_model.psi_i1.eval()
    torch_cann_model.psi_i2.eval()
    lamUT_vec = np.linspace(1, UTdata['F11'].iloc[-1], 30)
    lamET_vec = np.linspace(1, ETdata['F11'].iloc[-1], 30)
    lamPS_vec = np.linspace(1, PSdata['F11'].iloc[-1], 30)

    model = torch_cann_model
    P11_NN_UT_p = Numpy_P11_UT(lamUT_vec, model, normalization)
    P11_NN_ET_p = Numpy_P11_ET(lamET_vec, model, normalization)
    P11_NN_PS_p = Numpy_P11_PS(lamPS_vec, model, normalization)

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

    fig, ax = plotstresses([UTdata['F11'], ETdata['F11'], PSdata['F11']],
                           [UTdata['P11'], ETdata['P11'], PSdata['P11']],
                           [lamUT_vec, lamET_vec, lamPS_vec],
                           [P11_NN_UT_p, P11_NN_ET_p, P11_NN_PS_p])
    window = fig.canvas.manager.window
    window.setWindowTitle("trained on all data: UT ET PS")
    plt.show()


def train_icnn_model():
    device = torch.device("cpu")
    # read the data
    UTdata = pd.read_csv('../Data/UT20.csv')
    ETdata = pd.read_csv('../Data/ET20.csv')
    PSdata = pd.read_csv('../Data/PS20.csv')

    # stack into single array
    P11_data = np.hstack(
        [UTdata['P11'].to_numpy(), ETdata['P11'].to_numpy(), PSdata['P11'].to_numpy()])
    F11_data = np.hstack(
        [UTdata['F11'].to_numpy(), ETdata['F11'].to_numpy(), PSdata['F11'].to_numpy()])
    # indices for the three data sets
    indET = len(UTdata['P11'])
    indPS = indET + len(ETdata['P11'])

    lamUT = F11_data[0:indET]
    lamET = F11_data[indET:indPS]
    lamPS = F11_data[indPS:]

    tensor_lamUT = torch.from_numpy(lamUT).to(
        torch.float32).reshape((-1, 1)).requires_grad_().to(device)
    tensor_lamET = torch.from_numpy(lamET).to(
        torch.float32).reshape((-1, 1)).requires_grad_().to(device)
    tensor_lamPS = torch.from_numpy(lamPS).to(
        torch.float32).reshape((-1, 1)).requires_grad_().to(device)

    P11UT_exp = torch.from_numpy(P11_data[0:indET]).to(
        torch.float).reshape((-1, 1)).to(device)
    P11ET_exp = torch.from_numpy(P11_data[indET:indPS]).to(
        torch.float).reshape((-1, 1)).to(device)
    P11PS_exp = torch.from_numpy(P11_data[indPS:]).to(
        torch.float).reshape((-1, 1)).to(device)

    key = random.PRNGKey(0)
    torch_icnn_model = TorchICNN_model(
        layers=[1, 3, 4, 1], normalization=normalization, key=key)
    # print(torch_icnn_model.psi_i1.get_parameters()[1])
    torch_icnn_model.psi_i1.to(device)
    torch_icnn_model.psi_i2.to(device)
    parameters = list(torch_icnn_model.psi_i1.parameters()) + \
        list(torch_icnn_model.psi_i2.parameters())
    optimizer = torch.optim.Adam(parameters, lr=2e-4)

    mse_loss = torch.nn.MSELoss()

    torch_icnn_model.psi_i1.train()
    torch_icnn_model.psi_i2.train()

    epochs = 15000
    for i in range(epochs):
        optimizer.zero_grad()

        P11UT_pr = Torch_P11_UT(tensor_lamUT, torch_icnn_model, normalization)
        P11ET_pr = Torch_P11_ET(tensor_lamET, torch_icnn_model, normalization)
        P11PS_pr = Torch_P11_PS(tensor_lamPS, torch_icnn_model, normalization)

        loss = mse_loss(P11UT_pr, P11UT_exp) + mse_loss(P11ET_pr,
                                                        P11ET_exp) + mse_loss(P11PS_pr, P11PS_exp)
        loss.backward()
        if (i + 1) % 1000 == 0:
            print(i+1, loss.item())
        optimizer.step()

    # predict
    lamUT_vec = np.linspace(1, UTdata['F11'].iloc[-1], 30)
    lamET_vec = np.linspace(1, ETdata['F11'].iloc[-1], 30)
    lamPS_vec = np.linspace(1, PSdata['F11'].iloc[-1], 30)

    model = torch_icnn_model
    P11_NN_UT_p = Numpy_P11_UT(lamUT_vec, model, normalization)
    P11_NN_ET_p = Numpy_P11_ET(lamET_vec, model, normalization)
    P11_NN_PS_p = Numpy_P11_PS(lamPS_vec, model, normalization)

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

    fig, ax = plotstresses([UTdata['F11'], ETdata['F11'], PSdata['F11']],
                           [UTdata['P11'], ETdata['P11'], PSdata['P11']],
                           [lamUT_vec, lamET_vec, lamPS_vec],
                           [P11_NN_UT_p, P11_NN_ET_p, P11_NN_PS_p])
    window = fig.canvas.manager.window
    window.setWindowTitle("trained on all data: UT ET PS")
    plt.show()


if __name__ == "__main__":
    # validate_model()
    # train_cann_model()
    train_icnn_model()

    # theta = jnp.array([jnp.pi/2, 0])
    # theta_a = theta[0]
    # a0 = jnp.array([jnp.sin(theta_a), jnp.cos(theta_a), 0])
    # theta_s = theta[1]
    # s0 = jnp.array([jnp.sin(theta_s), jnp.cos(theta_s), 0])
    # kron = jnp.kron(a0, s0)
    # print(a0)
    # print(s0)
    # print(kron)

    # F = jnp.array(np.random.normal(size=(3, 3)))
    # def I1(F):
    #
    #     C = jnp.matmul(F, F.transpose())
    #     return jnp.trace(C)
    #
    # print(jnp.dot(F.flatten(), F.flatten()))
    # print(grad(I1)(F)- 2*F)
    #
    # k = jnp.kron(F,F)
    # print(k.shape)
