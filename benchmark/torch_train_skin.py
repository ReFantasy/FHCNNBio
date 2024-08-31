from src.torch_icnn_cann import gradients
from jax import random, config
import matplotlib.pyplot as plt
import torch.nn.functional as F
from  comutils.jax_node_icnn_cann import torch_icnn_psi
# torch.set_printoptions(precision=8)
from src.utils import *
config.update("jax_enable_x64", True)

sys.path.append('..')

key = random.PRNGKey(0)


class TorchIcnnPsiModel(torch.nn.Module):
    def __init__(self, key, layers=[1, 4, 3, 1]):
        super(TorchIcnnPsiModel, self).__init__()

        self.psi_I1 = torch_icnn_psi(layers, key)
        self.psi_I2 = torch_icnn_psi(layers, key)

        self.psi_I1_I4a = torch_icnn_psi(layers, key)
        self.psi_I1_I4s = torch_icnn_psi(layers, key)
        self.psi_I4a_I4s = torch_icnn_psi(layers, key)

        self.alpha1 = torch.nn.Parameter(torch.tensor([0.5]))
        self.alpha2 = torch.nn.Parameter(torch.tensor([0.5]))
        self.alpha3 = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        I1 = x[:, 0].reshape((-1, 1))
        I2 = x[:, 1].reshape((-1, 1))
        I4a = x[:, 3].reshape((-1, 1))
        I4s = x[:, 4].reshape((-1, 1))

        psi_i1 = self.psi_I1(I1)
        psi_i2 = self.psi_I2(I2)

        alpha1 = 0.5 * (F.tanh(self.alpha1) + 1.0)
        psi_1_4a = self.psi_I1_I4a(alpha1 * I1 + (1 - alpha1) * I4a)

        alpha2 = 0.5 * (F.tanh(self.alpha2) + 1.0)
        psi_1_4s = self.psi_I1_I4s(alpha2 * I1 + (1 - alpha2) * I4s)

        alpha3 = 0.5 * (F.tanh(self.alpha3) + 1.0)
        psi_4a_4s = self.psi_I4a_I4s(alpha3 * I4a + (1 - alpha3) * I4s)

        psi = psi_i1 + psi_i2 + psi_1_4a + psi_1_4s + psi_4a_4s

        return psi


def eval_Cauchy(lambx, lamby, model):
    I1 = lambx ** 2 + lamby ** 2 + (1. / (lambx * lamby) ** 2)
    I2 = lambx ** 2 * lamby ** 2 + lambx ** 2 * \
        (1. / (lambx * lamby) ** 2) + lamby ** 2 * (1. / (lambx * lamby) ** 2)
    I4a = lambx ** 2
    I4s = lamby ** 2
    I1 = (I1 - 3).requires_grad_(True)
    I2 = (I2 - 3).requires_grad_(True)
    I4a = (I4a - 1).requires_grad_(True)
    I4s = (I4s - 1).requires_grad_(True)

    I3 = torch.zeros_like(I1).requires_grad_(True)

    # compute energy
    I = torch.cat((I1, I2, I3, I4a, I4s), dim=1)
    psi = model(I)

    # get pressure from sigma_33 = 0
    lambz = 1. / (lambx * lamby)
    dpsidI1 = gradients(psi, I1)
    dpsidI2 = gradients(psi, I2)
    dpsidI4a = gradients(psi, I4a)
    dpsidI4s = gradients(psi, I4s)

    p = dpsidI1 * lambz ** 2 + dpsidI2 * (I1 * lambz ** 2 - lambz ** 4)
    sigx = dpsidI1 * lambx ** 2 + dpsidI2 * \
        (I1 * lambx ** 2 - lambx ** 4) + dpsidI4a * lambx ** 2 - p
    sigy = dpsidI1 * lamby ** 2 + dpsidI2 * \
        (I1 * lamby ** 2 - lamby ** 4) + dpsidI4s * lamby ** 2 - p
    return sigx, sigy


if __name__ == "__main__":
    # load data
    with open('../Data/P12AC1_bsxsy.npy', 'rb') as f:
        lamb, sigma = np.load(f, allow_pickle=True)
    lamb = lamb.astype(np.float64)
    sigma = sigma.astype(np.float64)
    ind_sx = 81
    ind_sy = 182
    lamb_sigma = np.hstack([lamb, sigma])

    lambx = lamb_sigma[:, 0]
    lamby = lamb_sigma[:, 1]
    sigmax = lamb_sigma[:, 2]
    sigmay = lamb_sigma[:, 3]

    device = query_device()
    tensor_lambx = torch.from_numpy(lambx).to(
        torch.float).reshape((-1, 1)).to(device)
    tensor_lamby = torch.from_numpy(lamby).to(
        torch.float).reshape((-1, 1)).to(device)
    tensor_sigmax_exp = torch.from_numpy(sigmax).to(
        torch.float).reshape((-1, 1)).to(device)
    tensor_sigmay_exp = torch.from_numpy(sigmay).to(
        torch.float).reshape((-1, 1)).to(device)

    # create model
    key, subkey = random.split(key)
    model = TorchIcnnPsiModel(key)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # train
    for it in range(100000):
        optimizer.zero_grad()

        sigx_pr, sigy_pr = eval_Cauchy(tensor_lambx, tensor_lamby, model)
        loss = (tensor_sigmax_exp - sigx_pr).square().mean() + \
            (tensor_sigmay_exp - sigy_pr).square().mean()
        loss.backward()
        # print(loss.dtype)

        print_freq = 1000
        if ((it + 1) % print_freq == 0):
            # print(loss.detach().numpy())
            to_print = "it %i, train loss = %e" % (
                it + 1, loss.detach().cpu().numpy())
            print(to_print)

        optimizer.step()

    # predict

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
    sigx, sigy = eval_Cauchy(tensor_lambx, tensor_lamby, model)
    sigx = torch.squeeze(sigx).flatten().cpu().detach().numpy()
    sigy = torch.squeeze(sigy).flatten().cpu().detach().numpy()

    fig, ax = plotstresses([lambx[ind_sx:ind_sy], lambx[0:ind_sx], lamby[ind_sy:]],
                           [sigmax[ind_sx:ind_sy], sigmax[0:ind_sx], sigmax[ind_sy:]],
                           [sigmay[ind_sx:ind_sy], sigmay[0:ind_sx], sigmay[ind_sy:]],
                           [lambx[ind_sx:ind_sy], lambx[0:ind_sx], lamby[ind_sy:]],
                           [sigx[ind_sx:ind_sy], sigx[0:ind_sx], sigx[ind_sy:]],
                           [sigy[ind_sx:ind_sy], sigy[0:ind_sx], sigy[ind_sy:]])
    plt.show()
