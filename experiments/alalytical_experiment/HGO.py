import sys

import torch

sys.path.append("../../src")

import matplotlib.pyplot as plt
import sys
import argparse
from io import StringIO

from model import PsiModel, BioConstitutiveModel
from utils import query_device, TrainTimer, ReduceTensorArray
from torch_icnn_cann import FICNN, PICNNWapper
from analytical_strain_energy import *
from continuum_mechanics import GradUT, GradET, GradPS, IncompressibleMaterialStress

from loguru import logger


class RubberPsiModel(PsiModel):
    def __init__(self, num_material_params, num_dir):
        super(RubberPsiModel, self).__init__(num_material_params, num_dir)
        self.PsiIiso = PICNNWapper(x_dims=[3, 8, 12, 12, 1])
        self.PsiIaniso = PICNNWapper(x_dims=[2 * num_dir + 2 * num_dir ** 2, 8, 12, 12, 1])
        self.PsiCom = PICNNWapper(x_dims=[3 + 2 * num_dir + 2 * num_dir ** 2, 8, 12, 12, 1])

        self.Psi = FICNN(layers=[3, 5, 5, 1])

    def forward(self, fibers_invariants, material_params=None):
        psi_iso = self.PsiIiso(fibers_invariants[:, 0:3])
        psi_aniso = self.PsiIaniso(fibers_invariants[:, 3:])
        psi_com = self.PsiCom(fibers_invariants)

        return self.Psi(torch.cat((psi_iso, psi_aniso, psi_com), dim=1))


def train_dataset():
    with open('../../Data/OrthotropicHGO/out1.txt', 'r') as file:
        contents = file.read()
    contents = contents.replace('{', '').replace('}', '').replace(',', ' ')
    contents = np.genfromtxt(StringIO(contents)).reshape((-1, 3, 3))
    F_UTx = torch.from_numpy(contents)
    F_UTx = F_UTx.to(torch.float64).requires_grad_(True)

    with open('../../Data/OrthotropicHGO/out2.txt', 'r') as file:
        contents = file.read()
    contents = contents.replace('{', '').replace('}', '').replace(',', ' ')
    contents = np.genfromtxt(StringIO(contents)).reshape((-1, 3, 3))
    F_UTy = torch.from_numpy(contents)
    F_UTy = F_UTy.to(torch.float64).requires_grad_(True)

    F_UT = torch.cat((F_UTx, F_UTy), dim=0)

    lam_et = np.linspace(0.8, 1.2, 200)
    F_ET = GradET(lam_et)
    F_ET = torch.from_numpy(F_ET).reshape((-1, 3, 3)).to(torch.float64).requires_grad_(True)

    psi_fn = HGO()
    _, P_UT, _, _ = IncompressibleMaterialStress(psi_fn(F_UT), F_UT)  # gradients(psi_fn(F_UT), F_UT)
    _, P_ET, _, _ = IncompressibleMaterialStress(psi_fn(F_ET), F_ET)

    num_shear_sample = 250
    r = torch.linspace(-0.2, 0.2, num_shear_sample)
    F_PS = torch.eye(3).unsqueeze(0).repeat((num_shear_sample, 1, 1)).to(torch.float64)
    F_PS[:, 0, 1] += r  # torch.rand(size=(num_shear_sample,))*0.5
    # F_PS[:, 1, 0] += torch.rand(size=(num_shear_sample,))*0.5
    F_PS.requires_grad_(True)
    _, P_PS, _, _ = IncompressibleMaterialStress(psi_fn(F_PS), F_PS)

    Fs = torch.cat((F_UT, F_ET, F_PS), dim=0)
    Ps = torch.cat((P_UT, P_ET, P_PS), dim=0)
    return Fs.detach().to(torch.float32), Ps.detach().to(torch.float32)

    # F, P = AnalyticalDataGenerator(sampler=SampleF, N_samples=200, energy_fn=SchröderNeffEbbing())

    # energy_fn = SchröderNeffEbbing()
    # F = SampleF(n_samples, l_bound=-0.5, u_bound=1.0)
    # F = torch.from_numpy(F).requires_grad_().to(torch.float64)
    # psi = energy_fn(F)
    # P = PK1_P(psi, F)
    # return F.detach().to(torch.float64), P.detach().to(torch.float64)


def test_dataset():
    num_biax_sample = 100
    lam = torch.linspace(-1, 1, num_biax_sample)
    lam1 = torch.ones((1, num_biax_sample)) + 0.2 * lam
    lam2 = torch.ones(1, num_biax_sample) + 0.1 * lam
    lam3 = 1.0 / (lam1 * lam2)
    F_B = torch.zeros((num_biax_sample, 3, 3))
    F_B[:, 0, 0] = lam1
    F_B[:, 1, 1] = lam2
    F_B[:, 2, 2] = lam3

    F_B = F_B.reshape((-1, 3, 3)).to(torch.float32).requires_grad_(True).to(torch.float32)
    psi_fn = HGO()
    _, P_B, _, _ = IncompressibleMaterialStress(psi_fn(F_B), F_B)
    # print(P_B)
    # print(P_B.shape)

    num_mixed_sample = 100
    lam = torch.linspace(-1, 1, num_mixed_sample)
    F_M = torch.eye(3).unsqueeze(0).repeat((num_mixed_sample, 1, 1)).to(torch.float32)
    F_M[:, 0, 0] += 0.2 * lam
    F_M[:, 1, 1] += 0.1 * lam
    F_M[:, 2, 2] = 1. / (F_M[:, 0, 0] * F_M[:, 1, 1])
    F_M[:, 0, 1] += 0.2 * lam
    F_M.requires_grad_(True)
    _, P_M, _, _ = IncompressibleMaterialStress(psi_fn(F_M), F_M)

    # print(torch.max(P_B[:, 1, 1]) / 100.)
    # print(torch.min(P_B[:, 1, 1]) / 100.)
    # Fs = torch.cat((F_B, F_M), dim=0)
    # Ps = torch.cat((P_B, P_M), dim=0)
    return F_B.detach(), P_B.detach(), F_M.detach(), P_M.detach()


def train(epochs):
    train_timer = TrainTimer(total_step=epochs, step_size=1000)
    #logger.add("../logs/HGO_{time}.log")
    #logger.formatter = ("{asctime} - {level} - {message}")
    device = query_device()

    F, P = train_dataset()
    F = F.detach().requires_grad_().to(device)
    P = P.detach().to(device)

    model = BioConstitutiveModel(num_material_params=1, num_dir=2, incomp=True)
    model.PsiSubNet = RubberPsiModel(model.num_material_params, model.num_dir)
    model.need_material_elasticity_tensor = False
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mse_loss = torch.nn.MSELoss()

    for i in range(epochs):
        optimizer.zero_grad()

        model.free_strss_ax = 2
        W_pred, P11_pr, P_pred, S_pred, sigma_pred, _ = model(F)

        loss = mse_loss(P_pred, P)
        loss.backward()

        optimizer.step()

        if (i + 1) % 1000 == 0:
            h, m, s = train_timer.elapsed_time(i)
            pred_dir = model.pred_dir[0]
            logger.info("  it: {} | loss: {} | rest time {} h {} m", i + 1, loss.detach().cpu().numpy(), int(h), int(m))

            # model.train(False)
            # model.eval()
            # predict(model)
            # model.train(True)

    torch.save(model, '../../outputs/HGO/HGO.pth')


def predict(model=None):
    if model is None:
        model = torch.load('../../outputs/HGO/HGO3.pth', map_location=torch.device('cpu'))
    device = query_device("cpu")
    model.to(device)
    model.train(False)
    model.eval()

    F_biaxial, P_biaxial, F_mixed, P_mixed = test_dataset()

    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    model.free_strss_ax = 2

    color = {}
    color["00"] = (228 / 255, 181 / 255, 46 / 255)
    color["11"] = (238 / 255, 149 / 255, 39 / 255)
    color["22"] = (172 / 255, 61 / 255, 54 / 255)
    color["01"] = (45 / 255, 59 / 255, 107 / 255)
    color["10"] = (86 / 255, 122 / 255, 185 / 255)
    color["02"] = (217 / 255, 217 / 255, 217 / 255)
    ps = 13

    # ----------------------  z tensile ----------------------------------
    model.free_strss_ax = 0
    with open('../../Data/OrthotropicHGO/out3.txt', 'r') as file:
        contents = file.read()
    contents = contents.replace('{', '').replace('}', '').replace(',', ' ')
    contents = np.genfromtxt(StringIO(contents)).reshape((-1, 3, 3))
    F = torch.from_numpy(contents).to(torch.float32).requires_grad_(True)
    psi_fn = HGO()
    P11, P_uniaxial, S, sigma = IncompressibleMaterialStress(psi_fn(F), F, free_stress_ax=0)
    P_uniaxial = P_uniaxial.detach().cpu()
    F11_uniaxial = F[:, 2, 2].detach().cpu()

    # F_z = F.detach()
    W_pred, P11_pr, P_uniaxial_pred, S_pred, sigma_pred, _ = model(F)
    P_uniaxial_pred = P_uniaxial_pred.detach().cpu()
    ax[0, 0].set_xlabel("$F_{33}$")
    ax[0, 0].set_ylabel("Uniaxial - $P_{ij}$ $\;[KPa]$")

    ax[0, 0].scatter(ReduceTensorArray(F11_uniaxial), ReduceTensorArray(P_uniaxial[:, 0, 0]), color=color["00"], s=ps)
    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 0, 0], color=color["00"], label="$P_{11}$")

    ax[0, 0].scatter(ReduceTensorArray(F11_uniaxial), ReduceTensorArray(P_uniaxial[:, 1, 0]), color=color["10"], s=ps, )
    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 1, 0], color=color["10"], label="$P_{21}$")

    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 2, 0], color=color["02"], label="$P_{31}$")

    ax[0, 0].scatter(ReduceTensorArray(F11_uniaxial), ReduceTensorArray(P_uniaxial[:, 0, 1]), color=color["01"], s=ps, )
    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 0, 1], color=color["01"], label="$P_{12}$")

    ax[0, 0].scatter(ReduceTensorArray(F11_uniaxial), ReduceTensorArray(P_uniaxial[:, 1, 1]), color=color["11"], s=ps, )
    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 1, 1], color=color["11"], label="$P_{22}$")

    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 2, 1], color=color["02"], label="$P_{32}$")

    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 0, 2], color=color["02"], label="$P_{13}$")

    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 1, 2], color=color["02"], label="$P_{23}$")

    ax[0, 0].scatter(ReduceTensorArray(F11_uniaxial), ReduceTensorArray(P_uniaxial[:, 2, 2]), color=color["22"], s=ps, )
    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 2, 2], color=color["22"], label="$P_{33}$")

    ax[0, 0].legend(loc='lower right', ncol=3, fontsize='small')

    #--------------------------------------------------------------
    model.free_strss_ax = 2
    num_shear_sample = 250
    r = torch.linspace(-0.2, 0.2, num_shear_sample)
    F_PS = torch.eye(3).unsqueeze(0).repeat((num_shear_sample, 1, 1))
    F_PS[:, 1, 0] += r  
    F_PS.requires_grad_(True)
    _, P_shear, _, _ = IncompressibleMaterialStress(psi_fn(F_PS), F_PS)
    P_shear = P_shear.detach().cpu()

    F_PS.detach().requires_grad_(True)
    W, P11, P_shear_pred, S, sigma, _ = model(F_PS)#IncompressibleMaterialStress(psi_fn(F_PS), F_PS, free_stress_ax=2)
    P_shear_pred = P_shear_pred.detach().cpu()

    F_shear = F_PS.detach()
    F12_shear = F_shear[:, 1, 0].cpu()

    ax[0, 1].set_xlabel("$F_{21}$")
    ax[0, 1].set_ylabel("Shear - $P_{ij}$ $\;[KPa]$")

    ax[0, 1].plot(F12_shear, P_shear_pred[:, 0, 2], color=color["02"])
    ax[0, 1].plot(F12_shear, P_shear_pred[:, 1, 2], color=color["02"])
    ax[0, 1].plot(F12_shear, P_shear_pred[:, 2, 0], color=color["02"])
    ax[0, 1].plot(F12_shear, P_shear_pred[:, 2, 1], color=color["02"])
    ax[0, 1].scatter(
        ReduceTensorArray(F12_shear),
        ReduceTensorArray(P_shear[:, 1, 1]),
        color=color["11"],
        s=ps,
    )
    ax[0, 1].plot(F12_shear, P_shear_pred[:, 1, 1], color=color["11"])
    ax[0, 1].scatter(
        ReduceTensorArray(F12_shear),
        ReduceTensorArray(P_shear[:, 2, 2]),
        color=color["22"],
        s=ps,
    )
    ax[0, 1].plot(F12_shear, P_shear_pred[:, 2, 2], color=color["22"])
    ax[0, 1].scatter(
        ReduceTensorArray(F12_shear),
        ReduceTensorArray(P_shear[:, 0, 0]),
        color=color["00"],
        s=ps,
    )
    ax[0, 1].plot(F12_shear, P_shear_pred[:, 0, 0], color=color["00"])
    ax[0, 1].scatter(
        ReduceTensorArray(F12_shear),
        ReduceTensorArray(P_shear[:, 0, 1]),
        color=color["01"],
        s=ps,
    )
    ax[0, 1].plot(F12_shear, P_shear_pred[:, 0, 1], color=color["01"])
    ax[0, 1].scatter(
        ReduceTensorArray(F12_shear),
        ReduceTensorArray(P_shear[:, 1, 0]),
        color=color["10"],
        s=ps,
    )
    ax[0, 1].plot(F12_shear, P_shear_pred[:, 1, 0], color=color["10"])


    # --------------------------------------------------------
    model.free_strss_ax = 2
    W_pred, P11_pr, P_biaxial_pred, S_pred, sigma_pred,_ = model(F_biaxial.requires_grad_())
    P_biaxial_pred = P_biaxial_pred.detach()

    F_biaxial = F_biaxial.detach()
    F11_biaxial = F_biaxial[:, 0, 0]

    # ax[1, 0].set_xlabel('F11')
    # ax[1, 0].set_ylabel('biaxial test - P')
    #
    # ax[1, 0].plot(F11_biaxial, P_biaxial[:, 0, 0], linestyle='-.', color='y')
    # ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 0, 0], color='y')
    #
    # ax[1, 0].plot(F11_biaxial, P_biaxial[:, 1, 1], linestyle='-.', color='c')
    # ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 1, 1], color='c')
    #
    # ax[1, 0].plot(F11_biaxial, P_biaxial[:, 2, 2], linestyle='-.', color='r')
    # ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 2, 2], color='r')

    ax[1, 0].set_xlabel("$F_{11}$")
    ax[1, 0].set_ylabel("biaxial test - $P_{ij}$ $\;[KPa]$")

    ax[1, 0].scatter(ReduceTensorArray(F11_biaxial), ReduceTensorArray(P_biaxial[:, 0, 2]), s=ps, color=color["02"], )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 0, 2], color=color["02"])
    ax[1, 0].scatter(ReduceTensorArray(F11_biaxial), ReduceTensorArray(P_biaxial[:, 1, 2]), s=ps, color=color["02"], )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 1, 2], color=color["02"])
    ax[1, 0].scatter(ReduceTensorArray(F11_biaxial), ReduceTensorArray(P_biaxial[:, 2, 0]), s=ps, color=color["02"], )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 2, 0], color=color["02"])
    ax[1, 0].scatter(ReduceTensorArray(F11_biaxial), ReduceTensorArray(P_biaxial[:, 2, 1]), s=ps, color=color["02"], )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 2, 1], color=color["02"])

    ax[1, 0].scatter(ReduceTensorArray(F11_biaxial),ReduceTensorArray(P_biaxial[:, 0, 0]),s=ps,color=color["00"],)
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 0, 0], color=color["00"])
    ax[1, 0].scatter(ReduceTensorArray(F11_biaxial),ReduceTensorArray(P_biaxial[:, 1, 1]), s=ps, color=color["11"], )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 1, 1], color=color["11"])
    ax[1, 0].scatter( ReduceTensorArray(F11_biaxial), ReduceTensorArray(P_biaxial[:, 0, 1]), s=ps,color=color["01"], )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 0, 1], color=color["01"])
    ax[1, 0].scatter( ReduceTensorArray(F11_biaxial), ReduceTensorArray(P_biaxial[:, 1, 0]), s=ps,color=color["10"],)
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 1, 0], color=color["10"])
    ax[1, 0].scatter( ReduceTensorArray(F11_biaxial), ReduceTensorArray(P_biaxial[:, 2, 2]),s=ps,color=color["22"],)
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 2, 2], color=color["22"])



    # --------------------------------------------------------
    W_pred, P11_pr, P_mixed_pred, S_pred, sigma_pred,_ = model(F_mixed.requires_grad_())
    P_mixed_pred = P_mixed_pred.detach()

    F_mixed = F_mixed.detach()
    F11_mixed = F_mixed[:, 0, 0]

    ax[1, 1].set_xlabel("$F_{11}$")
    ax[1, 1].set_ylabel("mixed test - $P_{ij}$ $\;[KPa]$")

    ax[1, 1].scatter(ReduceTensorArray(F11_mixed), ReduceTensorArray(P_mixed[:, 0, 2]), color=color["02"], s=ps, )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 0, 2], color=color["02"])
    ax[1, 1].scatter(ReduceTensorArray(F11_mixed), ReduceTensorArray(P_mixed[:, 1, 2]), color=color["02"], s=ps, )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 1, 2], color=color["02"])
    ax[1, 1].scatter(ReduceTensorArray(F11_mixed), ReduceTensorArray(P_mixed[:, 2, 0]), color=color["02"], s=ps, )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 2, 0], color=color["02"])
    ax[1, 1].scatter(ReduceTensorArray(F11_mixed), ReduceTensorArray(P_mixed[:, 2, 1]), color=color["02"], s=ps, )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 2, 1], color=color["02"])

    ax[1, 1].scatter(ReduceTensorArray(F11_mixed),ReduceTensorArray(P_mixed[:, 0, 0]),color=color["00"],s=ps,)
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 0, 0], color=color["00"])
    ax[1, 1].scatter(ReduceTensorArray(F11_mixed),ReduceTensorArray(P_mixed[:, 1, 1]),color=color["11"],s=ps,)
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 1, 1], color=color["11"])
    ax[1, 1].scatter(ReduceTensorArray(F11_mixed),ReduceTensorArray(P_mixed[:, 2, 2]),color=color["22"],s=ps,)
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 2, 2], color=color["22"])

    ax[1, 1].scatter(ReduceTensorArray(F11_mixed),ReduceTensorArray(P_mixed[:, 0, 1]),color=color["01"],s=ps,)
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 0, 1], color=color["01"])
    ax[1, 1].scatter(ReduceTensorArray(F11_mixed),ReduceTensorArray(P_mixed[:, 1, 0]),color=color["10"],s=ps,)
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 1, 0], color=color["10"])





    plt.tight_layout()
    plt.savefig('../../outputs/HGO/HGO3.eps', dpi=600)
    plt.show()
    # plt.close()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Analytical model experiment.')
    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('-e', '--epochs', type=int, default=250000)
    # parser.add_argument('-s', '--samples', type=int, default=2000)
    #
    # parser.add_argument('--test', action='store_true')
    #
    # args = parser.parse_args()
    #
    # if args.train:
    #     train(epochs=args.epochs, n_samples=args.samples)
    #     print('training finished')
    #     predict()
    # elif args.test:
    #     predict()
    #     print('testing finished')
    # else:
    #     print('Nothing to be done.')

    # train(epochs=700000)
    predict()

