import sys
sys.path.append("../../src")

from loguru import logger
from analytical_strain_energy import *
from torch_icnn_cann import FICNN, PICNNWapper
from utils import query_device, TrainTimer, gradients, ReduceArray, ReduceTensorArray
from model import PsiModel, BioConstitutiveModel
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from src.continuum_mechanics import GradUT, GradET, GradPS

from lr_scheduler import CustomLRScheduler


class RubberPsiModel(PsiModel):
    def __init__(self, num_material_params, num_dir):
        super(RubberPsiModel, self).__init__(num_material_params, num_dir)
        self.PsiIiso = PICNNWapper(x_dims=[3, 8, 12, 12, 1])
        self.PsiIaniso = PICNNWapper(
            x_dims=[2 * num_dir + 2 * num_dir**2, 8, 12, 12, 1]
        )
        self.PsiCom = PICNNWapper(
            x_dims=[3 + 2 * num_dir + 2 * num_dir**2, 8, 12, 12, 1]
        )

        self.Psi = FICNN(layers=[3, 5, 5, 1])

    def forward(self, fibers_invariants, material_params=None):
        psi_iso = self.PsiIiso(fibers_invariants[:, 0:3])
        psi_aniso = self.PsiIaniso(fibers_invariants[:, 3:])
        psi_com = self.PsiCom(fibers_invariants)

        return self.Psi(torch.cat((psi_iso, psi_aniso, psi_com), dim=1))


def train_dataset(n_samples):
    # data = np.loadtxt("../../Data/Schröder Neff and Ebbing/transversely_isotropic_data/uniaxial_tension.txt")
    # F_UT = torch.from_numpy(data).reshape((-1, 3, 3)).to(torch.float32).requires_grad_(True)
    # data = np.loadtxt( "../../Data/Schröder Neff and Ebbing/transversely_isotropic_data/equibiaxial_tension.txt")
    # F_ET = torch.from_numpy(data).reshape((-1, 3, 3)).to(torch.float32).requires_grad_(True)

    lam = np.linspace(0.5, 2.0, 200)
    F_UT = torch.from_numpy(GradUT(lam)).to(torch.float32).requires_grad_(True)
    F_ET = torch.from_numpy(GradET(lam)).to(torch.float32).requires_grad_(True)


    psi_fn = SchröderNeffEbbing()
    P_UT = gradients(psi_fn(F_UT), F_UT)
    P_ET = gradients(psi_fn(F_ET), F_ET)

    num_shear_sample = 600
    F_PS = torch.eye(3).unsqueeze(0).repeat(
        (num_shear_sample, 1, 1))
    r = torch.linspace(-0.5, 0.5, num_shear_sample)
    F_PS[:, 0, 1] = r  # torch.rand(size=(num_shear_sample,))*0.5
    F_PS[:, 1, 0] = 0.5*r  # torch.rand(size=(num_shear_sample,))*0.5

    # F_PS[:, 0, 0] = 1.0 + r
    # F_PS[:, 1, 1] = 1.0 + 0.5*r
    # F_PS[:, 2, 2] = 1.0 - 0.5*r

    F_PS.requires_grad_(True)
    P_PS = gradients(psi_fn(F_PS), F_PS)

    Fs = torch.cat((F_UT, F_ET, F_PS), dim=0)
    Ps = torch.cat((P_UT, P_ET, P_PS), dim=0)
    return Fs.detach(), Ps.detach()

    # F, P = AnalyticalDataGenerator(sampler=SampleF, N_samples=200, energy_fn=SchröderNeffEbbing())
    # energy_fn = SchröderNeffEbbing()
    # F = SampleF(n_samples, l_bound=-0.5, u_bound=1.0)
    # F = torch.from_numpy(F).requires_grad_().to(torch.float64)
    # psi = energy_fn(F)
    # P = PK1_P(psi, F)
    # return F.detach().to(torch.float64), P.detach().to(torch.float64)

def train_dataset_random(n_samples):
    # def sampler1(n_samples):
    #     return SampleF2(N=n_samples, l_bound = -0.5, u_bound = -0.000000001)
    # def sampler2(n_samples):
    #     return SampleF2(N=n_samples, l_bound = 0.000000001, u_bound = 0.5)
    #
    # F1, P1 = AnalyticalDataGenerator(
    #     sampler=sampler1, N_samples=n_samples, energy_fn=SchröderNeffEbbing()
    # )
    # F2, P2 = AnalyticalDataGenerator(
    #     sampler=sampler2, N_samples=n_samples, energy_fn=SchröderNeffEbbing()
    # )
    #
    # return torch.cat((F1, F2), dim=0), torch.cat((P1,P2), dim=0)

    # def sampler1(n_samples):
    #     return SampleF(N=n_samples, l_bound = -0.5, u_bound = 0.5)
    # F, P = AnalyticalDataGenerator(sampler=sampler1, N_samples=n_samples, energy_fn=SchröderNeffEbbing())
    F = SampleF(N=n_samples, l_bound=-0.5, u_bound=0.5)
    F = torch.from_numpy(F).requires_grad_().to(torch.float32)
    energy_fn = SchröderNeffEbbing()
    psi = energy_fn(F)
    P = PK1_P(psi, F)
    return F.detach(), P.detach()

def test_dataset():
    psi_fn = SchröderNeffEbbing()
    data_uf = np.loadtxt(
        "../../Data/Schröder Neff and Ebbing/transversely_isotropic_data/uniaxial_tension.txt"
    )
    U_F = (
        torch.from_numpy(data_uf).to(torch.float32)
        .reshape((-1, 3, 3))
        .requires_grad_(True)
    )
    U_P = gradients(psi_fn(U_F), U_F)

    data_bf = np.loadtxt(
        "../../Data/Schröder Neff and Ebbing/transversely_isotropic_data/biaxial_tension.txt"
    )
    B_F = (
        torch.from_numpy(data_bf).to(torch.float32)
        .reshape((-1, 3, 3))
        .requires_grad_(True)
    )
    psi_fn = SchröderNeffEbbing()
    B_P = gradients(psi_fn(B_F), B_F)
    # print(P_B)
    # print(P_B.shape)

    num_mixed_sample = 100
    lam = torch.linspace(-1, 2.5, num_mixed_sample)
    M_F = torch.eye(3).unsqueeze(0).repeat(
        (num_mixed_sample, 1, 1))
    M_F[:, 0, 0] = 1.0 + 0.2 * lam
    M_F[:, 1, 1] = 1.0 + 0.1 * lam
    M_F[:, 2, 2] = 1.0 - 0.1 * lam
    M_F[:, 0, 1] = 0.2 * lam
    M_F.requires_grad_(True)
    M_P = gradients(psi_fn(M_F), M_F)

    num_shear_sample = 250
    r = torch.linspace(0, 0.5, num_shear_sample)
    F_PS = torch.eye(3).unsqueeze(0).repeat(
        (num_shear_sample, 1, 1))
    F_PS[:, 0, 1] = r  # torch.rand(size=(num_shear_sample,))*0.5
    F_PS[:, 1, 0] = r  # torch.rand(size=(num_shear_sample,))*0.5
    F_PS.requires_grad_(True)
    P_PS = gradients(psi_fn(F_PS), F_PS)

    return (
        B_F.detach(),
        B_P.detach(),
        M_F.detach(),
        M_P.detach(),
        U_F.detach(),
        U_P.detach(),
        F_PS.detach(),
        P_PS.detach(),
    )


def train(epochs, n_samples, output_name = "Schroder"):
    train_timer = TrainTimer(total_step=epochs, step_size=1000)
    # logger.add("../logs/SchröderNeffEbbing_{time}.log")
    device = query_device()

    F, P = train_dataset(n_samples)
    #F, P = train_dataset_random(n_samples)

    F = F.detach().requires_grad_().to(device)
    P = P.detach().to(device)

    model = BioConstitutiveModel(
        num_material_params=1, num_dir=2, incomp=False)
    model.PsiSubNet = RubberPsiModel(model.num_material_params, model.num_dir)
    model.need_material_elasticity_tensor = False

    # model = torch.load("outputs/schroder/Schroder.pth", map_location=torch.device("cpu"))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 2000, eta_min=0, last_epoch=-1)
    # scheduler = CustomLRScheduler(optimizer=optimizer, warmup_lr_start=1e-5, warmup_total_iters = 10000,total_epochs = epochs, T=int(epochs/2000),  no_aug_iter = 20000, min_lr = 8e-4)
    mse_loss = torch.nn.MSELoss()

    for i in range(epochs):
        model.train(True)
        optimizer.zero_grad()

        W_pred, P11_pr, P_pred, S_pred, sigma_pred, _ = model(F)

        # loss00 = mse_loss(P_pred[:, 0, 0], P[:, 0, 0])
        # loss11 = mse_loss(P_pred[:, 1, 1], P[:, 1, 1])
        # loss22 = mse_loss(P_pred[:, 2, 2], P[:, 2, 2])
        # loss01 = mse_loss(P_pred[:, 0, 1], P[:, 0, 1])
        # loss10 = mse_loss(P_pred[:, 1, 0], P[:, 1, 0])
        # loss02 = mse_loss(P_pred[:, 0, 2], P[:, 0, 2])
        # loss12 = mse_loss(P_pred[:, 1, 2], P[:, 1, 2])
        # loss20 = mse_loss(P_pred[:, 2, 0], P[:, 2, 0])
        # loss21 = mse_loss(P_pred[:, 2, 1], P[:, 2, 1])
        # loss = (
        #     loss00
        #     + loss11
        #     + loss22
        #     + loss01*10
        #     + loss10*10
        #     + loss02
        #     + loss12
        #     + loss20
        #     + loss21
        # )
        # loss = loss / 9.0

        loss01 = mse_loss(P_pred[:, 0, 1].reshape((-1, 1)) * 10, P[:, 0, 1].reshape((-1, 1)) * 10)
        loss10 = mse_loss(P_pred[:, 1, 0].reshape((-1, 1)) * 10, P[:, 1, 0].reshape((-1, 1)) * 10)
        loss22 = mse_loss(P_pred[:, 2, 2].reshape((-1, 1)) * 10, P[:, 2, 2].reshape((-1, 1)) * 10)
        loss = mse_loss(P_pred.reshape((-1, 9)), P.reshape((-1, 9))) + loss01 + loss10 + loss22

        # loss = mse_loss(P_pred[:, 0, 0], P[:, 0, 0])
        loss.backward()

        optimizer.step()
        # scheduler.step()

        if (i + 1) % 1000 == 0:
            h, m, s = train_timer.elapsed_time(i)
            pred_dir = model.pred_dir[0]
            logger.info(f"\n  it: {i+1} | Loss: total:{loss}  S12:{loss01} S21:{loss10} | rest time {int(h)} h {int(m)} m")#S11:{loss00} S22:{loss11} S33:{loss22}
            model.train(False)
            model.eval()
            predict(model, modelname=output_name)
            plt.close()
            model.to(device)
            model.train(True)

    torch.save(model, f'../../outputs/schroder/{output_name}.pth')

def predict(model=None, modelname="Schroder"):
    if model is None:
        model = torch.load(
            f"outputs/schroder/{modelname}.pth", map_location=torch.device("cpu")
        )
    # model = torch.load('/Users/refantasy/code/BioConstitutiveNN/outputs/schroder/Schroder.pth', map_location=torch.device('cpu'))

    model.need_material_elasticity_tensor = False
    device = query_device('cpu')
    model.to(device)
    model.train(False)
    model.eval()

    (
        F_biaxial,
        P_biaxial,
        F_mixed,
        P_mixed,
        F_uniaxial,
        P_uniaxial,
        F_shear,
        P_shear,
    ) = test_dataset()
    F_biaxial = F_biaxial.to(device)
    F_mixed = F_mixed.to(device)
    F_uniaxial = F_uniaxial.to(device)
    F_shear = F_shear.to(device)
    P_uniaxial /=100
    P_biaxial /=100
    P_shear /=100
    P_mixed /=100

    W_pred, P11_pr, P_shear_pred, S_pred, sigma_pred, _ = model(F_shear.requires_grad_())
    P_shear_pred = P_shear_pred.detach().cpu()/100

    W_pred, P11_pr, P_uniaxial_pred, S_pred, sigma_pred, _ = model(F_uniaxial.requires_grad_())
    P_uniaxial_pred = P_uniaxial_pred.detach().cpu()/100

    W_pred, P11_pr, P_biaxial_pred, S_pred, sigma_pred, _ = model(F_biaxial.requires_grad_())
    P_biaxial_pred = P_biaxial_pred.detach().cpu()/100

    W_pred, P11_pr, P_mixed_pred, S_pred, sigma_pred, _ = model(F_mixed.requires_grad_())
    P_mixed_pred = P_mixed_pred.detach().cpu()/100

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    color = {}
    color["00"] = (228 / 255, 181 / 255, 46 / 255)
    color["11"] = (238 / 255, 149 / 255, 39 / 255)
    color["22"] = (172 / 255, 61 / 255, 54 / 255)
    color["01"] = (45 / 255, 59 / 255, 107 / 255)
    color["10"] = (86 / 255, 122 / 255, 185 / 255)
    color["02"] = (217 / 255, 217 / 255, 217 / 255)
    ps = 13
    # --------------------------------------------------------
    F_uniaxial = F_uniaxial.detach()
    F11_uniaxial = F_uniaxial[:, 0, 0].cpu()
    ax[0, 0].set_xlabel("$F_{11}$")
    ax[0, 0].set_ylabel("uniaxial - $P_{ij}$")

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

    ax[0, 0].scatter(ReduceTensorArray(F11_uniaxial), ReduceTensorArray(P_uniaxial[:, 2, 2]),color=color["22"], s=ps,)
    ax[0, 0].plot(F11_uniaxial, P_uniaxial_pred[:, 2,2], color=color["22"], label="$P_{33}$")



    ax[0, 0].legend(loc='upper left',  ncol =3, fontsize='small')

    # --------------------------------------------------------
    F_shear = F_shear.detach()
    F12_shear = F_shear[:, 0, 1].cpu()

    ax[0, 1].set_xlabel("$F_{12}$")
    ax[0, 1].set_ylabel("Shear - $P_{ij}$")

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
    F_biaxial = F_biaxial.detach()
    F11_biaxial = F_biaxial[:, 0, 0].cpu()

    ax[1, 0].set_xlabel("$F_{11}$")
    ax[1, 0].set_ylabel("biaxial test - $P_{ij}$")

    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 0, 2], color=color["02"])
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 1, 2], color=color["02"])
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 2, 0], color=color["02"])
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 2, 1], color=color["02"])

    ax[1, 0].scatter(
        ReduceTensorArray(F11_biaxial),
        ReduceTensorArray(P_biaxial[:, 0, 0]),
        s=ps,
        color=color["00"],
    )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 0, 0], color=color["00"])
    ax[1, 0].scatter(
        ReduceTensorArray(F11_biaxial),
        ReduceTensorArray(P_biaxial[:, 1, 1]),
        s=ps,
        color=color["11"],
    )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 1, 1], color=color["11"])
    ax[1, 0].scatter(
        ReduceTensorArray(F11_biaxial),
        ReduceTensorArray(P_biaxial[:, 0, 1]),
        s=ps,
        color=color["01"],
    )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 0, 1], color=color["01"])
    ax[1, 0].scatter(
        ReduceTensorArray(F11_biaxial),
        ReduceTensorArray(P_biaxial[:, 1, 0]),
        s=ps,
        color=color["10"],
    )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 1, 0], color=color["10"])
    ax[1, 0].scatter(
        ReduceTensorArray(F11_biaxial),
        ReduceTensorArray(P_biaxial[:, 2, 2]),
        s=ps,
        color=color["22"],
    )
    ax[1, 0].plot(F11_biaxial, P_biaxial_pred[:, 2, 2], color=color["22"])

    # --------------------------------------------------------
    F_mixed = F_mixed.detach()
    F11_mixed = F_mixed[:, 0, 0].cpu()

    ax[1, 1].set_xlabel("$F_{11}$")
    ax[1, 1].set_ylabel("mixed test - $P_{ij}$")

    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 0, 2], color=color["02"])
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 1, 2], color=color["02"])
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 2, 0], color=color["02"])
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 2, 1], color=color["02"])

    ax[1, 1].scatter(
        ReduceTensorArray(F11_mixed),
        ReduceTensorArray(P_mixed[:, 0, 0]),
        color=color["00"],
        s=ps,
    )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 0, 0], color=color["00"])
    ax[1, 1].scatter(
        ReduceTensorArray(F11_mixed),
        ReduceTensorArray(P_mixed[:, 1, 1]),
        color=color["11"],
        s=ps,
    )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 1, 1], color=color["11"])
    ax[1, 1].scatter(
        ReduceTensorArray(F11_mixed),
        ReduceTensorArray(P_mixed[:, 2, 2]),
        color=color["22"],
        s=ps,
    )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 2, 2], color=color["22"])

    ax[1, 1].scatter(
        ReduceTensorArray(F11_mixed),
        ReduceTensorArray(P_mixed[:, 0, 1]),
        color=color["01"],
        s=ps,
    )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 0, 1], color=color["01"])
    ax[1, 1].scatter(
        ReduceTensorArray(F11_mixed),
        ReduceTensorArray(P_mixed[:, 1, 0]),
        color=color["10"],
        s=ps,
    )
    ax[1, 1].plot(F11_mixed, P_mixed_pred[:, 1, 0], color=color["10"])

    plt.tight_layout()
    #plt.savefig(f"outputs/schroder/{modelname}.eps", dpi=600)
    plt.savefig(f"../../outputs/schroder/{modelname}.eps", dpi=600)
    plt.close()
    # plt.show()

if __name__ == "__main__":
    # torch.set_default_dtype(torch.float64)
    #
    # parser = argparse.ArgumentParser(
    #     description="Analytical model experiment.")
    # parser.add_argument("--train", action="store_true")
    # parser.add_argument("-e", "--epochs", type=int, default=250000)
    # parser.add_argument("-s", "--samples", type=int, default=2000)
    #
    # parser.add_argument("--test", action="store_true")
    #
    # args = parser.parse_args()
    #
    # if args.train:
    #     train(epochs=args.epochs, n_samples=args.samples)
    #     print("training finished")
    #     predict()
    # elif args.test:
    #     predict()
    #     print("testing finished")
    # else:
    #     print("Nothing to be done.")

    train(250000, 2000, "test")
