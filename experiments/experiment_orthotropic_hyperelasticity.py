import matplotlib.pyplot as plt
import sys
import argparse
sys.path.append("..")

from src.model import PsiModel, BioConstitutiveModel
from src.utils import query_device, TrainTimer
from src.torch_icnn_cann import FICNN, PICNNWapper
from src.analytical_strain_energy import *

from loguru import logger

class RubberPsiModel(PsiModel):
    def __init__(self, num_material_params, num_dir):
        super(RubberPsiModel, self).__init__(num_material_params, num_dir)
        self.PsiIiso = PICNNWapper(x_dims=[3, 8, 8, 8, 1])
        self.PsiIaniso = PICNNWapper(x_dims=[2 * num_dir + 2 * num_dir ** 2, 8, 8, 8, 1])
        self.PsiCom = PICNNWapper(x_dims=[3 + 2 * num_dir + 2 * num_dir ** 2, 8, 8, 8, 1])

        self.Psi = FICNN(layers=[3, 3, 4, 3, 1])

    def forward(self, fibers_invariants, material_params = None):
        psi_iso = self.PsiIiso(fibers_invariants[:, 0:3])
        psi_aniso = self.PsiIaniso(fibers_invariants[:, 3:])
        psi_com = self.PsiCom(fibers_invariants)

        return self.Psi(torch.cat((psi_iso,psi_aniso, psi_com), dim=1))


def train(epochs, n_samples):
    train_timer = TrainTimer(total_step=epochs, step_size=1000)
    logger.add("../logs/orthotropic_{time}.log")
    device = query_device()

    F, P = AnalyticalDataGenerator(sampler=SampleF, N_samples=n_samples, energy_fn=OrthotropicHyperelasticity())
    F = F.detach().to(torch.float).requires_grad_().to(device)
    P = P.to(torch.float).to(device)

    model = BioConstitutiveModel(num_material_params=1, num_dir=2, incomp=False)
    model.PsiSubNet = RubberPsiModel(model.num_material_params, model.num_dir)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mse_loss = torch.nn.MSELoss()

    for i in range(epochs):
        optimizer.zero_grad()

        W_pred, P11_pr, P_pred, S_pred, sigma_pred = model(F)

        loss = mse_loss(P_pred, P)
        loss.backward()

        if (i + 1) % 1000 == 0:
            #print(i + 1, loss.detach().cpu().numpy())
            h, m, s = train_timer.elapsed_time(i)
            pred_dir = model.pred_dir[0]
            logger.info("  it: {} | loss: {} | rest time {} h {} m", i + 1, loss.detach().cpu().numpy(), int(h), int(m))

            #
            #print(pred_dir.cpu().numpy())
            
        optimizer.step()

    torch.save(model, '../outputs/orthotropic.pth')


def predict():
    model = torch.load('../outputs/orthotropic.pth',map_location=torch.device('cpu'))
    device = query_device()
    model.to(device)

    bonet_hyper = OrthotropicHyperelasticity()

    #生成测试数据
    lam_ut = np.linspace(0.8, 1.2, 100)

    F_UT = torch.from_numpy(GradUT(lam_ut)).to(torch.float).requires_grad_().to(device)
    F_ET = torch.from_numpy(GradET(lam_ut)).to(torch.float).requires_grad_().to(device)
    F_PS = torch.from_numpy(GradPS(lam_ut)).to(torch.float).requires_grad_().to(device)

    psi_ut = bonet_hyper(F_UT)
    P_UT = PK1_P(psi_ut, F_UT).to(device)
    psi_et = bonet_hyper(F_ET)
    P_ET = PK1_P(psi_et, F_ET).to(device)
    psi_ps = bonet_hyper(F_PS)
    P_PS = PK1_P(psi_ps, F_PS).to(device)

    # 预测
    W_pred, P11_pr, P_UT_pred, S_pred, sigma_pred = model(F_UT)
    W_pred, P11_pr, P_ET_pred, S_pred, sigma_pred = model(F_ET)
    W_pred, P11_pr, P_PS_pred, S_pred, sigma_pred = model(F_PS)

    def plotstresses(x_gt, y_gt, x_pr, y_pr):

        fig, ax = plt.subplots(3, 3, figsize=(12, 8))
        labels = [['UT', 'ET', 'PS'],['UT', 'ET', 'PS'],['UT', 'ET', 'PS']]
        Plabels = [11, 22, 33]
        for axi, x_gti,y_gti,x_pri, y_pri, label, Plabel in zip(ax,x_gt,y_gt,x_pr, y_pr, labels, Plabels):

            for axii, x_gtii, y_gtii,x_prii, y_prii,labeli in zip(axi, x_gti,y_gti,x_pri, y_pri, label):
                axii.plot(x_gtii, y_gtii, 'k.')
                axii.plot(x_prii, y_prii)
                axii.set_title(labeli)
                axii.set_xlabel(r'Stretch $\lambda [-]$')

                #axii.set_ylabel(r'Nominal stress $P_{}_{} [MPa]$'.format("3", "3"))
                axii.set_ylabel("Nominal stress $P_{%i} [MPa]$" % (Plabel))
        return fig, ax

    # 绘图
    F_UT = F_UT.cpu().detach().numpy()
    P_UT = P_UT.cpu().detach().numpy()
    P_UT_pred = P_UT_pred.cpu().detach().numpy()

    F_ET = F_ET.cpu().detach().numpy()
    P_ET = P_ET.cpu().detach().numpy()
    P_ET_pred = P_ET_pred.cpu().detach().numpy()

    F_PS = F_PS.cpu().detach().numpy()
    P_PS = P_PS.cpu().detach().numpy()
    P_PS_pred = P_PS_pred.cpu().detach().numpy()


    P_UT_11 = P_UT[:, 0, 0]
    P_UT_22 = P_UT[:, 1, 1]
    P_UT_33 = P_UT[:, 2, 2]
    P_UT_11_pred = P_UT_pred[:, 0, 0]
    P_UT_22_pred = P_UT_pred[:, 1, 1]
    P_UT_33_pred = P_UT_pred[:, 2, 2]

    P_ET_11 = P_ET[:, 0, 0]
    P_ET_22 = P_ET[:, 1, 1]
    P_ET_33 = P_ET[:, 2, 2]
    P_ET_11_pred = P_ET_pred[:, 0, 0]
    P_ET_22_pred = P_ET_pred[:, 1, 1]
    P_ET_33_pred = P_ET_pred[:, 2, 2]

    P_PS_11 = P_PS[:, 0, 0]
    P_PS_22 = P_PS[:, 1, 1]
    P_PS_33 = P_PS[:, 2, 2]
    P_PS_11_pred = P_PS_pred[:, 0, 0]
    P_PS_22_pred = P_PS_pred[:, 1, 1]
    P_PS_33_pred = P_PS_pred[:, 2, 2]

    fig, ax = plotstresses([[lam_ut,lam_ut,lam_ut],                     [lam_ut,lam_ut,lam_ut],                     [lam_ut,lam_ut,lam_ut]],#[[1,2,3],[4,5,6]],#
                           [[P_UT_11, P_ET_11, P_PS_11],                [P_UT_22, P_ET_22, P_PS_22],                [P_UT_33, P_ET_33, P_PS_33]],
                           [[lam_ut,lam_ut,lam_ut],                     [lam_ut,lam_ut,lam_ut],                     [lam_ut,lam_ut,lam_ut]],
                           [[P_UT_11_pred, P_ET_11_pred, P_PS_11_pred], [P_UT_22_pred, P_ET_22_pred, P_PS_22_pred], [P_UT_33_pred, P_ET_33_pred, P_PS_33_pred]])

    plt.tight_layout()
    #plt.savefig('../outputs/orthotropic.eps', dpi=600)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analytical model experiment.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-e','--epochs', type=int, default=250000)
    parser.add_argument('-s','--samples', type=int, default=2000)

    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, n_samples=args.samples)
        print('training finished')
    elif args.test:
        predict()
        print('testing finished')
    else:
        print('Nothing to be done.')
