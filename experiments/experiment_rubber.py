import sys
sys.path.append('../src')

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil
import argparse


from data_loader import ReadRubberData
from model import BioConstitutiveModel
from continuum_mechanics import *
from utils import query_device
from torch_icnn_cann import FICNN, PICNNWapper





save_dir = '../outputs/rubber'

# class RubberPsiModel(PsiModel):
#     def __init__(self, num_material_params, num_dir):
#         super(RubberPsiModel, self).__init__(num_material_params, num_dir)
#         self.PsiIiso = PICNNWapper(x_dims=[3, 8, 8, 8, 1])
#         self.PsiIaniso = PICNNWapper(x_dims=[2 * num_dir + 2 * num_dir ** 2, 2, 2, 1])
#
#         self.Psi = FICNN(layers=[2, 3, 1])

class RubberPsiModel(torch.nn.Module):
    def __init__(self, num_material_params=0, num_dir=0):
        super(RubberPsiModel, self).__init__()
        self.num_material_params = num_material_params
        self.num_dir = num_dir

        # self.PsiIiso = PICNNWapper(x_dims=[3, 8, 12, 6, 1])
        # self.PsiIaniso = PICNNWapper(x_dims=[2 * num_dir + 2 * num_dir ** 2, 2, 1])
        # self.PsiCom = PICNNWapper(x_dims=[3 + 2 * num_dir + 2 * num_dir ** 2, 2, 1])

        self.PsiIiso = PICNNWapper(x_dims=[3, 8, 8, 8, 1])
        self.PsiIaniso = PICNNWapper(x_dims=[2 * num_dir + 2 * num_dir ** 2, 2, 2, 1])
        self.PsiCom = PICNNWapper(x_dims=[3 + 2 * num_dir + 2 * num_dir ** 2, 1, 1, 1])

        self.Psi = FICNN(layers=[3, 3, 1])

    def forward(self, fibers_invariants, material_params=None):
        psi_iso = self.PsiIiso(fibers_invariants[:, 0:3])
        psi_aniso = self.PsiIaniso(fibers_invariants[:, 3:])

        psi_com = self.PsiCom(fibers_invariants)

        return self.Psi(torch.cat((psi_iso, psi_aniso, psi_com), dim=1))

def test_dataset():
    UTdata = pd.read_csv('../Data/UT20.csv')
    ETdata = pd.read_csv('../Data/ET20.csv')
    PSdata = pd.read_csv('../Data/PS20.csv')

    F11_UT, F11_ET, F11_PS = UTdata['F11'], ETdata['F11'], PSdata['F11']
    P11_UT, P11_ET, P11_PS = UTdata['P11'], ETdata['P11'], PSdata['P11']
    return F11_UT.values, F11_ET.values, F11_PS.values, P11_UT.values, P11_ET.values, P11_PS.values

val_data = test_dataset()

def train(model, F, P11_exp, epochs, batch_size = 45, mode='full'):
    mse_loss = torch.nn.MSELoss()

    for i in range(epochs):
        optimizer.zero_grad()

        #随机索引
        indices = torch.randperm(F.shape[0])
        batch_size = batch_size
        n = len(indices) // batch_size + 1
        split_indices = np.array_split(indices.numpy(), n)

        for batch_indices in split_indices:
            batch_F = F[batch_indices]
            batch_F = batch_F.detach()
            batch_P11_exp = P11_exp[batch_indices]

            W_pred, P11_pr, P_pred, S_pred, sigma_pred = model(batch_F.requires_grad_())
            P11_pr = P11_pr.reshape((-1, 1))

            loss = mse_loss(P11_pr, batch_P11_exp)
            loss.backward()

            # 计算验证损失
            # F11_UT, F11_ET, F11_PS, P11_UT, P11_ET, P11_PS = val_data
            # P11_val_gt = torch.cat((torch.from_numpy(P11_UT).reshape((-1,1)),torch.from_numpy(P11_ET).reshape((-1,1)),torch.from_numpy(P11_PS).reshape((-1,1))), dim=0)
            # F_UT = torch.from_numpy(GradUT(F11_UT)).to(torch.float).requires_grad_()
            # F_ET = torch.from_numpy(GradET(F11_ET)).to(torch.float).requires_grad_()
            # F_PS = torch.from_numpy(GradPS(F11_PS)).to(torch.float).requires_grad_()
            # F_val = torch.cat((F_UT, F_ET, F_PS), dim=0)
            # _, P11_val_pr, _, _, _ = model(F_val)
            # val_loss = mse_loss(P11_val_pr.reshape((-1,1)), P11_val_gt)
            if i == 0:
                min_loss = loss
            elif min_loss > loss:
                min_loss = loss
                # 保存损失最小模型
                path = f'{save_dir}/model_rubber_{mode}_best.pth'  # save_dir +'model_'+mode+'_'+ str(i+1) + '.pth'
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i + 1}
                torch.save(checkpoint, path)

            if (i + 1) % 1000 == 0:
                print(f'{i + 1}, train loss:{loss.item():.6f} min loss:{min_loss.item()}')

                path = f'{save_dir}/model_rubber_{mode}_{i + 1}.pth'  # save_dir +'model_'+mode+'_'+ str(i+1) + '.pth'
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': i + 1}
                torch.save(checkpoint, path)
                shutil.copyfile(path, f'{save_dir}/model_rubber_{mode}_latest.pth')

            optimizer.step()




def predict(model, mode='full', ver = 'latest'):
    # UTdata = pd.read_csv('../Data/UT20.csv')
    # ETdata = pd.read_csv('../Data/ET20.csv')
    # PSdata = pd.read_csv('../Data/PS20.csv')

    F11_UT, F11_ET, F11_PS, P11_UT, P11_ET, P11_PS = test_dataset()

    #model = torch.load("../outputs/rubber_full_model.pth")
    checkpoint = torch.load(f'{save_dir}/model_rubber_{mode}_{ver}.pth')
    model.load_state_dict(checkpoint['model'])
    model.to(torch.device('cpu'))

    num_sample = 100
    lamUT_vec = np.linspace(1, F11_UT[-1], num_sample)
    lamET_vec = np.linspace(1, F11_ET[-1], num_sample)
    lamPS_vec = np.linspace(1, F11_PS[-1], num_sample)

    F_UT = torch.from_numpy(GradUT(lamUT_vec)).to(torch.float).requires_grad_()
    F_ET = torch.from_numpy(GradET(lamET_vec)).to(torch.float).requires_grad_()
    F_PS = torch.from_numpy(GradPS(lamPS_vec)).to(torch.float).requires_grad_()
    F = torch.cat((F_UT, F_ET, F_PS), dim=0)

    W_pred, P11_pr, P_pred, S_pred, sigma_pred = model(F)

    P11_NN_UT_p = P11_pr[0:num_sample]
    P11_NN_ET_p = P11_pr[num_sample:num_sample * 2]
    P11_NN_PS_p = P11_pr[num_sample * 2:]

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

    fig, ax = plotstresses([F11_UT, F11_ET, F11_PS],
                           [P11_UT, P11_ET, P11_PS],
                           [lamUT_vec, lamET_vec, lamPS_vec],
                           [P11_NN_UT_p.detach().numpy(), P11_NN_ET_p.detach().numpy(), P11_NN_PS_p.detach().numpy()])

    plt.tight_layout()
    #plt.savefig(f'../outputs/model_rubber_{mode}_{ver}.eps', dpi=600)
    plt.show()


if __name__ == "__main__":
    device = query_device()

    F_UT, F_ET, F_PS, P11UT_exp, P11ET_exp, P11PS_exp = ReadRubberData(device)

    F = torch.cat((F_UT, F_ET, F_PS), dim=0).to(torch.float)
    P11_exp = torch.cat((P11UT_exp, P11ET_exp, P11PS_exp), dim=0).to(torch.float)

    model = BioConstitutiveModel(num_material_params=1, num_dir=2, incomp=False)
    model.PsiSubNet = RubberPsiModel(model.num_material_params, model.num_dir)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    parser = argparse.ArgumentParser(description='Analytical model experiment.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-e', '--epochs', type=int, default=60000)
    parser.add_argument('-s', '--samples', type=int, default=2000)
    parser.add_argument('-b', '--batchsize', type=int, default=30)
    parser.add_argument('-m', '--mode', type=str, default='full')

    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.train:
        # 在完整数据集上训练
        train(model=model, F=F, P11_exp=P11_exp, epochs=args.epochs, batch_size=args.batchsize, mode=args.mode)
        predict(model, mode=args.mode, ver='latest')
        predict(model, mode=args.mode, ver='best')
        print('training finished')
    elif args.test:
        predict(model, mode=args.mode, ver='latest')
        predict(model, mode=args.mode, ver='best')
        print('testing finished')
    else:
        print('Nothing to be done.')


    # 在完整数据集上训练
    # train(model=model, F=F, P11_exp=P11_exp, epochs=50000, batch_size = 5, mode='full')
    # predict(model, mode='full', ver='latest')
    # predict(model, mode='full', ver='best')


    #train(model, F_UT, P11UT_exp, epochs=100000, mode='ut')
    # net = train(F_ET, P11ET_exp, epochs=30000, device=device)
    # net = train(F_PS, P11PS_exp, epochs=30000, device=device)
    # model.to(torch.device('cpu'))

    #---------------------------------------------------------------------

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

    # UTdata = pd.read_csv('../Data/UT20.csv')
    # ETdata = pd.read_csv('../Data/ET20.csv')
    # PSdata = pd.read_csv('../Data/PS20.csv')

    F11_UT, F11_ET, F11_PS, P11_UT, P11_ET, P11_PS = test_dataset()

    num_sample = 100
    lamUT_vec = np.linspace(1, F11_UT[-1], num_sample)
    lamET_vec = np.linspace(1, F11_ET[-1], num_sample)
    lamPS_vec = np.linspace(1, F11_PS[-1], num_sample)

    # 模型预测
    F_UT = torch.from_numpy(GradUT(lamUT_vec)).to(torch.float).requires_grad_()
    F_ET = torch.from_numpy(GradET(lamET_vec)).to(torch.float).requires_grad_()
    F_PS = torch.from_numpy(GradPS(lamPS_vec)).to(torch.float).requires_grad_()
    F = torch.cat((F_UT, F_ET, F_PS), dim=0)

    model = torch.load('../outputs/rubber/available_model/rubber_best.pth')
    W_pred, P11_pr, P_pred, S_pred, sigma_pred = model(F)

    P11_OURS_UT_p = P11_pr[0:num_sample].detach().numpy()
    P11_OURS_ET_p = P11_pr[num_sample:num_sample * 2].detach().numpy()
    P11_OURS_PS_p = P11_pr[num_sample * 2:].detach().numpy()

    # benchmark ICNN
    P11_ICNN_UT_p = np.load('../outputs/rubber/benchmark_data/rubber_icnn_ut.npy')
    P11_ICNN_ET_p = np.load('../outputs/rubber/benchmark_data/rubber_icnn_et.npy')
    P11_ICNN_PS_p = np.load('../outputs/rubber/benchmark_data/rubber_icnn_ps.npy')

    # benchmark CANNrubber_
    P11_CANN_UT_p = np.load('../outputs/rubber/benchmark_data/rubber_cann_ut.npy')
    P11_CANN_ET_p = np.load('../outputs/rubber/benchmark_data/rubber_cann_et.npy')
    P11_CANN_PS_p = np.load('../outputs/rubber/benchmark_data/rubber_cann_ps.npy')

    # benchmark Noderubber_
    P11_NODE_UT_p = np.load('../outputs/rubber/benchmark_data/rubber_node_ut.npy')
    P11_NODE_ET_p = np.load('../outputs/rubber/benchmark_data/rubber_node_et.npy')
    P11_NODE_PS_p = np.load('../outputs/rubber/benchmark_data/rubber_node_ps.npy')


    # fig, ax = plotstresses([F11_UT, F11_ET, F11_PS],
    #                        [P11_UT, P11_ET, P11_PS],
    #                        [lamUT_vec, lamET_vec, lamPS_vec],
    #                        [P11_NODE_UT_p, P11_NODE_ET_p, P11_NODE_PS_p])

    def plotstresses2(x_gt, y_gt, x_pr, y_prs):
        y_pr_icnn,y_pr_cann,y_pr_node, y_pr_ours = y_prs
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        #labels = ['UT', 'ET', 'PS']
        labels =[r'(a) Uniaxial tension', r'(b) Equibiaxial tension', r'(c) Pure shear']
        for axi, x_gti, y_gti, x_pri, y_pri_icnn,y_pri_cann,y_pri_node,y_pri_ours, label in zip(ax, x_gt, y_gt, x_pr, y_pr_icnn,y_pr_cann,y_pr_node,y_pr_ours, labels):

            axi.plot(x_pri, y_pri_icnn, color='#98a9d0', label='ICNN')
            axi.plot(x_pri, y_pri_cann, color='#75c8ae', label='CANN')
            axi.plot(x_pri, y_pri_node, color='#b0dc66', label='NODE')
            axi.plot(x_pri, y_pri_ours, color='#fc9871', label='FHCNNBio')
            axi.plot(x_gti, y_gti, 'o', ms=3 , label='Train data')

            # # 创建局部放大图
            # axins = axi.inset_axes([0.5, 0.5, 0.2, 0.2])
            # axins.plot(x_gti, y_gti, 'k.')
            # axins.plot(x_pri, y_pri_icnn)
            # axins.plot(x_pri, y_pri_cann)
            # axins.plot(x_pri, y_pri_node)
            # axins.plot(x_pri, y_pri_ours)
            # axins.set_xlim(2, 3)
            # axins.set_ylim(0, 1)
            # axins.set_xlabel('Zoom In')
            # axins.set_title('Inset Plot')
            # # 在主图上标记放大区域
            # axi.indicate_inset_zoom(axins)

            axi.set_title(label)
            axi.set_xlabel(r'Stretch $\lambda [-]$')
            axi.set_ylabel(r'Nominal stress $P_{11} [MPa]$')
            axi.legend()

        return fig, ax


    fig, ax = plotstresses2([F11_UT, F11_ET, F11_PS],
                           [P11_UT, P11_ET, P11_PS],
                           [lamUT_vec, lamET_vec, lamPS_vec],
                           [[P11_ICNN_UT_p, P11_ICNN_ET_p, P11_ICNN_PS_p],
                                 [P11_CANN_UT_p, P11_CANN_ET_p, P11_CANN_PS_p],
                                 [P11_NODE_UT_p, P11_NODE_ET_p, P11_NODE_PS_p],
                                 [P11_OURS_UT_p, P11_OURS_ET_p, P11_OURS_PS_p]]
                           )

    plt.tight_layout()
    plt.savefig(f'../outputs/rubber.eps', dpi=1200)
    plt.show()




