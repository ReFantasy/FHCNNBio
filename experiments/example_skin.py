import sys
sys.path.append('../src')

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse
import time
import json
from loguru import logger
import numpy as np

from data_loader import ReadSkinData
from model import BioConstitutiveModel
from continuum_mechanics import *
from utils import query_device, split_data_uniform, tensorboard_smoothing


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


def train(model, Fs, sigma_exps, epochs, model_saved_name):
    F_train = Fs[0].detach().requires_grad_(True)
    sigma_exp_train = sigma_exps[0].detach()

    F_test = Fs[1].detach().requires_grad_(True)
    sigma_exp_test = sigma_exps[1].detach()

    writer = SummaryWriter('../logs/' + time.strftime('%m-%d %H:%M:%S', time.localtime()))
    recorded_loss = [[],[]]

    for epoch in range(epochs):  # 700000
        optimizer.zero_grad()

        psi, P11, P, S, sigma_pr = model(F_train)

        loss = (MseLoss(sigma_exp_train[:, 0, 0], sigma_pr[:, 0, 0]) + MseLoss(sigma_exp_train[:, 1, 1], sigma_pr[:, 1, 1])) / 2

        print_freq = 1000
        if ((epoch + 1) % print_freq == 0):
            psi, P11, P, S, sigma_test_pr = model(F_test)
            val_loss = (MseLoss(sigma_exp_test[:, 0, 0], sigma_test_pr[:, 0, 0]) + MseLoss(sigma_exp_test[:, 1, 1], sigma_test_pr[:, 1, 1])) / 2
            to_print = f'epoch {epoch+1}, train loss = {loss.item()} val loss = {val_loss.item()}'
            logger.info(to_print)

        if ((epoch + 1) % 50 == 0):
            psi, P11, P, S, sigma_test_pr = model(F_test)
            val_loss = (MseLoss(sigma_exp_test[:, 0, 0], sigma_test_pr[:, 0, 0]) + MseLoss(sigma_exp_test[:, 1, 1], sigma_test_pr[:, 1, 1])) / 2
            writer.add_scalars(main_tag='loss', tag_scalar_dict={'train': loss, 'val': val_loss}, global_step=epoch)
            recorded_loss[0].append(loss.item())
            recorded_loss[1].append(val_loss.item())

        loss.backward()

        optimizer.step()

    torch.save(model, f'../outputs/{model_saved_name}.pth')
    np.save(f'../outputs/{model_saved_name}_loss.npy', recorded_loss)



def predict(model_name):
    # load data
    with open('../Data/P12AC1_bsxsy.npy', 'rb') as f:
        lamb, sigma = np.load(f, allow_pickle=True)
    lamb = lamb.astype(np.float64)
    sigma = sigma.astype(np.float64)
    ind_sx = 81
    ind_sy = 182

    lambx = lamb[:, 0]
    lamby = lamb[:, 1]
    lambz = 1. / (lambx * lamby)
    sigmax = sigma[:, 0]
    sigmay = sigma[:, 1]

    device = query_device()
    tensor_F_EB, tensor_F_SX, tensor_F_SY, tensor_Sigma_EB, tensor_Sigma_SX, tensor_Sigma_SY = ReadSkinData(device)
    F = torch.cat((tensor_F_EB, tensor_F_SX, tensor_F_SY), dim=0)

    model = torch.load(f'../outputs/{model_name}.pth', map_location=torch.device('cpu'))
    model.to(device)

    psi, P11, P, S, sigma = model(F)
    sigx, sigy = sigma[:, 0, 0].reshape((-1, 1)), sigma[:, 1, 1].reshape((-1, 1))

    sigx = sigx.detach().cpu().numpy()
    sigy = sigy.detach().cpu().numpy()
    fig, ax = plotstresses([lambx[ind_sx:ind_sy], lambx[0:ind_sx], lamby[ind_sy:]],
                           [sigmax[ind_sx:ind_sy], sigmax[0:ind_sx], sigmax[ind_sy:]],
                           [sigmay[ind_sx:ind_sy], sigmay[0:ind_sx], sigmay[ind_sy:]],
                           [lambx[ind_sx:ind_sy], lambx[0:ind_sx], lamby[ind_sy:]],
                           [sigx[ind_sx:ind_sy], sigx[0:ind_sx], sigx[ind_sy:]],
                           [sigy[ind_sx:ind_sy], sigy[0:ind_sx], sigy[ind_sy:]])
    plt.tight_layout()
    #plt.savefig('../outputs/skin.eps', dpi=600)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analytical model experiment.')
    #parser.add_argument('-h','--help', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-t', '--type', type=str, default='full', help='train only on dataset: full, sx, sy or eb')
    parser.add_argument('-n','--name', type=str, default='skin_model', help='name of the model to be saved after trainning finshed')
    parser.add_argument('-e', '--epochs', type=int, default=200000)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()


    device = query_device()

    F_EB, F_SX, F_SY, Sigma_EB, Sigma_SX, Sigma_SY = ReadSkinData(device)
    F = torch.cat((F_EB, F_SX, F_SY), dim=0)
    sigma_exp = torch.cat((Sigma_EB, Sigma_SX, Sigma_SY), dim=0)

    # 划分训练集合 和 验证集
    F_EB_train, Sigma_EB_train, F_EB_test, Sigma_EB_test = split_data_uniform(F_EB, Sigma_EB)
    F_SX_train, Sigma_SX_train, F_SX_test, Sigma_SX_test = split_data_uniform(F_SX, Sigma_SX)
    F_SY_train, Sigma_SY_train, F_SY_test, Sigma_SY_test = split_data_uniform(F_SY, Sigma_SY)

    F_train = torch.cat((F_EB_train, F_SX_train, F_SY_train), dim=0)
    sigma_exp_train = torch.cat((Sigma_EB_train, Sigma_SX_train, Sigma_SY_train), dim=0)

    F_test = torch.cat((F_EB_test, F_SX_test, F_SY_test), dim=0)
    sigma_exp_test = torch.cat((Sigma_EB_test, Sigma_SX_test, Sigma_SY_test), dim=0)

    # 创建模型
    model = BioConstitutiveModel(num_material_params=1, num_dir=2, incomp=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    MseLoss = torch.nn.MSELoss()

    #logger.add("../logs/skin_{time}.log")

    if args.train:
        train(model=model, Fs=[F_train, F_test], sigma_exps=[sigma_exp_train, sigma_exp_test], epochs=args.epochs, model_saved_name=args.name)
        print('training finished')

        # if args.type == 'full':
        #     print('training on full dataset')
        #     train(model=model, F=F, sigma_exp=sigma_exp, epochs=args.epochs, model_saved_name=args.name)
        # if args.type == 'sx':
        #     print('training on sx dataset')
        #     train(model=model, F=F_SX, sigma_exp=Sigma_SX, epochs=args.epochs, model_saved_name=args.name)
        # if args.type == 'sy':
        #     print('training on sy dataset')
        #     train(model=model, F=F_SY, sigma_exp=Sigma_SY, epochs=args.epochs, model_saved_name=args.name)
        # if args.type == 'eb':
        #     print('training on eb dataset')
        #     train(model=model, F=F_EB, sigma_exp=Sigma_EB, epochs=args.epochs, model_saved_name=args.name)
    elif args.test:
        predict(model_name=args.name)
        print('testing finished')
    else:
        print('Nothing to be done.')


    #------------------------------------------------------------------------------------
    #    test curve
    # ------------------------------------------------------------------------------------
    model = torch.load(f'../outputs/skin/skin_model3.pth', map_location=torch.device('cpu'))
    model.to(device)

    F_EB, F_SX, F_SY, Sigma_EB, Sigma_SX, Sigma_SY = ReadSkinData(device)
    F_EB = F_EB.detach()
    F_SX = F_SX.detach()
    F_SY = F_SY.detach()
    F = torch.cat((F_EB, F_SX, F_SY), dim=0)

    psi, P11, P, S, sigma = model(F.requires_grad_())
    sigma = sigma.detach()
    sigx, sigy = sigma[:, 0, 0], sigma[:, 1, 1]

    ind_sx = 81
    ind_sy = 182
    sig_sx_x, sig_eb_x, sig_sy_x = sigx[ind_sx:ind_sy], sigx[0:ind_sx], sigx[ind_sy:]
    sig_sx_y, sig_eb_y, sig_sy_y = sigy[ind_sx:ind_sy], sigy[0:ind_sx], sigy[ind_sy:]
    OurSX = [sig_sx_x, sig_sx_y]
    OurEB = [sig_eb_x, sig_eb_y]
    OurSY = [sig_sy_x, sig_sy_y]

    # 组织数据
    def load_benchmark(filename_sigx, filename_sigy):
        sigx = np.load(filename_sigx)
        sigy = np.load(filename_sigy)
        sig_sx_x, sig_eb_x, sig_sy_x = sigx[ind_sx:ind_sy], sigx[0:ind_sx], sigx[ind_sy:]
        sig_sx_y, sig_eb_y, sig_sy_y = sigy[ind_sx:ind_sy], sigy[0:ind_sx], sigy[ind_sy:]
        return [sig_sx_x,sig_sx_y],[sig_eb_x,sig_eb_y],[sig_sy_x,sig_sy_y]

    cann_SX, cann_EB, cann_SY = load_benchmark('../outputs/skin/benchmark_data/cann_sigx.npy', '../outputs/skin/benchmark_data/cann_sigy.npy')
    icnn_SX, icnn_EB, icnn_SY = load_benchmark('../outputs/skin/benchmark_data/icnn_sigx.npy', '../outputs/skin/benchmark_data/icnn_sigy.npy')
    node_SX, node_EB, node_SY = load_benchmark('../outputs/skin/benchmark_data/node_sigx.npy', '../outputs/skin/benchmark_data/node_sigy.npy')

    data = [[cann_SX, icnn_SX, node_SX, OurSX],
            [cann_EB, icnn_EB, node_EB, OurEB],
            [cann_SY, icnn_SY, node_SY, OurSY]]
    gt = [[F_SX[:, 0, 0], F_EB[:, 0, 0], F_SY[:, 1, 1]],
          [Sigma_SX[:, 0, 0], Sigma_EB[:, 0, 0], Sigma_SY[:, 0, 0]],
          [Sigma_SX[:, 1, 1], Sigma_EB[:, 1, 1], Sigma_SY[:, 1, 1]]]



    def plotstresses3(x_gt, Figures):
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        labels = ['SX', 'EB', 'SY']
        for row in range(2):
            for col in range(3):
                fig = Figures[row][col]

                ax[row][col].plot(x_gt[col], fig[2], color='#98a9d0', label='ICNN')
                ax[row][col].plot(x_gt[col], fig[3], color='#75c8ae', label='CANN')
                ax[row][col].plot(x_gt[col], fig[4], color='#b0dc66', label='NODE')
                ax[row][col].plot(x_gt[col], fig[1], color='#fc9871', label='OURS')
                ax[row][col].plot(x_gt[col], fig[0], 'o', ms=1, label='Ground Truth')
                ax[row][col].legend()

                ax[row][col].set_xlabel(r'Stretch $\lambda [-]$')
                type = 'x' if row==0 else 'y'
                ax[row][col].set_ylabel(f'Cauchy stress $\sigma_{type} \: [MPa]$')
                #ax[row][col].set_ylabel(f'Cauchy stress $\sigma_x\quad[MPa]$')

        ax[0][0].set_title('Strip biaxial-x')
        ax[0][1].set_title('Equibiaxial tension')
        ax[0][2].set_title('Strip biaxial-y')
        ax[1][0].set_title('Strip biaxial-x')
        ax[1][1].set_title('Equibiaxial tension')
        ax[1][2].set_title('Strip biaxial-y')


        return fig, ax


    f1 = [Sigma_SX[:, 0, 0], sig_sx_x, icnn_SX[0], cann_SX[0],node_SX[0]]
    f2 = [Sigma_EB[:, 0, 0], sig_eb_x, icnn_EB[0], cann_EB[0],node_EB[0]]
    f3 = [Sigma_SY[:, 0, 0], sig_sy_x, icnn_SY[0], cann_SY[0],node_SY[0]]

    f4 = [Sigma_SX[:, 1, 1], sig_sx_y, icnn_SX[1], cann_SX[1],node_SX[1]]
    f5 = [Sigma_EB[:, 1, 1], sig_eb_y, icnn_EB[1], cann_EB[1],node_EB[1]]
    f6 = [Sigma_SY[:, 1, 1], sig_sy_y, icnn_SY[1], cann_SY[1],node_SY[1]]

    # sig_sy_y = sig_sy_y.numpy()
    # error = sig_sy_y - Sigma_SY[:, 1, 1].numpy()
    # print(np.max(np.abs(error)))
    # print(np.mean(error))
    # print(np.std(error))

    fig, ax = plotstresses3([F_SX[:, 0, 0], F_EB[:, 0, 0], F_SY[:, 1, 1]],
                        [[f1, f2, f3],
                        [f4, f5, f6]] )


    plt.tight_layout()
    plt.savefig('../outputs/skin.eps', dpi=600)
    plt.show()

    #----------------------------------------------------------------------
    # loss curve
    # ----------------------------------------------------------------------
    # 第一组损失
    with open("../outputs/skin/run-04-09 15_37_05_loss_train-tag-loss.json", 'r') as file:
        data_train = json.load(file)
    with open("../outputs/skin/run-04-09 15_37_05_loss_val-tag-loss.json", 'r') as file:
        data_val = json.load(file)

    x_train = np.array([data_train[i][1] for i in range(len(data_train))])
    y_train = np.array([data_train[i][2] for i in range(len(data_train))])
    x_val = np.array([data_val[i][1] for i in range(len(data_val))])
    y_val = np.array([data_val[i][2] for i in range(len(data_val))])

    # 第二组损失
    with open("../outputs/skin/run-04-11 16_35_56_loss_train-tag-loss.json", 'r') as file:
        data_train = json.load(file)
    with open("../outputs/skin/run-04-11 16_35_56_loss_val-tag-loss.json", 'r') as file:
        data_val = json.load(file)
    x_train2 = np.array([data_train[i][1] for i in range(len(data_train))])
    y_train2 = np.array([data_train[i][2] for i in range(len(data_train))])
    x_val2 = np.array([data_val[i][1] for i in range(len(data_val))])
    y_val2 = np.array([data_val[i][2] for i in range(len(data_val))])

    # 第三组损失
    with open("/Users/refantasy/code/BioConstitutiveNN/outputs/skin/run-04-11 21_40_00_loss_train-tag-loss.json", 'r') as file:
        data_train = json.load(file)
    with open("/Users/refantasy/code/BioConstitutiveNN/outputs/skin/run-04-11 21_40_00_loss_val-tag-loss.json", 'r') as file:
        data_val = json.load(file)
    x_train3 = np.array([data_train[i][1] for i in range(len(data_train))])
    y_train3 = np.array([data_train[i][2] for i in range(len(data_train))])
    x_val3 = np.array([data_val[i][1] for i in range(len(data_val))])
    y_val3 = np.array([data_val[i][2] for i in range(len(data_val))])

    fig, ax = plt.subplots(figsize=(16, 9))
    # ax.plot(x_train, (y_train), alpha=0.1,  label='sample 1 train loss')
    # ax.plot(x_train, tensorboard_smoothing(y_train, smooth=0.85), alpha=1.0, label='sample 1 smoothed train loss')
    # ax.plot(x_val, (y_val), alpha=0.1,  label='sample 1 validation loss')
    # ax.plot(x_val, tensorboard_smoothing(y_val, smooth=0.85), alpha=1.0, label='sample 1 smothed validation loss')
    #
    # ax.plot(x_train2, (y_train2), alpha=0.1,  label='sample 2 train loss')
    # ax.plot(x_train2, tensorboard_smoothing(y_train2, smooth=0.85), alpha=1.0,  label='sample 2 smoothed train loss')
    # ax.plot(x_val2, (y_val2), alpha=0.1,  label='sample 2 validation loss')
    # ax.plot(x_val2, tensorboard_smoothing(y_val2, smooth=0.85), alpha=1.0,  label='sample 2 smothed validation loss')
    #
    # ax.plot(x_train3, (y_train3), alpha=0.1, label='sample 3 train loss')
    # ax.plot(x_train3, tensorboard_smoothing(y_train3, smooth=0.85), alpha=1.0, label='sample 3 smoothed train loss')
    # ax.plot(x_val3, (y_val3), alpha=0.1, label='sample 3 validation loss')
    # ax.plot(x_val3, tensorboard_smoothing(y_val3, smooth=0.85), alpha=1.0, label='sample 3 smothed validation loss')

    # 设置全局线宽
    plt.rcParams['lines.linewidth'] = 3

    ax.plot(x_train, (y_train), color=(230/255.,238/255.,245/255.), label='sample 1 train loss')
    ax.plot(x_train, tensorboard_smoothing(y_train, smooth=0.85), color=(242/255.,122/255.,37/255.), label='sample 1 smoothed train loss')
    ax.plot(x_val, (y_val), color=(232/255.,243/255.,230/255.), label='sample 1 validation loss')
    ax.plot(x_val, tensorboard_smoothing(y_val, smooth=0.85), color=(195/255.,51/255.,39/255.), label='sample 1 smothed validation loss')

    ax.plot(x_train2, (y_train2), color=(241/255.,237/255.,246/255.), label='sample 2 train loss')
    ax.plot(x_train2, tensorboard_smoothing(y_train2, smooth=0.85), color=(123/255.,78/255.,67/255.), label='sample 2 smoothed train loss')
    ax.plot(x_val2, (y_val2), color=(250/255.,239/255.,246/255.), label='sample 2 validation loss')
    ax.plot(x_val2, tensorboard_smoothing(y_val2, smooth=0.85), color=(116/255.,116/255.,116/255.), label='sample 2 smothed validation loss')

    ax.plot(x_train3, (y_train3), color=(247/255.,247/255.,230/255.), label='sample 3 train loss')
    ax.plot(x_train3, tensorboard_smoothing(y_train3, smooth=0.85), color=(67/255.,179/255.,199/255.), label='sample 3 smoothed train loss')
    ax.plot(x_val3, (y_val3), color=(238/255.,245/255.,249/255.), label='sample 3 validation loss')
    ax.plot(x_val3, tensorboard_smoothing(y_val3, smooth=0.85), color=(242/255.,122/255.,37/255.), label='sample 3 smothed validation loss')


    ax.set_ylim(0.000001, 1000)
    ax.set_xlim(0,505000)
    ax.set_yscale('log', base=10)
    ax.grid(axis='y', color="grey",linestyle=":", linewidth=1, alpha=0.2)
    ax.legend()


    plt.tight_layout()

    plt.savefig('../outputs/skin_loss_train_val_1.eps', dpi=1200)
    plt.show()


