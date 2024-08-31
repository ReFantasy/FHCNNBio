from continuum_mechanics import GradUT, GradET, GradPS
import numpy as np
import torch
import pandas as pd

# ----------------------------------------------------------------
# Read Rubber Data
# ----------------------------------------------------------------


def ReadRubberData(device=torch.device('cpu')):
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

    #tensor_lamUT = torch.from_numpy(lamUT).to(torch.float32).reshape((-1, 1)).requires_grad_()
    #tensor_lamET = torch.from_numpy(lamET).to(torch.float32).reshape((-1, 1)).requires_grad_()
    #tensor_lamPS = torch.from_numpy(lamPS).to(torch.float32).reshape((-1, 1)).requires_grad_()

    P11UT_exp = torch.from_numpy(P11_data[0:indET]).to(
        torch.float).reshape((-1, 1))
    P11ET_exp = torch.from_numpy(P11_data[indET:indPS]).to(
        torch.float).reshape((-1, 1))
    P11PS_exp = torch.from_numpy(P11_data[indPS:]).to(
        torch.float).reshape((-1, 1))

    F_UT = torch.from_numpy(GradUT(lamUT)).to(torch.float).requires_grad_()
    F_ET = torch.from_numpy(GradET(lamET)).to(torch.float).requires_grad_()
    F_PS = torch.from_numpy(GradPS(lamPS)).to(torch.float).requires_grad_()

    # move tensor to device
    F_UT = F_UT.to(device)
    F_ET = F_ET.to(device)
    F_PS = F_PS.to(device)
    P11UT_exp = P11UT_exp.to(device)
    P11ET_exp = P11ET_exp.to(device)
    P11PS_exp = P11PS_exp.to(device)

    return F_UT, F_ET, F_PS, P11UT_exp, P11ET_exp, P11PS_exp


def ReadSkinData(device=torch.device('cpu')):
    with open('../Data/P12AC1_bsxsy.npy', 'rb') as f:
        lamb, sigma = np.load(f, allow_pickle=True)
    lamb = lamb.astype(np.float64)
    sigma = sigma.astype(np.float64)


    lambx = lamb[:, 0]
    lamby = lamb[:, 1]
    lambz = 1. / (lambx * lamby)

    sigmax = sigma[:, 0]
    sigmay = sigma[:, 1]

    ind_sx = 81
    ind_sy = 182

    F = np.zeros([len(lambx), 3, 3])
    F[:, 0, 0] = lambx
    F[:, 1, 1] = lamby
    F[:, 2, 2] = lambz

    F_EB = F[0:ind_sx]
    F_SX = F[ind_sx:ind_sy]
    F_SY = F[ind_sy:]


    sigma = np.zeros([len(sigmax), 3, 3])
    sigma[:, 0, 0] = sigmax
    sigma[:, 1, 1] = sigmay

    Sigma_EB = sigma[0:ind_sx]
    Sigma_SX = sigma[ind_sx:ind_sy]
    Sigma_SY = sigma[ind_sy:]

    # print(F_EB)
    # print(Sigma_EB)
    # print(F_SX)
    # print(Sigma_SX)
    # print(F_SY)
    # print(Sigma_SY)

    tensor_F_EB = torch.from_numpy(F_EB).to(torch.float32).requires_grad_().to(device)
    tensor_F_SX = torch.from_numpy(F_SX).to(torch.float32).requires_grad_().to(device)
    tensor_F_SY = torch.from_numpy(F_SY).to(torch.float32).requires_grad_().to(device)
    tensor_Sigma_EB = torch.from_numpy(Sigma_EB).to(torch.float32).to(device)
    tensor_Sigma_SX = torch.from_numpy(Sigma_SX).to(torch.float32).to(device)
    tensor_Sigma_SY = torch.from_numpy(Sigma_SY).to(torch.float32).to(device)

    return tensor_F_EB, tensor_F_SX, tensor_F_SY, tensor_Sigma_EB, tensor_Sigma_SX, tensor_Sigma_SY


if __name__ == '__main__':
    ReadRubberData()
