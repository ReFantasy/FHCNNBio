from utils import gradients
import torch
import numpy as np

# ----------------------------------------------------------------------------------------------------------
#    通过拉伸量计算形变梯度
# ----------------------------------------------------------------------------------------------------------


def GradUT(lam: np.ndarray):
    """
    Deformation gradient for incompressible uniaxial tension loading
    """
    F = np.zeros([len(lam), 3, 3])
    F[:, 0, 0] = lam
    F[:, 1, 1] = 1.0 / (np.sqrt(lam))
    F[:, 2, 2] = 1.0 / (np.sqrt(lam))
    return F


def GradET(lam: np.ndarray):
    """
    Deformation gradient for incompressible equi-biaxial loading
    """
    F = np.zeros([len(lam), 3, 3])
    F[:, 0, 0] = lam
    F[:, 1, 1] = lam
    F[:, 2, 2] = 1.0 / lam ** 2
    return F


def GradPS(lam: np.ndarray):
    """
    Deformation gradient for incompressible pure shear loading
    """
    F = np.zeros([len(lam), 3, 3])
    F[:, 0, 0] = lam
    F[:, 1, 1] = 1.0
    F[:, 2, 2] = 1 / lam
    return F


# ----------------------------------------------------------------------------------------------------------
#    通用模块
# ----------------------------------------------------------------------------------------------------------
def F2C(F):
    """
    Right Cauchy-Green tensor : :math:`C=F^T F`
    """
    F_t = F.transpose(1, 2)
    C = torch.matmul(F_t, F)
    return C


def PK1_P(Psi, F):
    """
    First Piola Kirchhoff stress tensor: :math:`P = \\frac{\partial \Psi}{\partial F}`
    能量对 F 的导数
    """
    grad = gradients(Psi, F)
    return grad


def PK2_S(P, F):
    """
    Second Piola Kirchhoff stress tensor:
    .. math:: S = F^{-1} P
    """
    return torch.matmul(torch.inverse(F), P)


def Sigma(P, F, J):
    """
    Cauchy stress tensor:
    .. math:: \sigma = J^{-1} P F^T
    """
    one_over_J = torch.unsqueeze(1.0 / J, 2).repeat((1, 3, 3))
    return torch.multiply(one_over_J, torch.matmul(P, F.transpose(1, 2)))


# ----------------------------------------------------------------------------------------------------------
#    拉格朗日乘子
# ----------------------------------------------------------------------------------------------------------
def LagrangianMultiplier(dPsi_iso_dF, F, free_stress_ax):
    '''
    针对UT ET PS 测试计算 Lagrangian Multiplier
    '''
    # P_iso = gradients(Psi, F)
    # J = torch.det(F)
    # dJdF = gradients(J, F)
    # lag_mul = P_iso[:, 2, 2] / dJdF[:, 2, 2]

    F_int = torch.inverse(F.transpose(1, 2))
    # lag_mul = dPsi_iso_dF[:, 2, 2] / F_int[:, 2, 2]
    lag_mul = dPsi_iso_dF[:, free_stress_ax, free_stress_ax] / \
        F_int[:, free_stress_ax, free_stress_ax]
    # print('-----------------')
    # print(lag_mul)

    # print('P_iso', P_iso)
    # print('F_int', F_int)
    return lag_mul


# ----------------------------------------------------------------------------------------------------------
#    Stress Tensor
# ----------------------------------------------------------------------------------------------------------
def compute_stress_tensor(P, F):
    """
    Common parts from post_Psi & post_Psi_incomp
    """
    P11 = P[:, 0, 0]

    S = PK2_S(P, F)
    J = torch.det(F).unsqueeze(1)
    sigma = Sigma(P, F, J)
    return P11, P, S, sigma


def CompressibleMaterialStress(Psi, F):
    """
    Deals with everything after the strain-energy is used [variant for compressible materials] (stresses)
    """
    P = PK1_P(Psi, F)
    return compute_stress_tensor(P, F)


def IncompressibleMaterialStress(Psi_iso, F, lag_mul=None, free_stress_ax=2):
    """
    拉伸测试：设置 lag_mul 为 None
    非拉伸测试：直接传入 lag_mul 的值
    """
    dPsi_iso_dF = gradients(Psi_iso, F)

    # print(P_iso[:,2,2])

    if lag_mul == None:
        lag_mul = LagrangianMultiplier(
            dPsi_iso_dF, F, free_stress_ax=free_stress_ax)

    # P = P_iso - lag_mul * dJdF ???????

    lag_mul = torch.unsqueeze(torch.unsqueeze(lag_mul, 1), 2).repeat(1, 3, 3)

    P = dPsi_iso_dF - lag_mul * torch.inverse(F.transpose(1, 2))
    return compute_stress_tensor(P, F)


# ----------------------------------------------------------------------------------------------------------
#    Warpper FUNCTIONS
# ----------------------------------------------------------------------------------------------------------
def ComputeMaterialStress(psi, F, incomp):
    if incomp == True:
        return IncompressibleMaterialStress(Psi_iso=psi, F=F)
    else:
        return CompressibleMaterialStress(psi, F)
