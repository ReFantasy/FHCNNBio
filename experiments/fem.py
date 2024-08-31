from loguru import logger
from utils import query_device
from model import BioConstitutiveModel
from continuum_mechanics import *
import sys
import torch
import numpy as np

sys.path.append('../src')


def train(model,  Fs, sigma_exps, epochs, model_saved_name):
    MseLoss = torch.nn.MSELoss()

    for epoch in range(epochs):  # 700000
        optimizer.zero_grad()

        psi, P11, P, S, sigma_pr = model(Fs)

        loss = MseLoss(sigma_exps, sigma_pr)

        print_freq = 1000
        if ((epoch + 1) % print_freq == 0):
            to_print = f'epoch {epoch+1}, train loss = {loss.item()}'
            logger.info(to_print)

        loss.backward()

        optimizer.step()

    torch.save(model, f'../outputs/{model_saved_name}.pth')


def read_simulation_data():
    # 读取数据
    F_uniaxial = np.load('../Data/fem/uniaxial_feb/Fs.npy').reshape((-1, 3, 3))
    F_equibiaxial = np.load(
        '../Data/fem/equibiaxial_feb/Fs.npy').reshape((-1, 3, 3))
    F_shear = np.load('../Data/fem/shear_feb/Fs.npy').reshape((-1, 3, 3))

    F_uniaxial = torch.from_numpy(F_uniaxial)
    F_equibiaxial = torch.from_numpy(F_equibiaxial)
    F_shear = torch.from_numpy(F_shear)
    Fs = torch.cat((F_uniaxial, F_equibiaxial, F_shear), dim=0).to(
        device=device, dtype=torch.float32).requires_grad_()

    # -----------------------------
    Sigma_uniaxial = np.load(
        '../Data/fem/uniaxial_feb/Sigmas.npy').reshape((-1, 3, 3))
    Sigma_equibiaxial = np.load(
        '../Data/fem/equibiaxial_feb/Sigmas.npy').reshape((-1, 3, 3))
    Sigma_shear = np.load(
        '../Data/fem/shear_feb/Sigmas.npy').reshape((-1, 3, 3))
    Sigma_uniaxial = torch.from_numpy(Sigma_uniaxial)
    Sigma_equibiaxial = torch.from_numpy(Sigma_equibiaxial)
    Sigma_shear = torch.from_numpy(Sigma_shear)
    Sigmas = torch.cat((Sigma_uniaxial, Sigma_equibiaxial, Sigma_shear), dim=0).to(
        device=device, dtype=torch.float32)
    return Fs, Sigmas


def read_fiber_simulation_data():
    data_dir = '../Data/fem'
    # 读取数据
    F_uniaxial_x = np.load(
        f'{data_dir}/uniaxial_x/feb/Fs.npy').reshape((-1, 3, 3))
    F_uniaxial_y = np.load(
        f'{data_dir}/uniaxial_y/feb/Fs.npy').reshape((-1, 3, 3))
    F_uniaxial_z = np.load(
        f'{data_dir}/uniaxial_z/feb/Fs.npy').reshape((-1, 3, 3))

    F_equibiaxial_xy = np.load(
        f'{data_dir}/equibiaxial_xy/feb/Fs.npy').reshape((-1, 3, 3))
    F_equibiaxial_xz = np.load(
        f'{data_dir}/equibiaxial_xz/feb/Fs.npy').reshape((-1, 3, 3))

    F_shear_xy = np.load(f'{data_dir}/shear_xy/feb/Fs.npy').reshape((-1, 3, 3))
    F_shear_xz = np.load(f'{data_dir}/shear_xz/feb/Fs.npy').reshape((-1, 3, 3))

    F_uniaxial_x = torch.from_numpy(F_uniaxial_x)
    F_uniaxial_y = torch.from_numpy(F_uniaxial_y)
    F_uniaxial_z = torch.from_numpy(F_uniaxial_z)
    F_equibiaxial_xy = torch.from_numpy(F_equibiaxial_xy)
    F_equibiaxial_xz = torch.from_numpy(F_equibiaxial_xz)
    F_shear_xy = torch.from_numpy(F_shear_xy)
    F_shear_xz = torch.from_numpy(F_shear_xz)
    Fs = torch.cat((F_uniaxial_x, F_uniaxial_y, F_uniaxial_z,
                    F_equibiaxial_xy, F_equibiaxial_xz, F_shear_xy, F_shear_xz), dim=0).to(device=device, dtype=torch.float32).requires_grad_()

    # ------------------------------------
    Sigma_uniaxial_x = np.load(
        f'{data_dir}/uniaxial_x/feb/Sigmas.npy').reshape((-1, 3, 3)).astype(np.float64)
    Sigma_uniaxial_y = np.load(
        f'{data_dir}/uniaxial_y/feb/Sigmas.npy').reshape((-1, 3, 3)).astype(np.float64)
    Sigma_uniaxial_z = np.load(
        f'{data_dir}/uniaxial_z/feb/Sigmas.npy').reshape((-1, 3, 3)).astype(np.float64)

    Sigma_equibiaxial_xy = np.load(
        f'{data_dir}/equibiaxial_xy/feb/Sigmas.npy').reshape((-1, 3, 3)).astype(np.float64)
    Sigma_equibiaxial_xz = np.load(
        f'{data_dir}/equibiaxial_xz/feb/Sigmas.npy').reshape((-1, 3, 3)).astype(np.float64)

    Sigma_shear_xy = np.load(
        f'{data_dir}/shear_xy/feb/Sigmas.npy').reshape((-1, 3, 3)).astype(np.float64)
    Sigma_shear_xz = np.load(
        f'{data_dir}/shear_xz/feb/Sigmas.npy').reshape((-1, 3, 3)).astype(np.float64)

    Sigma_uniaxial_x = torch.from_numpy(Sigma_uniaxial_x)
    Sigma_uniaxial_y = torch.from_numpy(Sigma_uniaxial_y)
    Sigma_uniaxial_z = torch.from_numpy(Sigma_uniaxial_z)
    Sigma_equibiaxial_xy = torch.from_numpy(Sigma_equibiaxial_xy)
    Sigma_equibiaxial_xz = torch.from_numpy(Sigma_equibiaxial_xz)
    Sigma_shear_xy = torch.from_numpy(Sigma_shear_xy)
    Sigma_shear_xz = torch.from_numpy(Sigma_shear_xz)
    Sigmas = torch.cat((Sigma_uniaxial_x, Sigma_uniaxial_y, Sigma_uniaxial_z,
                        Sigma_equibiaxial_xy, Sigma_equibiaxial_xz, Sigma_shear_xy, Sigma_shear_xz), dim=0).to(device=device,
                                                                                                               dtype=torch.float32).requires_grad_()

    return Fs, Sigmas


if __name__ == "__main__":

    device = query_device()  # torch.device('cpu')#
    print(f'device {device}')

    Fs, Sigmas = read_fiber_simulation_data()
    print(f'Fs {Fs.shape}')
    print(f'Sigmas {Sigmas.shape}')

    # 创建模型
    model = BioConstitutiveModel(
        num_material_params=1, num_dir=2, incomp=False, need_material_elasticity_tensor=False)
    model.to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # 训练
    train(model=model, Fs=Fs, sigma_exps=Sigmas,
          epochs=5000, model_saved_name='fem2')
