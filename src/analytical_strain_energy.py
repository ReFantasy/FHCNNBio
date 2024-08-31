import torch
import math
import numpy as np
from scipy.stats import qmc
from continuum_mechanics import GradET, GradPS, GradUT, F2C, PK1_P
from invariants import Invariant_I1, Invariant_I2, Invariant_I3


# ---------------------------------------------------------------------
#    Sample deformation gradient
# ---------------------------------------------------------------------
def SampleF(N=1000, l_bound=-0.2, u_bound=0.2):
    """
    Sample N deformation gradient samples according to the paper:
    Learning hyperelastic anisotropy from data via a tensor basis neural network
    :param N_samples:
    :param l_bound:
    :param u_bound:
    :return: sampled deformation gradient [batch, 3, 3]
    """
    sampler = qmc.LatinHypercube(d=9)
    sample = sampler.random(n=N)
    l_bounds = [l_bound for _ in range(9)]
    u_bounds = [u_bound for _ in range(9)]
    sample = qmc.scale(sample, l_bounds, u_bounds)
    sample = sample.reshape((-1, 3, 3))
    identity_matrices = np.tile(np.eye(3), (N, 1, 1))
    F = sample + identity_matrices
    return F.astype(np.float32)


def SampleF2(N=300, l_bound=0.8, u_bound=1.2):
    # sampler = qmc.LatinHypercube(d=3)
    # sample = sampler.random(n=N)
    # l_bounds = [l_bound for _ in range(3)]
    # u_bounds = [u_bound for _ in range(3)]
    # sample = qmc.scale(sample, l_bounds, u_bounds)
    # lam_ut = sample[:, 0]
    # lam_et = sample[:, 0]
    # lam_ps = sample[:, 0]

    sampler = qmc.LatinHypercube(d=1, seed=1)
    sample = sampler.random(n=N)
    lam_ut = qmc.scale(sample, l_bound, u_bound)

    sampler = qmc.LatinHypercube(d=1, seed=2)
    sample = sampler.random(n=N)
    lam_et = qmc.scale(sample, l_bound, u_bound)

    sampler = qmc.LatinHypercube(d=1, seed=3)
    sample = sampler.random(n=N)
    lam_ps = qmc.scale(sample, l_bound, u_bound)

    F_UT = GradUT(lam_ut.flatten())
    F_ET = GradET(lam_et.flatten())
    F_PS = GradPS(lam_ps.flatten())

    F = np.concatenate((F_UT, F_ET, F_PS), axis=0)
    return F.astype(np.float32)


class NeoHookean(torch.nn.Module):
    def __init__(self, c1, c2):
        super(NeoHookean, self).__init__()
        self.c1 = c1
        self.c2 = c2

    def forward(self, F):
        C = F2C(F)
        I1 = Invariant_I1(C)
        J = torch.det(F).reshape((-1, 1))
        return (
                0.5 * self.c1 * (I1 - 3.0)
                - self.c1 * torch.log(J)
                + 0.5 * self.c2 * (J - 1.0) ** 2
        )


class BonetHyperelasticity(torch.nn.Module):
    """
    Transversely isotropic hyperelasticity:
    A simple orthotropic, transversely isotropic hyperelastic constitutive equation for large strain computations
    """

    def __init__(self, c0=1.0, c1=1.0, c2=1.0):
        super(BonetHyperelasticity, self).__init__()
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        # self.neo_hookean = NeoHookean(c1=1. / math.sqrt(2), c2=10 / 3.)
        self.neo_hookean = NeoHookean(c1=self.c1, c2=self.c2)

    def forward(self, F):
        C = F2C(F)
        J = torch.det(F).reshape((-1, 1))

        n = (
            torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0])
            .reshape((-1, 1))
            .to(F.device)
        )
        N = torch.kron(n, n.reshape((1, -1)))
        N = N.repeat((F.shape[0], 1, 1))

        I4 = torch.einsum("b...ii->b...", torch.matmul(C, N)).reshape((-1, 1))
        I5 = torch.einsum("b...ii->b...", torch.matmul(torch.matmul(C, C), N)).reshape(
            (-1, 1)
        )

        psi_transverse = (I4 - 1) * (
                self.c0 + self.c1 * torch.log(J) + self.c2 * (I4 - 1)
        ) - 0.5 * self.c0 * (I5 - 1)
        return psi_transverse + self.neo_hookean(F)


class OrthotropicHyperelasticity(torch.nn.Module):
    def __init__(self, c1=5.5, c2=0.75, c3=5.0, c4=1.5, c5=1.5):
        super(OrthotropicHyperelasticity, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5

        self.n1 = torch.tensor(
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0]
        ).reshape((-1, 1))
        self.n2 = torch.tensor(
            [-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0]
        ).reshape((-1, 1))

    def forward(self, F):
        C = F2C(F)
        J = torch.det(F).reshape((-1, 1))

        n1 = self.n1.to(F.device)
        n2 = self.n2.to(F.device)
        N1 = torch.kron(n1, n1.reshape((1, -1)))
        N1 = N1.repeat((F.shape[0], 1, 1))
        N2 = torch.kron(n2, n2.reshape((1, -1)))
        N2 = N2.repeat((F.shape[0], 1, 1))

        I1 = Invariant_I1(C)

        I4 = torch.einsum("b...ii->b...", torch.matmul(C, N1)).reshape((-1, 1))
        I6 = torch.einsum("b...ii->b...", torch.matmul(C, N2)).reshape((-1, 1))

        psi = (
                self.c1 * (I1 - 3)
                + self.c1 / self.c2 * (torch.pow(J, -2 * self.c2) - 1)
                + self.c3
                * (
                        torch.exp(self.c4 * (I4 - 1) ** 4)
                        + torch.exp(self.c5 * (I6 - 1) ** 4)
                        - 2
                )
        )
        return psi


class HGO(torch.nn.Module):
    def __init__(self, mu=1.27, k1=21.6, k2=8.21, p=0.25):
        super(HGO, self).__init__()
        self.mu = mu
        self.k1 = k1
        self.k2 = k2
        self.p = p

        theta = 20.61 / 180 * torch.pi
        self.n1 = torch.tensor([math.cos(theta), 0.0, math.sin(theta)]).reshape((-1, 1))
        self.n2 = torch.tensor([math.cos(theta), 0.0, -math.sin(theta)]).reshape(
            (-1, 1)
        )

    def forward(self, F):
        C = F2C(F)
        I1 = Invariant_I1(C)

        n1 = self.n1.to(F.device).to(F.dtype)
        n2 = self.n2.to(F.device).to(F.dtype)
        N1 = torch.kron(n1, n1.reshape((1, -1)))
        N1 = N1.repeat((F.shape[0], 1, 1))
        N2 = torch.kron(n2, n2.reshape((1, -1)))
        N2 = N2.repeat((F.shape[0], 1, 1))

        I4 = torch.einsum("b...ii->b...", torch.matmul(C, N1)).reshape((-1, 1))
        I6 = torch.einsum("b...ii->b...", torch.matmul(C, N2)).reshape((-1, 1))

        psi = (self.mu * (I1 - 3) + self.k1 / (2 * self.k2) * (torch.exp(self.k2 * ((1 - self.p) * (I1 - 3) ** 2 + self.p * (I4 - 1) ** 2)) - 1)
                                  + self.k1 / (2 * self.k2) * (torch.exp(self.k2 * ((1 - self.p) * (I1 - 3) ** 2 + self.p * (I6 - 1) ** 2))- 1))

        return psi


# ---------------------------------------------------------------------
#    Analytical transversely isotropic potential
# ---------------------------------------------------------------------


class SchröderNeffEbbing(torch.nn.Module):
    def __init__(self, b=2, a1=8, a2=0, delt1=10, delt2=56, a4=2, ita=10):
        super(SchröderNeffEbbing, self).__init__()
        self.b = b
        self.a1 = a1
        self.a2 = a2
        self.delt1 = delt1
        self.delt2 = delt2
        self.a4 = a4
        self.ita = ita

    def forward(self, F):
        C = F2C(F)
        I1 = Invariant_I1(C)
        I2 = Invariant_I2(C)
        I3 = Invariant_I3(C)
        Gti = torch.diag(torch.tensor([self.b ** 2, 1.0 / self.b, 1.0 / self.b]))
        Gti = Gti.unsqueeze(0).repeat(F.shape[0], 1, 1)
        J4 = torch.einsum("b...ii->b...", torch.matmul(C, Gti)).reshape((-1, 1))

        cof_c = torch.det(C).reshape((-1, 1)).unsqueeze(2).repeat(
            1, 3, 3
        ) * torch.transpose(torch.inverse(C), 1, 2)
        J5 = torch.einsum("b...ii->b...", torch.matmul(cof_c, Gti)).reshape((-1, 1))

        W = (
                self.a1 * I1
                + self.a2 * I2
                + self.delt1 * I3
                - self.delt2 * torch.log(torch.sqrt(I3))
                + self.ita
                / (
                        self.a4
                        * (torch.einsum("b...ii->b...", Gti).reshape((-1, 1))) ** self.a4
                )
                * (J4 ** self.a4 + J5 ** self.a4)
        )

        return W


def AnalyticalDataGenerator(sampler, N_samples, energy_fn):
    F = sampler(N_samples)
    F = torch.from_numpy(F).requires_grad_().to(torch.float)
    psi = energy_fn(F)
    P = PK1_P(psi, F)
    return F.detach(), P.detach()


# ---------------------------------------------------------------------
#    ANALYTICAL STRAIN ENERGY DENSITY FUNCTIONS
# ---------------------------------------------------------------------
class MooneyRivlin6term(torch.nn.Module):
    def __init__(
            self, c10=1.6e-1, c20=-1.4e-3, c30=3.9e-5, c01=1.5e-2, c02=-2e-6, c03=1e-10
    ):
        super(MooneyRivlin6term, self).__init__()
        self.c10 = c10
        self.c20 = c20
        self.c30 = c30
        self.c01 = c01
        self.c02 = c02
        self.c03 = c03

    def forward(self, I):
        I1 = I[:, 0].reshape((-1, 1))
        I2 = I[:, 1].reshape((-1, 1))

        psi = (
                self.c10 * (I1 - 3.0)
                + self.c20 * (I1 - 3.0) ** 2
                + self.c30 * (I1 - 3.0) ** 3
                + self.c01 * (I2 - 3.0)
                + self.c02 * (I2 - 3.0) ** 2
                + self.c03 * (I2 - 3.0) ** 3
        )

        # psi = torch.exp(0.0001*(I1 - 3)) + 0.1*(I2 - 3) + 0.0001*torch.pow(I1 - 3, 2) + 0.0001*torch.pow(I1 - 3, 3) + 0.0001*torch.multiply(I1 - 3, I2 - 3)

        return psi


if __name__ == "__main__":
    pass
    # neo_hookean = NeoHookean(c1=1. / math.sqrt(2), c2=10 / 3.)
    # F, P = AnalyticalDataGenerator(sampler=SampleF, N_samples=1000, energy_fn=neo_hookean)
    # print(F.shape)
    # torch.set_default_dtype(torch.float64)

    # bonet_hyperelastic = HGO()#BonetHyperelasticity(1, 1, 1)

    # data = np.loadtxt('../Data/Schröder Neff and Ebbing/transversely_isotropic_data/uniaxial_tension.txt')
    # data = np.loadtxt('../Data/Schröder Neff and Ebbing/transversely_isotropic_data/equibiaxial_tension.txt')
    # data = np.loadtxt('../Data/Schröder Neff and Ebbing/transversely_isotropic_data/equibiaxial_tension.txt')
    # F = torch.from_numpy(data).reshape((-1,3,3)).to(torch.float64).requires_grad_(True)
    # W = bonet_hyperelastic(F)
    # P = gradients(W, F)
    # print(P)

    # Fs = pd.read_csv("/Users/refantasy/Downloads/out.CSV")
    # Fs = np.loadtxt("/Users/refantasy/Downloads/out.txt")
    # Fs = scipy.io.loadmat("/Users/refantasy/Downloads/out.MAT")

    # from io import StringIO
    # import numpy as np
    #
    # with open('/Users/refantasy/Downloads/out3.txt', 'r') as file:
    #     contents = file.read()
    # contents = contents.replace('{','').replace('}','').replace(',',' ')
    # contents = np.genfromtxt(StringIO(contents)).reshape((-1,3,3))
    #
    # F = torch.from_numpy(contents)
    #
    # F = F.to(torch.float64).requires_grad_(True)
    #
    # psi_fn = HGO()
    # psi = psi_fn(F)
    # P11, P, S, sigma = IncompressibleMaterialStress(psi,F, free_stress_ax=0)
    # print(P[:,2,2].detach().numpy())

    # from comutils.utils import bhessian2, bhessian
    #
    # class NeoHookean2(torch.nn.Module):
    #     def __init__(self, mu, lam):
    #         super(NeoHookean2, self).__init__()
    #         self.mu = mu
    #         self.lam = lam
    #
    #     def forward(self, F):
    #         C = F2C(F)
    #         I1 = Invariant_I1(C)
    #         J = torch.sqrt(torch.det(C))
    #         w =  self.mu/2 * (I1 - 3.0) - self.mu * torch.log(J) +  self.lam/2 * (torch.log(J)) ** 2
    #         CC = 4 * bhessian(w, C)
    #
    #         return w, CC
    #
    # hook2 = NeoHookean2(2.0,2.0)
    #
    # np_F = np.array([1,0,0,0,2,0,0,0,3], dtype=np.float32).reshape((3,3))
    # F = torch.from_numpy(np_F)
    # F = torch.unsqueeze(F, dim=0).requires_grad_()
    #
    #
    # w, CC = hook2(F)
    #
    #
    # def spatial_elasticity_tensor(F, CC):
    #     J = torch.det(F)
    #     se = torch.einsum("iI,jJ,kK,lL,IJKL->ijkl", F, F, F, F, CC) / J
    #     se = (torch.transpose(se, dim0=2, dim1=3) + se) / 2
    #     return se
    #
    # se = torch.vmap(spatial_elasticity_tensor)(F,CC)
    # print(se)
