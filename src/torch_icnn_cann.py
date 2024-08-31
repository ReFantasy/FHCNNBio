import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------
#    ICNN 模型 2
# ----------------------------------------------


class ICNNLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, non_negative_weights=False):
        super(ICNNLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.non_negative_weights = non_negative_weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # 初始化权重和偏置
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.xavier_normal_(self.weight)
        if self.bias != None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        w = self.weight
        if self.non_negative_weights == True:
            w = torch.exp(self.weight)
        return torch.nn.functional.linear(input, w, self.bias)

    def __repr__(self):
        return "NonNegativeLinear(in_features=%d, out_features=%d, bias=%s)" % (
            self.in_features, self.out_features, str(self.bias))


class FICNN(nn.Module):
    '''
    Refer to paper: Input Convex Neural Networks
    '''

    def __init__(self, layers=[3, 2, 5, 7, 4]):
        super(FICNN, self).__init__()
        num_hidden_layers = len(layers) - 2  # 3

        Wzs = []
        for i in range(num_hidden_layers):
            Wzs.append(ICNNLinear(
                layers[i + 1], layers[i + 2], bias=False, non_negative_weights=True))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        # 当能量输入为 I 时，保证凸性
        Wxs.append(ICNNLinear(layers[0], layers[1],
                   bias=True, non_negative_weights=True))
        for i in range(num_hidden_layers):
            Wxs.append(torch.nn.Linear(layers[0], layers[i + 2], bias=True))
        self.Wxs = torch.nn.ModuleList(Wxs)

        self.act = torch.nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wxs[0](x))
        for Wz, Wx in zip(self.Wzs[:-1], self.Wxs[1:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)


class PICNN(nn.Module):
    '''
    Refer to paper: Input Convex Neural Networks
    '''

    def __init__(self, x_dim=10, y_dim=3, u_dims=[4, 3, 4], z_dims=[2, 7, 4, 1]):
        super(PICNN, self).__init__()

        Wy = []
        Wy.append(ICNNLinear(y_dim, z_dims[0],
                  bias=True, non_negative_weights=True))
        for i in range(len(z_dims) - 1):
            Wy.append(torch.nn.Linear(y_dim, z_dims[i + 1]))
        self.Wy = torch.nn.ModuleList(Wy)

        Wz = []
        for i in range(len(z_dims) - 1):
            Wz.append(ICNNLinear(
                z_dims[i], z_dims[i + 1], bias=False, non_negative_weights=True))
        self.Wz = torch.nn.ModuleList(Wz)

        Wu = []
        Wu.append(torch.nn.Linear(x_dim, u_dims[0]))
        for i in range(len(u_dims) - 1):
            Wu.append(torch.nn.Linear(u_dims[i], u_dims[i + 1]))
        self.Wu = torch.nn.ModuleList(Wu)

        Wuu = []
        for i in range(len(u_dims)):
            Wuu.append(torch.nn.Linear(u_dims[i], z_dims[i + 1]))
        self.Wuu = torch.nn.ModuleList(Wuu)

        Wzu = []
        for i in range(len(u_dims)):
            Wzu.append(torch.nn.Linear(u_dims[i], z_dims[i]))
        self.Wzu = torch.nn.ModuleList(Wzu)

        Wyu = []
        Wyu.append(torch.nn.Linear(x_dim, y_dim))
        for i in range(len(u_dims)):
            Wyu.append(torch.nn.Linear(u_dims[i], y_dim))
        self.Wyu = torch.nn.ModuleList(Wyu)

        self.act = torch.nn.Softplus()
        self.act_u = torch.nn.Sigmoid()

    def forward(self, x, y):

        z = self.Wy[0](y * self.Wyu[0](x))  # --> z1
        u = self.Wu[0](x)  # --> u1

        for Wu, Wz, Wy, Wuu, Wzu, Wyu in zip(self.Wu[1:], self.Wz[:-1], self.Wy[1:-1], self.Wuu[:-1], self.Wzu[:-1],
                                             self.Wyu[1:-1]):  # z1->z2, z2->z3
            uz = torch.nn.functional.softplus(Wzu(u))  # Wzu(u)
            uy = Wyu(u)  # torch.nn.functional.sigmoid(Wyu(u))
            z = self.act(Wz(z * uz) + Wy(y * uy) +
                         torch.nn.functional.sigmoid(Wuu(u)))
            u = self.act_u(Wu(u))

        return self.Wz[-1](z * self.act(self.Wzu[-1](u))) + self.Wy[-1](y * self.Wyu[-1](u)) + self.Wuu[-1](u)


class PICNNWapper(PICNN):  # (x_dims=[4, 3, 4], y_dims=[2, 7, 4, 1]):
    """
    关于x凸
    """

    def __init__(self, x_dims=[4, 3, 4], y_dims=None):
        if y_dims == None:
            y_dims = x_dims[:-1]
        assert len(y_dims) == (len(x_dims) - 1)
        self.y_dims = y_dims
        x_dim = y_dims[0]
        u_dims = y_dims[1:]
        y_dim = x_dims[0]
        z_dims = x_dims[1:]
        super().__init__(x_dim=x_dim, y_dim=y_dim, u_dims=u_dims, z_dims=z_dims)

    def forward(self, x, y=None):
        if y == None:
            batch_size = x.shape[0]
            y = torch.ones(size=(batch_size, self.y_dims[0])).to(x.device)
        return super().forward(y, x)


def test_picnn():
    torch.set_default_dtype(torch.float64)
    # ------------------------------------------------
    #    Test 1
    # ------------------------------------------------
    # x_dim = 9
    # y_dim = 4
    # net = PICNN(x_dim=x_dim, y_dim=y_dim)
    #
    # batch_size = 4000
    #
    # for i in range(3000):
    #     x = torch.rand((batch_size, x_dim))
    #     y1 = torch.rand((batch_size, y_dim)) * 1000000 - 1000000 / 2
    #     y2 = torch.rand((batch_size, y_dim)) * 1000000 - 1000000 / 2
    #     theta = 0.5  # torch.rand((batch_size, 1))
    #
    #     lhs = theta * y1 + (1.0 - theta) * y2
    #     lhs = net(x, lhs)
    #
    #     rhs = theta * net(x, y1) + (1.0 - theta) * net(x, y2)
    #
    #     assert ((rhs - lhs) >= -0.000001).sum() == batch_size

    # ------------------------------------------------
    #    Test 2
    # ------------------------------------------------
    print('Testing convexity')
    n = 640
    dim = 123
    dimc = 11
    picnn = PICNN(x_dim=dimc, y_dim=dim, u_dims=[
                  4, 3, 4], z_dims=[2, 7, 4, 123])
    x1 = torch.randn(n, dim)
    x2 = torch.randn(n, dim)
    c = torch.randn(n, dimc)
    print(np.all((((picnn(c, x1) + picnn(c, x2)) / 2 -
          picnn(c, (x1 + x2) / 2)) > 0).cpu().data.numpy()))

    # print('Visualizing convexity')
    # dim = 1
    # dimh = 16
    # dimc = 1
    # num_hidden_layers = 2
    # picnn = PICNN(x_dim=dimc, y_dim=dim, u_dims=[4, 3, 4], z_dims=[2, 7, 4, 1])
    #
    # c = torch.zeros(1, dimc)
    # x = torch.linspace(-10, 10, 100).view(100, 1)
    # for c_ in np.linspace(-5, 5, 10):
    #     plt.plot(x.squeeze().numpy(), picnn(c + c_, x).squeeze().data.numpy())
    # plt.show()


class MooneyRivlin6term(torch.nn.Module):
    def __init__(self):
        super(MooneyRivlin6term, self).__init__()

    def forward(self, X):
        I = X[:, 0].reshape((-1, 1))
        J = X[:, 1].reshape((-1, 1))

        Ic = I * 3.0
        IIc = J * 3.0
        # IIIc = X[2]

        # psi = self.c10 * (Ic - 3.0) + self.c20 * (Ic - 3.0) ** 2 + self.c30 * (Ic - 3.0) ** 3 + self.c01 * (
        #         IIc - 3.0) + self.c02 * (
        #               IIc - 3.0) ** 2 + self.c03 * (IIc - 3.0) ** 3

        psi = torch.exp(0.0001 * (Ic - 3)) + 0.1 * (IIc - 3) + 0.0001 * torch.pow(Ic - 3, 2) + 0.0001 * torch.pow(
            Ic - 3, 3) + 0.0001 * torch.multiply(Ic - 3, IIc - 3)

        return psi


if __name__ == "__main__":
    pass
    # test_picnn()

    # torch.set_default_dtype(torch.float64)
    # picnn = PICNNWapper(x_dims=[3, 4, 5, 7, 1], y_dims=[7, 8, 2, 1])
    # batch_size = 4000
    #
    # for i in range(1000):
    #     u = torch.rand((batch_size, 7))
    #     theta = torch.rand((batch_size, 1))
    #     x = torch.rand((batch_size, 3)) * 1000000 - 1000000 / 2
    #     y = torch.rand((batch_size, 3)) * 1000000 - 1000000 / 2
    #
    #     lhs = theta * x + (1.0 - theta) * y
    #     lhs = picnn(lhs, u)
    #
    #     rhs = theta * picnn(x, u) + (1.0 - theta) * picnn(y, u)
    #
    #     assert ((rhs - lhs) >= -0.000001).sum() == batch_size

    # for i in range(1000):
    #     u = torch.rand((batch_size, 3))
    #     theta = torch.rand((batch_size, 1))
    #     x = torch.rand((batch_size, 7)) * 1000000 - 1000000 / 2
    #     y = torch.rand((batch_size, 7)) * 1000000 - 1000000 / 2
    #
    #     lhs = theta * x + (1.0 - theta) * y
    #     lhs = picnn(u, lhs)
    #
    #     rhs = theta * picnn(u,x) + (1.0 - theta) * picnn(u, y)
    #
    #     assert ((rhs - lhs) >= -0.000001).sum() == batch_size
