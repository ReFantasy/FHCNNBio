import torch


# ----------------------------------------------------------------------------------------------------------
#    Strain Invariants
# ----------------------------------------------------------------------------------------------------------
def Invariant_I1(C: torch.Tensor):
    """
    Invariants I
    :param C: [batch, 3,3]
    """
    return torch.einsum('b...ii->b...', C).reshape((-1, 1))


def Invariant_I2(C: torch.Tensor):
    """
    Invariants I2
    :param C: [batch, 3,3]
    """
    C_trace_square = (torch.einsum(
        'b...ii->b...', C).square()).reshape((-1, 1))
    C_square_trace = torch.einsum(
        'b...ii->b...', torch.matmul(C, C)).reshape((-1, 1))
    return 0.5 * (C_trace_square - C_square_trace)


def Invariant_I3(C: torch.Tensor):
    """
    Invariants I3
    :param C: [batch, 3,3]
    """
    det = torch.det(C).reshape((-1, 1))
    return det


def Invariant_I4(C: torch.Tensor, A: torch.Tensor):
    """
    Invariants I4
    :param C: [batch, 3,3]
    :param A: [batch, num_dir, 3, 3]
    """
    num_dir = A.shape[1]
    C = torch.unsqueeze(C, dim=1).repeat((1, num_dir, 1, 1))
    tmp = torch.matmul(C, A)
    trace = torch.einsum('b...ii->b...', tmp)
    return trace


def Invariant_I5(C: torch.Tensor, A: torch.Tensor):
    """
    Invariants I5
    :param C: [batch, 3,3]
    :param A: [batch, num_dir, 3, 3]
    """
    num_dir = A.shape[1]
    C = torch.unsqueeze(C, dim=1).repeat((1, num_dir, 1, 1))
    C2 = torch.matmul(C, C)
    tmp = torch.matmul(C2, A)
    trace = torch.einsum('b...ii->b...', tmp)
    return trace


def Invariant_I8(C: torch.Tensor, Ni: torch.Tensor, Nj: torch.Tensor):
    """
    Invariants I8
    :param C: [batch, 3,3]
    :param Ni: [batch, num_dir, 3]
    :param Nj: [batch, num_dir, 3]
    """
    num_dir = Ni.shape[1]
    C = torch.unsqueeze(C, dim=1).repeat((1, num_dir * num_dir, 1, 1))
    Ni = Ni.repeat_interleave(num_dir, dim=1)
    Nj = Nj.repeat((1, num_dir, 1))

    Ni = Ni.unsqueeze(2)
    Nj = Nj.unsqueeze(3)

    lhs = torch.matmul(Ni, Nj)
    # lhs = torch.squeeze(lhs)
    lhs = torch.squeeze(lhs, dim=3)
    lhs = torch.squeeze(lhs, dim=2)

    rhs = torch.matmul(Ni, torch.matmul(C, Nj))
    # rhs = torch.squeeze(rhs)
    rhs = torch.squeeze(rhs, dim=3)
    rhs = torch.squeeze(rhs, dim=2)
    I8 = torch.multiply(lhs, rhs)

    return I8


def Invariant_I9(Ni: torch.Tensor, Nj: torch.Tensor):
    """
    Invariants I9
    :param C: [batch, 3,3]
    :param Ni: [batch, num_dir, 3]
    :param Nj: [batch, num_dir, 3]
    """
    num_dir = Ni.shape[1]
    Ni = Ni.repeat_interleave(num_dir, dim=1)
    Nj = Nj.repeat((1, num_dir, 1))

    Ni = Ni.unsqueeze(2)
    Nj = Nj.unsqueeze(3)

    tmp = torch.matmul(Ni, Nj)
    # tmp = torch.squeeze(tmp)
    tmp = torch.squeeze(tmp, dim=3)
    tmp = torch.squeeze(tmp, dim=2)
    I9 = torch.pow(tmp, 2)

    return I9


# ----------------------------------------------------------------------------------------------------------
#    Warpper FUNCTIONS
# ----------------------------------------------------------------------------------------------------------
def GenerateInvariant(C, A, Ni, Nj):
    I1 = Invariant_I1(C)

    I2 = Invariant_I2(C)

    I3 = Invariant_I3(C)

    if A == None:
        return torch.cat((I1, I2, I3), dim=1)
    else:
        I4 = Invariant_I4(C, A)
        I5 = Invariant_I5(C, A)
        I8 = Invariant_I8(C, Ni, Nj)
        I9 = Invariant_I9(Ni, Nj)
        return torch.cat((I1, I2, I3, I4, I5, I8, I9), dim=1)
