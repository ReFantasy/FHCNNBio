import time
import numpy as np
import torch


def split_data_uniform(X, y, test_size=0.3):
    n = X.shape[0]
    skip = int(1.0 / test_size)
    indices = np.arange(n)
    test_indices = indices[1:-1:skip]
    train_indices = np.delete(indices, test_indices)
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def tensorboard_smoothing(x, smooth=0.6):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):
        x[i] = (x[i - 1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            only_inputs=True,
            retain_graph=True,
        )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def query_device(spec=None, full_mem=None):
    if spec != None:
        device = torch.device(spec)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
            free_memory, total_memory = torch.cuda.mem_get_info(device.index)
            if full_mem != None:
                torch.cuda.caching_allocator_alloc(int(free_memory * 0.9), device.index)
            # torch.cuda.caching_allocator_alloc(int(total_memory * 0.98), device.index)

        elif torch.backends.mps.is_available():
            # device = torch.device("mps")
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    # print(f"Available device: {device}")
    return device


def Hessian(J, F, dim=3):
    """
    计算标量 J 对 张量 F 的二阶导数\n
    返回 4阶张量
    """
    dJ_dF = gradients(J, F)
    dJ_dF = torch.flatten(dJ_dF)
    dJ_dF2 = []
    for it in dJ_dF:
        d2 = gradients(it, F)
        dJ_dF2.append(d2)
    H = torch.cat(dJ_dF2)
    H = H.reshape(dim, dim, dim, dim)
    return H


def bhessian(J, F):
    batch_size = J.shape[0]
    dJ_dF = gradients(J, F)

    dJ_dF2_list = []
    for i in range(3):
        for j in range(3):
            tmp = gradients(dJ_dF[:, i, j], F)
            dJ_dF2_list.append(tmp)
    y = torch.stack(dJ_dF2_list, dim=0)
    y = y.permute(1, 0, 2, 3)
    y = y.reshape((batch_size, 3, 3, 3, 3))
    return y


def bhessian2(J, F):
    """
    J : 标量
    F : 9元素向量
    """
    # batch_size = J.shape[0]
    dJ_dF = gradients(J, F)

    dJ_dF2_list = []
    for i in range(9):
        tmp = gradients(dJ_dF[:, i], F)
        dJ_dF2_list.append(tmp)
    y = torch.stack(dJ_dF2_list, dim=0)
    y = y.permute(1, 0, 2)
    # y = y.reshape((batch_size, 9,9))
    return y


class TrainTimer:
    def __init__(self, total_step=1000, step_size=1):
        self.last_time = time.time()
        self.total_step = total_step
        self.step_size = step_size

    def elapsed_time(self, cur_step=1):
        elapsed_time = time.time() - self.last_time
        rest_time = (elapsed_time / self.step_size) * (self.total_step - cur_step)
        # print("el time: ", elapsed_time)
        self.last_time = time.time()
        m, s = divmod(rest_time, 60)
        h, m = divmod(m, 60)
        return h, m, s


def ReduceArray(arr, size=35):
    arr_len = len(arr)
    step = int(arr_len / size)

    new_arr = []
    for i in range(2, arr_len-2, step):
        new_arr.append(arr[i])

    return np.array(new_arr)


def ReduceTensorArray(arr, size=20):
    reduced_arr = ReduceArray(arr.cpu().numpy(), size=size)
    return torch.from_numpy(reduced_arr)
