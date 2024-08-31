from flask import Flask, request
import numpy as np
import torch
import sys
sys.path.append('../src')


app = Flask(__name__)
# device = query_device(full_mem=0)
# device = torch.device("cuda:0")
device = torch.device("cpu")


@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获数据
    data = request.get_data()
    float_array = np.frombuffer(data)
    float_array = float_array.astype(np.float32)  # 如果需要指定数据类型
    np_F = float_array.reshape((3, 3))

    # 运行模型进行预测
    F = torch.from_numpy(np_F)
    F = torch.unsqueeze(F, dim=0).requires_grad_().to(device)

    psi, P11, P, S, sigma, CC = model(F)

    # 计算 spatial elasticity tensor
    def spatial_elasticity_tensor(F, CC):
        J = torch.det(F)
        se = torch.einsum("iI,jJ,kK,lL,IJKL->ijkl", F, F, F, F, CC) / J
        se = (torch.transpose(se, dim0=2, dim1=3) + se) / 2
        return se

    se = torch.vmap(spatial_elasticity_tensor)(F, CC)

    # 返回预测结果
    return sigma.cpu().detach().numpy().astype(np.float64).tobytes() + se.cpu().detach().numpy().astype(np.float64).tobytes()


model = torch.load('../outputs/fem2.pth')
model.need_material_elasticity_tensor = True
# model.to(torch.device('cpu'))
model.to(device)


# app.run(debug=False, host="0.0.0.0", port=6000)

# 部署
# gunicorn -w 18 -b localhost:6000 'fem_server:app'
