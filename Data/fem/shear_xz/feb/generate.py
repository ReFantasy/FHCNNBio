import os
import subprocess
import numpy as np
import xml.etree.ElementTree as ET

# <logfile>
#		<element_data data="Fxx;Fxy;Fxz;Fyx;Fyy;Fyz;Fzx;Fzy;Fzz" file="F">956</element_data>
#		<element_data data="sx;sy;sz;sxy;syz;sxz" file="sigma">956</element_data>
# </logfile>
# 解析XML字符串
#root = ET.fromstring(xml_content)
# with open('uniaxial.feb', 'r') as file:
#     f = file.read().strip()
#     root = ET.fromstring(f)



def sim(root, lam):
    for elem in root.findall('Boundary'):  # 遍历根节点的所有子节点
        for sub_elem in elem.findall('bc'):
            if sub_elem.get('name') == 'PrescribedDisplacement1':
                value = sub_elem.find('value')
                value.text = str(lam)
            if sub_elem.get('name') == 'PrescribedDisplacement2':
                value = sub_elem.find('value')
                value.text = str(-lam)

    tree.write('tmp.feb', encoding='utf-8', xml_declaration=True)

    # 执行febio
    command = "/home/tdl/Software/FEBioStudio/bin/febio4 tmp.feb -silent"
    result = subprocess.run(command, shell=True)

    # 读取变形梯度和应力
    with open('F', 'r') as file:
        contents = file.read().strip()
        contents = contents.split('\n')
    #F = np.array(contents[-1].split('\x20')[1:]).astype(np.float64)
    Fs = []
    Fs.append(np.array(contents[-4].split('\x20')[1:]).astype(np.float64))
    Fs.append(np.array(contents[-3].split('\x20')[1:]).astype(np.float64))
    Fs.append(np.array(contents[-2].split('\x20')[1:]).astype(np.float64))
    Fs.append(np.array(contents[-1].split('\x20')[1:]).astype(np.float64))


    with open('sigma', 'r') as file:
        contents = file.read().strip()
        contents = contents.split('\n')
    # sigma_upper = contents[-1].split('\x20')[1:]
    # sigma = [sigma_upper[0], sigma_upper[3], sigma_upper[5],
    #          sigma_upper[3], sigma_upper[1], sigma_upper[4],
    #          sigma_upper[5], sigma_upper[4], sigma_upper[2]]
    #sigma = np.array(sigma).astype(np.float64)
    #print(sigma)
    sigmas = []
    for i in [-4,-3,-2,-1]:
        sigma_upper = contents[i].split('\x20')[1:]
        sigma = [sigma_upper[0], sigma_upper[3], sigma_upper[5],
                 sigma_upper[3], sigma_upper[1], sigma_upper[4],
                 sigma_upper[5], sigma_upper[4], sigma_upper[2]]
        sigmas.append(sigma.astype(np.float64))

    # 删除临时文件
    returncode = subprocess.run("rm -f tmp.* F sigma", shell=True)
    return Fs, sigmas


if __name__ == "__main__":
    # 定义拉伸量
    lams = np.linspace(-0.025, 0.025, 100)
    Fs = []
    sigmas = []

    tree = ET.parse('shear_xz.feb')  # 解析XML文件
    root = tree.getroot()
    for i, lam in enumerate(lams):
        print(f'simulation {i+1} lam = {lam}')
        F, sigma = sim(root, lam)
        for f,s in zip(F, sigma):
            Fs.append(f)
            sigmas.append(s)
            #print(f)
            #print(s)

    np.save('Fs.npy', Fs)
    np.save('Sigmas.npy', sigmas)







