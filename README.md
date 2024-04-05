# GNNinEAFramework

学士学位论文代码。以 [EAkit](https://github.com/THU-KEG/EAkit) 为基础，研究基于图神经网络的知识图谱对齐方法的性能影响因素。

## 总体架构

![image](https://github.com/TimDinggw/GNNinEAFramework/blob/main/fig/framework.png)

## 运行环境

- python                             3.9.12
- torch                                1.12.0
- torch-cluster                  1.6.0+pt112cpu
- torch-geometric            2.4.0
- torch-scatter                  2.1.0+pt112cpu
- torch-sparse                   0.6.16+pt112cpu
- torch-spline-conv         1.2.1+pt112cpu

## 主要数据集及模型

数据集:

- DBP15K 从 [HGCN](https://github.com/StephanieWyt/HGCN-JE-JR/) 获取 [数据集](https://drive.google.com/drive/folders/1mfaeLXdqFnOHLYBXiTHWI7MLwtfTgPYQ) 放至 ./data/ 下

- SRPRS  从 [EvalFramework](https://github.com/YF-SHU/EvalFramework) 获取放至 ./data/ 下，但没有名称初始化词典。

  

编码器:

- GCN-Align from EAkit
- KECG from https://github.com/THU-KEG/KECG
- GraphSAGE, AGNN from torch_geometric.nn

解码器:

- Align from EAkit

## 运行方法

主要参数：

```
python run.py

--encoder default="GCN-Align", choices=['GCN-Align', 'GraphSAGE', 'KECG', 'AGNN']

--hiddens default="300,300,300" (including in_dim and out_dim)

--ent_init default='random', choices=['random', 'name']

--skip_conn default='none', choices=['none', 'highway', 'concatall', 'concat0andl', 'residual', 'concatallhighway']

--activation default='elu', choices=['none', 'elu', 'relu', 'tanh', 'sigmoid'])
```

例子:

```
python run.py --encoder="GCN-Align" --hiddens="300,300,300" --ent_init="random" --skip_conn="none" --activation="none"
```

## 参考代码

EAkit(https://github.com/THU-KEG/EAkit)
