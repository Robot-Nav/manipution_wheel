# TidyBot++：开源全向移动操作机器人平台——从硬件设计到策略学习的完整方案

> **GitHub 仓库**：[https://github.com/Robot-Nav/manipution_wheel](https://github.com/Robot-Nav/manipution_wheel)

> **原始项目**：[https://github.com/jimmyyhwu/tidybot2/tree/main](https://github.com/jimmyyhwu/tidybot2/tree/main)

> **项目主页**：[https://tidybot2.github.io/](https://tidybot2.github.io/)

> **论文**：[arXiv:2412.10447](https://arxiv.org/abs/2412.10447)

> **发表会议**：CoRL 2024（Conference on Robot Learning）

> **作者**：Jimmy Wu¹'², William Chong², Robert Holmberg³, Aaditya Prasad², Yihuai Gao², Oussama Khatib², Shuran Song², Szymon Rusinkiewicz¹, Jeannette Bohg²（¹普林斯顿大学 ²斯坦福大学 ³Dexterity）

---

## 一、项目简介

在机器人学习领域，模仿学习（Imitation Learning）正展现出越来越大的潜力。然而，一个关键瓶颈在于**数据的可获取性**——与自然语言处理可以从互联网获取海量数据不同，真实世界的机器人操作数据采集极其困难。特别是在移动操作（Mobile Manipulation）场景下，缺乏合适的研究硬件更是制约了这一方向的发展。

**TidyBot++** 正是为解决这一痛点而生。它是由斯坦福大学和普林斯顿大学联合提出的**开源全向移动操作机器人平台**，旨在为机器人学习（尤其是模仿学习）提供一个低成本、鲁棒、灵活的移动操作解决方案，使研究者能够轻松采集大规模人类遥操作示范数据，并训练出可在真实家庭环境中执行多种操作任务的策略。

TidyBot++ 可以完成开门、擦拭台面、装载洗碗机、倒垃圾、浇花、装载洗衣机等多种家庭任务——这些任务均通过 Diffusion Policy 训练后实现全自主执行。

### 核心创新点

1. **全向移动底座（Holonomic Base）**：采用动力脚轮（Powered Caster）设计，实现全向运动，可独立且同时控制全部平面自由度 $(x, y, \theta)$，消除非全向底座的运动学约束
2. **手机遥操作接口**：基于 WebXR 的手机网页端遥操作，无需额外硬件，降低数据采集门槛
3. **完整策略学习流水线**：从数据采集、格式转换、策略训练到策略推理的全流程支持

### 为什么全向运动如此重要？

非全向机器人（如差速驱动平台）存在运动学约束——最显著的后果是**无法侧向移动**。例如，汽车无法直接侧向驶入街边停车位，必须执行多步平行停车操作。

相比之下，全向机器人没有运动学约束，可以同时独立控制所有三个自由度。日常生活中最常见的全向车辆就是**办公椅**——脚轮的偏移设计使得轮子自动对齐运动方向，实现任意方向的平滑移动。TidyBot++ 的底座就像一把"电动办公椅"，四个动力脚轮使其能够瞬间向任意方向加速，极大简化了遥操作和策略学习。

---

## 二、系统架构与模块总览

### 2.1 代码模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 仿真环境 | `mujoco_env.py` | 基于 MuJoCo 的仿真环境，支持物理仿真、图像渲染、状态观测 |
| 真实环境 | `real_env.py` | 真实机器人环境接口，通过 RPC 连接底座和机械臂控制器 |
| 逆运动学 | `ik_solver.py` | 基于阻尼最小二乘法的数值 IK 求解器，含零空间优化 |
| 底座控制器 | `base_controller.py` | 全向移动底座的运动学建模与速度/位置控制 |
| 机械臂控制器 | `arm_controller.py` | 关节空间柔顺控制器，含摩擦补偿 |
| 机械臂驱动 | `kinova.py` | Kinova Gen3 力矩控制接口，基于 Pinocchio 重力补偿 |
| 遥操作策略 | `policies.py` | WebXR 手机遥操作与远程策略推理 |
| 策略服务器 | `policy_server.py` | ZMQ 策略推理服务器，集成 Diffusion Policy |
| 数据存储 | `episode_storage.py` | 示范数据的读写（MP4视频 + pickle） |
| 数据转换 | `convert_to_robomimic_hdf5.py` | 将原始数据转换为 robomimic HDF5 格式 |
| 主入口 | `main.py` | 数据采集主程序 |
| 数据回放 | `replay_episodes.py` | 示范数据回放 |
| 相机 | `cameras.py` | Logitech 基座相机与 Kinova 腕部相机接口 |
| 常量 | `constants.py` | 全局配置参数 |
| MuJoCo模型 | `models/` | TidyBot 与 Kinova Gen3 + Robotiq 2F-85 的 MJCF 模型 |

### 2.2 仿真环境架构

```
┌─────────────────────────────────────────────────────┐
│                   Dev Machine                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ MuJoCo   │   │ Renderer │   │ Visualizer       │ │
│  │ Physics  │◄──│ (Offscrn)│──►│ (cv2.imshow)     │ │
│  │ Loop     │   └──────────┘   └──────────────────┘ │
│  │(Proc)    │                                        │
│  └────┬─────┘   ┌──────────────┐                    │
│       │         │ SharedMemory │                    │
│       └────────►│ (State+Image)│◄──────────────────┐│
│                 └──────────────┘                   ││
│  ┌──────────────────────────────────────────────┐  ││
│  │            MujocoEnv (Main Process)           │  ││
│  │  get_obs() ← SharedMemory                   │  ││
│  │  step(action) → CommandQueue → MujocoSim     │  ││
│  └──────────────────────────────────────────────┘  ││
│                     │                               ││
│              ┌──────┴──────┐                        ││
│              │  Policy     │                        ││
│              │(Teleop/Remote)                       ││
│              └─────────────┘                        ││
└─────────────────────────────────────────────────────┘│
```

### 2.3 真实机器人架构

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Phone     │────►│   Router    │◄────│  GPU Laptop  │
│ (WebXR App) │ WiFi│ (Wireless) │ WiFi│(Policy Train │
└─────────────┘     └──────┬──────┘     │  & Inference)│
                           │            └──────────────┘
                    ┌──────┴──────┐
                    │  Mini PC    │
                    │ (Robot Side)│
                    │ ┌─────────┐ │
                    │ │Base Srv │ │ ← 250Hz 实时控制
                    │ │Arm Srv  │ │ ← 1kHz 实时控制
                    │ └─────────┘ │
                    └─────────────┘
```

真实环境中，手机通过 WiFi 连接路由器发送遥操作指令；GPU 笔记本电脑负责策略推理并通过 ZMQ 与机器人端通信；Mini PC 运行底座（250Hz）和机械臂（1kHz）的实时控制服务。

---

## 三、核心算法原理

### 3.1 全向移动底座运动学

TidyBot++ 的底座采用 4 个动力脚轮（Powered Caster），实现全向运动。每个脚轮包含转向电机和驱动电机。

#### 3.1.1 运动学模型

设底座在全局坐标系下的位姿为 $\mathbf{x} = [x, y, \theta]^T$，关节空间速度为 $\dot{\mathbf{q}} = [\dot{q}_1^s, \dot{q}_1^d, \dot{q}_2^s, \dot{q}_2^d, \dot{q}_3^s, \dot{q}_3^d, \dot{q}_4^s, \dot{q}_4^d]^T$（$s$ 为转向，$d$ 为驱动），则操作空间速度与关节速度的关系为：

$$\dot{\mathbf{q}} = \mathbf{C}(\mathbf{q}) \cdot \dot{\mathbf{x}}_{local}$$

其中 $\mathbf{C}$ 矩阵的各元素由脚轮几何参数决定。

对于第 $i$ 个脚轮，设其转向角为 $q_i^s$，车辆中心到转向轴距离为 $(h_x^i, h_y^i)$，脚轮偏移为 $b_x, b_y$，车轮半径为 $r$：

**转向行：**

$$C_{2i-1,1} = \frac{\sin(q_i^s)}{b_x}, \quad C_{2i-1,2} = \frac{-\cos(q_i^s)}{b_x}, \quad C_{2i-1,3} = \frac{-h_x^i \cos(q_i^s) - h_y^i \sin(q_i^s)}{b_x} - 1$$

**驱动行：**

$$C_{2i,1} = \frac{\cos(q_i^s)}{r} - \frac{b_y \sin(q_i^s)}{b_x r}, \quad C_{2i,2} = \frac{\sin(q_i^s)}{r} + \frac{b_y \cos(q_i^s)}{b_x r}$$

$$C_{2i,3} = \frac{h_x^i \sin(q_i^s) - h_y^i \cos(q_i^s)}{r} + \frac{b_y(h_x^i \cos(q_i^s) + h_y^i \sin(q_i^s))}{b_x r}$$

#### 3.1.2 里程计

操作空间速度通过伪逆从关节速度恢复：

$$\dot{\mathbf{x}}_{local} = \mathbf{C}_{p}^{local} \cdot \dot{\mathbf{q}}$$

其中 $\mathbf{C}_{p}^{\dagger} = (\mathbf{C}_{p}^T \mathbf{C}_{p})^{-1} \mathbf{C}_{p}^T \mathbf{C}_{q}^{-1}$

全局坐标系下的速度通过旋转矩阵转换：

$$\dot{\mathbf{x}}_{global} = \mathbf{R}(\bar{\theta}) \cdot \dot{\mathbf{x}}_{local}$$

$$\mathbf{R}(\bar{\theta}) = \begin{bmatrix} \cos\bar{\theta} & -\sin\bar{\theta} & 0 \\ \sin\bar{\theta} & \cos\bar{\theta} & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

其中 $\bar{\theta} = \theta + \frac{1}{2}\dot{\theta}_{local} \cdot \Delta t$ 为中点积分角度。实测里程计精度：平移漂移 < 1cm/m，旋转漂移 < 1°/360°。

> **参考**：动力脚轮运动学建模源自 Holmberg & Khatib 的研究，详见 TidyBot++ 论文中的参考文献 [28]。

### 3.2 逆运动学求解器（IK Solver）

采用**阻尼最小二乘法（Damped Least Squares）**结合**零空间优化**的迭代 IK 求解器。

#### 3.2.1 误差计算

给定目标末端位姿 $(\mathbf{p}_{target}, \mathbf{q}_{target})$ 和当前末端位姿 $(\mathbf{p}_{curr}, \mathbf{q}_{curr})$：

**位置误差：**

$$\mathbf{e}_{pos} = \mathbf{p}_{target} - \mathbf{p}_{curr}$$

**姿态误差：** 通过四元数误差转换为角速度表示：

$$\mathbf{q}_{err} = \mathbf{q}_{target} \otimes \mathbf{q}_{curr}^{-1}$$

$$\mathbf{e}_{rot} = \text{quat2vel}(\mathbf{q}_{err})$$

总误差 $\mathbf{e} = [\mathbf{e}_{pos}^T, \mathbf{e}_{rot}^T]^T \in \mathbb{R}^6$

#### 3.2.2 关节更新

计算雅可比矩阵 $\mathbf{J} \in \mathbb{R}^{6 \times 7}$，关节更新量为：

$$\Delta\mathbf{q} = \mathbf{J}^T (\mathbf{J}\mathbf{J}^T + \lambda \mathbf{I})^{-1} \mathbf{e} + (\mathbf{I} - \mathbf{J}^T (\mathbf{J}\mathbf{J}^T + \lambda \mathbf{I})^{-1} \mathbf{J}) \cdot \mathbf{e}_{null}$$

其中：
- $\lambda = 10^{-12}$ 为阻尼系数
- $\mathbf{e}_{null} = \text{wrap}(\mathbf{q}_0 - \mathbf{q}_{curr})$ 为零空间误差，驱动关节趋向参考构型（Retract）
- 第二项为零空间投影，在不影响任务空间精度的前提下优化关节配置

单次最大角度变化限制为 $45°$，迭代最多 20 次，收敛阈值 $\|\mathbf{e}\| < 10^{-4}$。

> **参考**：阻尼最小二乘法（DLS / Levenberg-Marquardt）是机器人学中经典的 IK 求解方法，详见 [Siciliano et al., "Robotics: Modelling, Planning and Control"](https://link.springer.com/book/10.1007/978-1-84628-642-1)。零空间优化的思想源自冗余度_resolution_理论。

### 3.3 关节空间柔顺控制器

机械臂采用**关节空间柔顺控制器**，基于 [Cornell EmPRISE Lab](https://github.com/cornell-emprise/kinova-drivers) 的开源实现，实现力矩级控制与柔顺交互。

#### 3.3.1 控制律

**任务力矩：**

$$\boldsymbol{\tau}_{task} = -\mathbf{K}_p (\mathbf{q}_n - \mathbf{q}_d) - \mathbf{K}_d (\dot{\mathbf{q}}_n - \dot{\mathbf{q}}_d) + \mathbf{g}(\mathbf{q})$$

其中 $\mathbf{q}_n, \dot{\mathbf{q}}_n$ 为名义模型状态，$\mathbf{q}_d, \dot{\mathbf{q}}_d$ 为期望轨迹，$\mathbf{g}(\mathbf{q})$ 为重力补偿项（由 [Pinocchio](https://github.com/stack-of-tasks/pinocchio) 计算）。

**名义电机模型：**

$$\ddot{\mathbf{q}}_n = \mathbf{K}_r^{-1}(\boldsymbol{\tau}_{task} - \boldsymbol{\tau}_s^f)$$

$$\dot{\mathbf{q}}_n \leftarrow \dot{\mathbf{q}}_n + \ddot{\mathbf{q}}_n \cdot \Delta t, \quad \mathbf{q}_n \leftarrow \mathbf{q}_n + \dot{\mathbf{q}}_n \cdot \Delta t$$

**名义摩擦补偿：**

$$\boldsymbol{\tau}_f = \mathbf{K}_r \mathbf{K}_l \left[(\dot{\mathbf{q}}_n - \dot{\mathbf{q}}_s) + \mathbf{K}_{lp}(\mathbf{q}_n - \mathbf{q}_s)\right]$$

**最终力矩命令：**

$$\boldsymbol{\tau}_c = \boldsymbol{\tau}_{task} + \boldsymbol{\tau}_f$$

#### 3.3.2 增益参数

| 参数 | 前4关节 | 后3关节 |
|------|---------|---------|
| $\mathbf{K}_p$ | 100 | 50 |
| $\mathbf{K}_d$ | 3 | 2 |
| $\mathbf{K}_r$ | diag(0.3) | diag(0.18) |
| $\mathbf{K}_l$ | 75 | 40 |
| $\mathbf{K}_{lp}$ | 5 | 4 |

力矩传感器信号通过低通滤波（$\alpha = 0.01$）平滑处理。

> **参考**：柔顺控制器的理论基础源自阻抗控制（Impedance Control），详见 [Hogan, N., "Impedance Control: An Approach to Manipulation"](https://asmedigitalcollection.asme/dscc/article-abstract/107/1/1/422826)。摩擦补偿策略参考 Cornell EmPRISE Lab 的 [kinova-drivers](https://github.com/cornell-emprise/kinova-drivers) 项目。

### 3.4 在线轨迹生成（OTG）

底座和机械臂均使用 [**Ruckig**](https://ruckig.com/) 库进行在线轨迹生成，保证速度和加速度约束下的时间最优轨迹。

对于底座（3 DOF）：
- 最大速度：$[0.5, 0.5, 3.14]$ m/s, rad/s
- 最大加速度：$[0.5, 0.5, 2.36]$ m/s², rad/s²

对于机械臂（7 DOF）：
- 最大速度：前4关节 $80°/s$，后3关节 $140°/s$
- 最大加速度：前4关节 $240°/s²$，后3关节 $450°/s²$

> **参考**：Ruckig 是由 KIT（卡尔斯鲁厄理工学院）开发的下一代运动规划库，可在 250μs 内完成轨迹计算，支持加加速度（Jerk）约束。详见 [Berscheid et al., "Jerk-limited Real-time Trajectory Generation with Arbitrary Target States", RSS 2021](https://arxiv.org/abs/2105.04830)，GitHub: [pantor/ruckig](https://github.com/pantor/ruckig)。

### 3.5 Diffusion Policy（扩散策略）

策略训练使用 Columbia 大学的 [**Diffusion Policy**](https://diffusion-policy.cs.columbia.edu/) 框架，核心思想是将动作生成建模为去噪扩散过程。

#### 3.5.1 扩散模型原理

给定观测序列 $\mathbf{o}_{t-n_o+1:t}$，策略预测动作序列 $\mathbf{a}_{t:t+n_a-1}$。

**前向过程**（加噪）：对动作添加高斯噪声

$$\mathbf{a}^k = \sqrt{\bar{\alpha}_k}\mathbf{a}^0 + \sqrt{1-\bar{\alpha}_k}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

**反向过程**（去噪）：训练 U-Net $\boldsymbol{\epsilon}_\theta$ 预测噪声

$$\mathcal{L} = \mathbb{E}_{k, \mathbf{a}^0, \boldsymbol{\epsilon}} \left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}^k, k, \mathbf{o})\|^2\right]$$

**推理**：从随机噪声开始，经 $K$ 步去噪生成动作

$$\mathbf{a}^{k-1} = \frac{1}{\sqrt{\alpha_k}}\left(\mathbf{a}^k - \frac{1-\alpha_k}{\sqrt{1-\bar{\alpha}_k}}\boldsymbol{\epsilon}_\theta(\mathbf{a}^k, k, \mathbf{o})\right) + \sigma_k \mathbf{z}$$

Diffusion Policy 的核心优势：
- **多模态动作分布**：能够优雅地处理同一任务中的多种有效动作模式
- **高维动作空间**：可联合预测未来动作序列，保证时间一致性
- **训练稳定性**：通过学习能量函数的梯度，避免了隐式策略中负采样带来的训练不稳定

#### 3.5.2 网络架构

- **观测**：2帧图像（基座相机 84×84 + 腕部相机 84×84）+ 本体感知（base_pose 3D + arm_pos 3D + arm_quat 4D + gripper_pos 1D）
- **动作空间**：13维 = base_pose(3) + arm_pos(3) + arm_quat(6D旋转表示) + gripper_pos(1)
- **U-Net**：下采样维度 [256, 512, 1024]，扩散步嵌入维度 128
- **推理步数**：16步 DDPM
- **预测范围**：8步动作（n_action_steps=8），2步观测历史（n_obs_steps=2）

#### 3.5.3 延迟隐藏

`PolicyWrapper` 提前 200ms 发起下一次推理，确保当前动作序列耗尽时新序列已就绪，隐藏 115ms 的推理延迟。

$$\text{LATENCY\_STEPS} = \lceil \text{LATENCY\_BUDGET} / \text{POLICY\_CONTROL\_PERIOD} \rceil = \lceil 0.2 / 0.1 \rceil = 2$$

> **参考**：Diffusion Policy 由 Chi et al. 提出，发表于 RSS 2023，后续扩展发表于 IJRR 2024。详见 [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)，项目主页：[diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/)，GitHub: [columbia-ai-robotics/diffusion_policy](https://github.com/columbia-ai-robotics/diffusion_policy)。

### 3.6 WebXR 遥操作坐标变换

手机 [WebXR](https://www.w3.org/TR/webxr/) 坐标系（+x 右, +y 上, +z 后）到机器人坐标系（+x 前, +y 左, +z 上）的变换：

$$\mathbf{p}_{robot} = [-p_z^{xr}, -p_x^{xr}, p_y^{xr}]$$

$$\mathbf{q}_{robot} = [-q_z^{xr}, -q_x^{xr}, q_y^{xr}, q_w^{xr}]$$

并施加设备相机偏移补偿，使旋转围绕设备中心而非相机：

$$\mathbf{p}_{robot} \leftarrow \mathbf{p}_{robot} + \mathbf{R}_{robot} \cdot \boldsymbol{\delta}_{camera}$$

其中 $\boldsymbol{\delta}_{camera} = [0, 0.02, -0.04]$ 为 iPhone 14 Pro 的相机偏移。

> **参考**：WebXR 是 W3C 制定的用于在 Web 上访问 VR/AR 设备的 API 标准，详见 [WebXR Device API Specification](https://www.w3.org/TR/webxr/) 和 [MDN WebXR 文档](https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API)。

---

## 四、观测与动作空间

**观测空间：**

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `base_pose` | 3 | 底座全局位姿 $(x, y, \theta)$ |
| `arm_pos` | 3 | 末端执行器位置（底座局部坐标系） |
| `arm_quat` | 4 | 末端执行器姿态四元数 $(x, y, z, w)$ |
| `gripper_pos` | 1 | 夹爪开合度 $[0, 1]$ |
| `base_image` | 640×360×3 | 基座相机 RGB 图像 |
| `wrist_image` | 640×480×3 | 腕部相机 RGB 图像 |

**动作空间（13维）：**

| 动作项 | 维度 | 说明 |
|--------|------|------|
| `base_pose` | 3 | 目标底座位姿 $(x, y, \theta)$ |
| `arm_pos` | 3 | 目标末端位置 |
| `arm_quat` | 4→6 | 目标末端姿态（训练时用6D旋转表示） |
| `gripper_pos` | 1 | 目标夹爪开合度 |

---

## 五、MuJoCo 仿真案例

### 5.1 环境搭建

```bash
# 创建 conda 环境
mamba create -n tidybot2 python=3.10.14
mamba activate tidybot2
pip install -r requirements.txt
```

### 5.2 案例1：随机动作仿真

```python
import time
import numpy as np
from constants import POLICY_CONTROL_PERIOD
from mujoco_env import MujocoEnv

env = MujocoEnv(show_images=True)
env.reset()
try:
    for _ in range(100):
        action = {
            'base_pose': 0.1 * np.random.rand(3) - 0.05,
            'arm_pos': 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4]),
            'arm_quat': np.random.rand(4),
            'gripper_pos': np.random.rand(1),
        }
        env.step(action)
        obs = env.get_obs()
        print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
        time.sleep(POLICY_CONTROL_PERIOD)
finally:
    env.close()
```

### 5.3 案例2：示范数据回放

```bash
# 下载示例数据
mkdir data
wget -O data/sim-v1.tar.gz "https://www.dropbox.com/scl/fi/fsotfgbwg3m545jenj457/sim-v1.tar.gz?rlkey=lbkuq4fhg3pi1meci1kta41ny&dl=1"
tar -xf data/sim-v1.tar.gz -C data

# 回放示范（仿真环境）
python replay_episodes.py --sim --input-dir data/sim-v1

# 回放并显示图像
python replay_episodes.py --sim --input-dir data/sim-v1 --show-images

# 以观测作为动作执行
python replay_episodes.py --sim --input-dir data/sim-v1 --execute-obs
```

### 5.4 案例3：WebXR 遥操作姿态可视化

```python
import mujoco
import mujoco.viewer
import numpy as np
from policies import TeleopPolicy

policy = TeleopPolicy()
policy.reset()  # 等待手机端点击 "Start episode"

xml = """
<mujoco>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5"/>
  </asset>
  <worldbody>
    <light directional="true"/>
    <geom name="floor" size="0 0 .05" type="plane" material="groundplane"/>
    <body name="target" pos="0 0 .5" mocap="true">
      <geom type="box" size=".05 .05 .05" rgba=".6 .3 .3 .5"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mocap_id = m.body('target').mocapid[0]
with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    while viewer.is_running():
        mujoco.mj_step(m, d)
        obs = {
            'base_pose': np.zeros(3),
            'arm_pos': d.mocap_pos[mocap_id],
            'arm_quat': d.mocap_quat[mocap_id][[1, 2, 3, 0]],
            'gripper_pos': np.zeros(1),
        }
        action = policy.step(obs)
        if action == 'reset_env':
            break
        if isinstance(action, dict):
            d.mocap_pos[mocap_id] = action['arm_pos']
            d.mocap_quat[mocap_id] = action['arm_quat'][[3, 0, 1, 2]]
        viewer.sync()
```

### 5.5 策略训练流水线

#### Step 1：数据采集

```bash
python main.py --sim --teleop --save --output-dir data/demos
```

#### Step 2：数据格式转换

```bash
python convert_to_robomimic_hdf5.py --input-dir data/sim-v1 --output-path data/sim-v1.hdf5
```

转换过程中，四元数姿态被转换为轴角表示（Axis-Angle）：

$$\mathbf{a}_{action} = [\mathbf{p}_{base}, \mathbf{p}_{arm}, \text{quat2rotvec}(\mathbf{q}_{arm}), g_{gripper}] \in \mathbb{R}^{13}$$

> **参考**：数据格式采用 [robomimic](https://robomimic.github.io/) 的 HDF5 标准，robomimic 是由 NVIDIA 和斯坦福开发的离线模仿学习框架，详见 [Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation", CoRL 2021](https://arxiv.org/abs/2108.03298)，GitHub: [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic)。

#### Step 3：策略训练（GPU Laptop）

```bash
cd ~/diffusion_policy
mamba activate robodiff

# 应用兼容补丁
cp ~/tidybot2/diffusion-policy.patch ~/diffusion_policy/
git checkout 548a52b
git apply diffusion-policy.patch

# 修改任务配置文件中的 name 字段
# diffusion_policy/config/task/square_image_abs.yaml → name: sim-v1

# 启动训练
python train.py --config-name=train_diffusion_unet_real_hybrid_workspace
```

#### Step 4：策略推理

```bash
# GPU Laptop 上启动策略服务器
python policy_server.py --ckpt-path data/outputs/.../checkpoints/epoch=0500-train_loss=0.001.ckpt

# Dev Machine 上运行策略推理
python main.py --sim
```

---

## 六、MuJoCo 模型说明

### 6.1 机器人模型结构

仿真模型定义在 `models/stanford_tidybot/` 目录下：

- **`tidybot.xml`**：完整 TidyBot 模型（底座 + Kinova Gen3 + Robotiq 2F-85）
- **`scene.xml`**：仿真场景（含地面、立方体目标、天空盒）
- **`base.xml`**：仅底座模型

### 6.2 关键物理参数

**底座：**
- 质量：60 kg
- 3个关节：`joint_x`（滑动）、`joint_y`（滑动）、`joint_th`（旋转）
- 位置控制器增益：$k_p = 10^6, k_v = 5 \times 10^4$（x/y），$k_p = 5 \times 10^4, k_v = 10^3$（θ）

**Kinova Gen3 机械臂：**
- 7自由度，自重 8.2 kg，负载 4 kg，工作半径 902 mm
- 关节限位：joint_2 $\pm 2.24$ rad，joint_4 $\pm 2.57$ rad，joint_6 $\pm 2.09$ rad
- 大关节执行器：$k_p = 2000, k_v = 100$，力限制 $\pm 105$ N·m
- 小关节执行器：$k_p = 500, k_v = 50$，力限制 $\pm 52$ N·m
- 1kHz 闭环控制，全关节内置扭矩传感器

> **参考**：[Kinova Gen3 官方文档](https://www.kinovarobotics.com/product/gen3-robots)，[Kinova Kortex API](https://github.com/Kinovarobotics/kortex)

**Robotiq 2F-85 夹爪：**
- 驱动关节范围：$[0, 0.8]$
- 通过肌腱（tendon）实现双指同步
- 4连杆机构通过等式约束（equality constraint）建模

> **参考**：[Robotiq 2F-85 官方页面](https://robotiq.com/products/2f85-140-adaptive-robot-gripper)

**仿真目标物：**
- 立方体：$0.04 \times 0.04 \times 0.04$ m，质量 0.1 kg
- 初始位置随机化：$x, y \in [-0.1, 0.1]$，$\theta \in [-\pi, \pi]$

### 6.3 仿真环境特性

- 全局重力补偿：`body_gravcomp[:] = 1.0`（除目标物体外）
- 积分器：`implicitfast`（隐式快速积分器）
- 摩擦锥：`elliptic`，阻抗比 10
- 双相机：基座相机（640×360，FOV 52.2°），腕部相机（640×480，FOV 41.8°）
- 控制频率：10 Hz（策略层），物理仿真步长由 MuJoCo 自动确定

> **参考**：[MuJoCo](https://mujoco.org/)（Multi-Joint dynamics with Contact）是由 Google DeepMind 维护的开源物理引擎，最初由 Roboti LLC 开发，2021 年被 DeepMind 收购并免费开放，2022 年开源。详见 [Todorov et al., "MuJoCo: A physics engine for model-based control"](https://ieeexplore.ieee.org/document/6386109)，GitHub: [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)。

---

## 七、硬件规格与可扩展性

### 7.1 可更换机械臂

TidyBot++ 提供多种高度可定制的参考设计，可轻松修改以支持不同机械臂：

| 机械臂 | 厂商 |
|--------|------|
| Kinova Gen3 | Kinova |
| Franka | Franka Emika |
| ARX5 | ARX |
| xArm | UFACTORY |
| UR5 | Universal Robots |
| ViperX | Trossen Robotics |

### 7.2 负载能力

虽然设计初衷仅搭载一个机械臂，但实测底座可轻松应对更高负载：
- 配重板总计约 120 kg（270 lb）
- 五个机械臂总计约 90 kg（200 lb）
- 乘客约 70 kg（150 lb）

### 7.3 地形通过性

底座可在多种地面工作，从硬质地板到高绒地毯，并能跨越常见地面障碍：门槛、电梯缝隙、减速带、钢板、路缘坡道、装载坡道（6.5° 坡度）。

---

## 八、依赖环境

```
# 通用
numpy==1.26.4
scipy==1.12.0
opencv-python==4.9.0.80
h5py==3.11.0
tqdm==4.66.2

# 机器人
mujoco==3.2.4          # 物理仿真
pin==2.7.0             # 刚体动力学（重力补偿）
ruckig==0.12.2         # 在线轨迹生成

# 通信
flask==3.0.2           # WebXR 手机端 Web 服务
flask_socketio==5.3.6  # WebSocket 通信
pyzmq==25.1.2          # ZMQ 策略服务器通信
redis==5.0.6           # 状态可视化数据流

# 底座硬件
phoenix6               # CTRE 电机控制器
pygame==2.5.2          # 手柄遥操作
```

---

## 九、关键参考链接汇总

### 论文与项目

| 名称 | 链接 |
|------|------|
| TidyBot++ 论文 | [arXiv:2412.10447](https://arxiv.org/abs/2412.10447) |
| TidyBot++ 项目主页 | [tidybot2.github.io](https://tidybot2.github.io/) |
| TidyBot++ GitHub | [github.com/jimmyyhwu/tidybot2](https://github.com/jimmyyhwu/tidybot2/tree/main) |
| TidyBot++ 组装指南 | [tidybot2.github.io/docs](http://tidybot2.github.io/docs) |
| TidyBot++ 使用指南 | [tidybot2.github.io/docs/usage](http://tidybot2.github.io/docs/usage) |

### 核心算法与框架

| 名称 | 说明 | 链接 |
|------|------|------|
| Diffusion Policy | 扩散策略，动作生成的去噪扩散模型 | [论文](https://arxiv.org/abs/2303.04137) / [项目主页](https://diffusion-policy.cs.columbia.edu/) / [GitHub](https://github.com/columbia-ai-robotics/diffusion_policy) |
| MuJoCo | 多关节接触动力学物理引擎 | [官网](https://mujoco.org/) / [GitHub](https://github.com/google-deepmind/mujoco) / [文档](https://mujoco.readthedocs.io/) |
| Pinocchio | 刚体动力学算法及其解析导数 | [GitHub](https://github.com/stack-of-tasks/pinocchio) / [文档](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/) / [论文](https://hal-laas.archives-ouvertes.fr/hal-01866228) |
| Ruckig | 在线轨迹生成库，加加速度约束，时间最优 | [官网](https://ruckig.com/) / [GitHub](https://github.com/pantor/ruckig) / [论文 (RSS 2021)](https://arxiv.org/abs/2105.04830) / [文档](https://docs.ruckig.com/) |
| robomimic | 离线模仿学习框架与数据格式标准 | [官网](https://robomimic.github.io/) / [GitHub](https://github.com/ARISE-Initiative/robomimic) / [论文](https://arxiv.org/abs/2108.03298) |
| WebXR | W3C VR/AR 设备访问 API 标准 | [W3C 规范](https://www.w3.org/TR/webxr/) / [MDN 文档](https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API) |

### 硬件相关

| 名称 | 说明 | 链接 |
|------|------|------|
| Kinova Gen3 | 7自由度超轻量协作机械臂 | [官网](https://www.kinovarobotics.com/product/gen3-robots) / [Kortex API](https://github.com/Kinovarobotics/kortex) |
| Robotiq 2F-85 | 自适应两指夹爪 | [官网](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) |
| Cornell EmPRISE Kinova Drivers | 柔顺控制器开源实现 | [GitHub](https://github.com/cornell-emprise/kinova-drivers) |
| CTRE Phoenix 6 | 底座电机控制器 | [官网](https://pro.docs.ctr-electronics.com/) |

### 经典参考书籍与论文

| 名称 | 说明 | 链接 |
|------|------|------|
| Siciliano et al., "Robotics: Modelling, Planning and Control" | 机器人学经典教材，含 IK/DLS 理论 | [Springer](https://link.springer.com/book/10.1007/978-1-84628-642-1) |
| Hogan, N., "Impedance Control" | 阻抗控制理论基础 | [ASME](https://asmedigitalcollection.asme.org/dsc) |
| Featherstone, R., "Rigid Body Dynamics Algorithms" | 刚体动力学算法经典参考 | [Springer](https://link.springer.com/book/10.1007/978-0-387-74315-2) |
| DDPM 原始论文 | 去噪扩散概率模型 | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |

---

## 十、引用

```bibtex
@inproceedings{wu2024tidybot,
  title = {TidyBot++: An Open-Source Holonomic Mobile Manipulator for Robot Learning},
  author = {Wu, Jimmy and Chong, William and Holmberg, Robert and Prasad, Aaditya and Gao, Yihuai and Khatib, Oussama and Song, Shuran and Rusinkiewicz, Szymon and Bohg, Jeannette},
  booktitle = {Conference on Robot Learning},
  year = {2024}
}

@article{chi2023diffusion,
  title = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author = {Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  journal = {The International Journal of Robotics Research},
  year = {2024}
}

@inproceedings{berscheid2021ruckig,
  title = {Jerk-limited Real-time Trajectory Generation with Arbitrary Target States},
  author = {Berscheid, Lars and Kr{\"o}ger, Torsten},
  booktitle = {Robotics: Science and Systems},
  year = {2021}
}

@article{mandlekar2021robomimic,
  title = {What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author = {Mandlekar, Ajay and Xu, Soroush and Wong, Josiah and Nasiriany, Soroush and Wang, Chen and Kulkarni, Rohan and Fei-Fei, Li and Savarese, Silvio and Zhu, Yuke and Mart{\'i}n-Mart{\'i}n, Roberto},
  journal = {Conference on Robot Learning},
  year = {2021}
}
```
