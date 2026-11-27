<h1 align="center"><img src="img/logo.png" width="40"> Open-TeleVision：RTX 50 系列仿真复现版</h1>

> **声明：本仓库不是官方仓库。** 本仓库基于 [OpenTeleVision/TeleVision](https://github.com/OpenTeleVision/TeleVision)（CoRL 2024）修改，用于在 RTX 50 系列显卡（Blackwell / sm_120）上复现仿真侧的模仿学习流程。原始代码、模型设计和数据集的全部功劳归原作者所有。使用本仓库的研究成果时，请引用原论文（见文末[引用](#引用)）。
>
> 完整的复现过程、踩坑记录和仿真效果见 [docs/reproduction-guide.md](docs/reproduction-guide.md)。

## 许可证与致谢

本仓库沿用原仓库的 **Apache License 2.0**，完整条款见 [`LICENSE`](LICENSE)。

- **原作者版权**：Copyright 2024 Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang and Xiaolong Wang。
- **原项目依赖的开源代码**（各自许可证请见对应仓库）：
  1. [tonyzhaozh/act](https://github.com/tonyzhaozh/act)
  2. [facebookresearch/detr](https://github.com/facebookresearch/detr)
  3. [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting)
  4. [vuer-ai/vuer](https://github.com/vuer-ai/vuer)
- **本版本额外依赖**：[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)（Apache License 2.0）。本仓库不包含它的代码，需要按[准备 DINOv2](#准备-dinov2必需)单独下载。
- **数据集**：示例数据集由原作者提供（见[训练指南](#训练指南)中的下载链接），不随本仓库分发。

按照 Apache License 2.0 第 4 条的要求，本仓库保留了原 `LICENSE` 文件和全部版权声明，修改过的文件列在下一节。

## 相对原版的修改

本版本只改动了仿真侧的模仿学习流程（数据预处理 → ACT 训练 → JIT 导出 → 仿真推理可视化）。真机遥操作相关代码（`teleop/`）**没有改动，也没有验证**。

| 文件 | 改动 | 原因 |
|------|------|------|
| `act/detr/models/backbone.py` | 从本地 `dinov2` 目录加载 DINOv2，不再从 GitHub 拉取 main 分支 | 新版 DINOv2 的 hub 代码使用了 Python 3.10 语法，与旧环境不兼容 |
| `act/utils.py` | DataLoader 的 `num_workers` 从 24/16 改为 4/2 | 避免 16GB 内存的笔记本内存不足 |
| `act/imitate_episodes.py` | `dataset_stats.pkl` 改为保存 Python list；checkpoint 用 `map_location='cpu'` 加载；JIT 改在 CPU 上 trace | 让 `tv` 环境（numpy 1.x、旧版 PyTorch）能读取 `tv_train` 环境的产物 |
| `scripts/deploy_sim.py` | 策略推理改在 CPU 上运行；加载 stats 时转换为 numpy 数组；新增 `--record`；逐帧读取图像，时间聚合改用滚动缓冲 | `tv` 环境中的旧版 PyTorch 无法在 sm_120 上执行 CUDA kernel；便于保存仿真效果；内存占用从约 15GB 降到约 4GB |
| `scripts/replay_demo.py` | `Player` 支持用离屏相机录制 mp4 | 为 `--record` 提供实现 |
| `.gitignore` | 忽略 `data/`；对 `docs/images/`、`docs/videos/` 取消图片和视频的忽略规则 | 数据集和 checkpoint 体积超过 100GB，不应提交；文档中的截图和演示视频需要提交 |
| `README.md` | 改写为中文，补充本版本说明 | — |
| `docs/`（新增） | 复现记录文章、截图、演示 GIF 和完整仿真视频 | 记录复现过程和仿真效果 |

## 原项目简介

<p align="center">
    <a href="https://chengxuxin.github.io/"><strong>Xuxin Cheng*</strong></a>
    ·
    <a href=""><strong>Jialong Li*</strong></a>
    ·
    <a href="https://aaronyang1223.github.io/"><strong>Shiqi Yang</strong></a>
    <br>
    <a href="https://www.episodeyang.com/"><strong>Ge Yang</strong></a>
    ·
    <a href="https://xiaolonw.github.io/"><strong>Xiaolong Wang</strong></a>
</p>

<p align="center">
    <img src="img/UCSanDiegoLogo-BlueGold.png" height="50"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
    <img src="img/mit-logo.png" height="50">
</p>

<h3 align="center"> CoRL 2024 </h3>

<p align="center">
<h3 align="center"><a href="https://robot-tv.github.io/">项目主页</a> | <a href="https://arxiv.org/abs/2407.01512/">arXiv</a> | <a href="https://github.com/OpenTeleVision/TeleVision">官方仓库</a></h3>
</p>

<p align="center">
<img src="./img/main.webp" width="80%"/>
</p>

论文标题：*Open-TeleVision: Teleoperation with Immersive Active Visual Feedback*。原仓库包含 Open-TeleVision 的遥操作和模仿学习实现：操作员佩戴 Apple Vision Pro 或 Meta Quest 3，通过沉浸式立体视觉遥操作 Unitree H1 人形机器人（配 Inspire 灵巧手）采集示范数据，再用 ACT 训练模仿学习策略。

## 本版本使用说明

### 为什么需要两个环境

- RTX 50 系列需要 PyTorch ≥ 2.7（CUDA 12.8），而这些版本没有 Python 3.8 的安装包。
- Isaac Gym Preview 4 只支持 Python ≤ 3.8。
- 最新版 DINOv2 的 hub 代码使用了 Python 3.10 语法（`float | None`）。

因此本版本把训练和仿真拆到两个 conda 环境：

| 环境 | Python | PyTorch | 用途 |
|------|--------|---------|------|
| `tv` | 3.8 | 2.3（官方 `requirements.txt`） | `post_process.py`（ZED）、`replay_demo.py`、`deploy_sim.py`（Isaac Gym） |
| `tv_train` | 3.11 | 2.7+ / cu128 | `imitate_episodes.py` 训练与 JIT 导出 |

> 如果你的显卡是 RTX 40 系列或更早，可以继续用官方的单环境方案；也可以把 `deploy_sim.py` 中的 `device = "cpu"` 改回 `"cuda"`，加快推理。

### 准备 DINOv2（必需）

`backbone.py` 会从**与 `TeleVision` 同级的 `dinov2` 目录**加载模型，需要先手动准备：

```bash
# 在 TeleVision 的上一级目录执行
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2 && git checkout b48308a
```

目录结构应为：

```text
parent_dir/
├── TeleVision/
└── dinov2/        # 固定在 b48308a
```

首次运行时，`dinov2_vits14` 的预训练权重会自动下载到 `~/.cache/torch/hub/checkpoints/`。

### 安装

**1. 仿真 / 预处理环境 `tv`**

```bash
conda create -n tv python=3.8
conda activate tv
pip install -r requirements.txt
cd act/detr && pip install -e .
# Isaac Gym Preview 4：https://developer.nvidia.com/isaac-gym/
cd /path/to/isaacgym/python && pip install -e .
# ZED SDK：https://www.stereolabs.com/developers/release/
# ZED Python API（post_process.py 读取 .svo 需要）
cd /usr/local/zed/ && python get_python_api.py
```

**2. 训练环境 `tv_train`**

```bash
conda create -n tv_train python=3.11
conda activate tv_train
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install 'numpy<2.1' einops h5py opencv-python matplotlib tqdm PyYAML scikit-learn IPython wandb packaging
cd act/detr && pip install -e .
```

检查显卡是否可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

输出的架构列表中应包含 `sm_120`。

### 运行流程（以 `00-can-sorting` 为例）

先按[训练指南](#训练指南)下载数据集，放到 `data/recordings/` 下。

```bash
# 1. 预处理（tv）
conda activate tv
cd scripts && python post_process.py

# 2. 训练（tv_train）
conda activate tv_train
cd ../act
python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 \
  --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 30000 \
  --lr 5e-5 --seed 0 --taskid 00 --exptid 01 --no_wandb

# 3. 导出 JIT（tv_train，用验证集 loss 最低的 checkpoint 编号）
python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 \
  --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 30000 \
  --lr 5e-5 --seed 0 --taskid 00 --exptid 01 --no_wandb \
  --save_jit --resume_ckpt 17000

# 4. 仿真推理可视化（tv）
conda activate tv
cd ../scripts
python deploy_sim.py --taskid 00 --exptid 01 --resume_ckpt 17000
# 加 --record 保存视频：data/logs/<task>/<exptid>/deploy_sim_<ckpt>.mp4
# 画面从左到右依次是仿真机器人、数据集左眼、数据集右眼，60fps（与录制频率一致，实时播放）
python deploy_sim.py --taskid 00 --exptid 01 --resume_ckpt 17000 --record
```

说明：

- `--batch_size 8` 和 `--num_epochs 30000` 是根据笔记本资源调整的，官方配置是 45 / 50000。代码里的 `num_epochs` 实际表示优化步数，每步一个 batch。
- 训练结束时会打印验证集 loss 最低的 epoch，并保存为 `policy_epoch_<best>_seed_0.ckpt` 和 `policy_best.ckpt`。导出时应选这个编号，而不是照搬官方示例中的 `25000`。
- 每 1000 步保存一个约 360MB 的 checkpoint，30000 步约需 11GB 磁盘空间。
- 加了 `--no_wandb` 后不会保存 loss 曲线。如果需要事后分析训练过程，可以去掉这个参数，或者自行把 loss 写入文件。

### 参考结果（RTX 5060 Laptop，16GB 内存）

| 项目 | 数值 |
|------|------|
| 训练速度 | 约 2.7 it/s（batch_size 8） |
| 30000 步耗时 | 约 3 小时 |
| 验证集最优 epoch | 17000 |
| 验证集 loss | 约 0.119（L1 约 0.117） |
| 仿真推理（CPU） | 约 3 帧/秒，运行时内存约 4GB |

### 已知限制

- `deploy_sim.py` 是**开环可视化**：输入是数据集中录制好的图像和状态，Isaac Gym 场景里只有机器人和地面。它适合用来检查预测动作是否合理，不能代表任务成功率。
- 策略在 CPU 上推理，窗口播放明显慢于实时；用 `--record` 录制的视频按 60fps 写入，播放速度正常。
- 官方 `deploy_sim.py` 会把整段图像（约 9GB）和一个约 6GB 的时间聚合数组一次性载入内存，16GB 的机器容易崩溃。本版本改为逐帧读取图像，并用只保留最近 60 个预测的滚动缓冲做时间聚合（结果与原实现一致）。

---

以下为原仓库文档的中文翻译，内容与官方一致（官方使用单个 `tv` 环境）。在 RTX 50 系列上请以上面的[本版本使用说明](#本版本使用说明)为准。

## 安装

```bash
conda create -n tv python=3.8
conda activate tv
pip install -r requirements.txt
cd act/detr && pip install -e .
```

安装 ZED SDK：https://www.stereolabs.com/developers/release/

安装 ZED Python API：

```bash
cd /usr/local/zed/ && python get_python_api.py
```

如果想在仿真环境中试用遥操作示例（`teleop_hand.py`），还需安装 Isaac Gym：https://developer.nvidia.com/isaac-gym/

## 遥操作指南

### 局域网串流

**Quest** 的局域网串流请参考[这个 issue](https://github.com/OpenTeleVision/TeleVision/issues/12#issue-2401541144)。

**Apple** 不允许在非 https 连接上使用 WebXR。要在本地测试，需要创建自签名证书并安装到客户端。你需要一台 Ubuntu 电脑和一个路由器，把 Vision Pro 和 Ubuntu 电脑连到同一个路由器。

1. 安装 mkcert：https://github.com/FiloSottile/mkcert
2. 查看本机 IP 地址：

   ```bash
   ifconfig | grep inet
   ```

   假设 Ubuntu 电脑的局域网 IP 是 `192.168.8.102`。

3. 创建证书：

   ```bash
   mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.8.102 localhost 127.0.0.1
   ```

   把生成的 `cert.pem` 和 `key.pem` 放到 `teleop` 目录下。

4. 在服务器上开放防火墙端口：

   ```bash
   sudo iptables -A INPUT -p tcp --dport 8012 -j ACCEPT
   sudo iptables-save
   sudo iptables -L
   ```

   或者使用 `ufw`：

   ```bash
   sudo ufw allow 8012
   ```

5. 创建 `OpenTeleVision` 时设置 `ngrok=False`：

   ```python
   tv = OpenTeleVision(self.resolution_cropped, shm.name, image_queue, toggle_streaming, ngrok=False)
   ```

6. 在 Vision Pro 上安装 CA 证书。先查看根证书位置：

   ```bash
   mkcert -CAROOT
   ```

   通过 AirDrop 把 `rootCA.pem` 发送到 Vision Pro 并安装。

   进入“设置 > 通用 > 关于本机 > 证书信任设置”，在“针对根证书启用完全信任”下打开该证书的信任。

   进入“设置 > App > Safari 浏览器 > 高级 > 功能标志”，启用 WebXR 相关功能。

7. 在 Vision Pro 的 Safari 中打开 `https://192.168.8.102:8012?ws=wss://192.168.8.102:8012`。

8. 点击 `Enter VR`，再点击 `Allow`，进入 VR 会话。

### 公网串流

在 Meta Quest 3 上安装证书比较麻烦，可以改用公网串流方案：用 `ngrok` 给服务器建立安全隧道。这种方法同时适用于 Vision Pro 和 Meta Quest 3。

1. 安装 ngrok：https://ngrok.com/download
2. 运行 ngrok：

   ```bash
   ngrok http 8012
   ```

3. 复制生成的 https 地址，在 Meta Quest 3 的浏览器中打开。

注意：使用 ngrok 公网串流时，创建 `OpenTeleVision` 需要设置 `ngrok=True`：

```python
self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True)
```

### 仿真遥操作示例

1. 按上面任一方式配置好串流后，可以在 Isaac Gym 中遥操作两只机器人手：

   ```bash
   cd teleop && python teleop_hand.py
   ```

2. 在 Vision Pro 上打开 vuer 页面，点击 `Enter VR` 和 `Allow` 进入沉浸式环境。

3. 你会在 3D 场景中看到自己的双手。

<img src=img/sim.png>

## 训练指南

1. 从 [Google Drive](https://drive.google.com/drive/folders/11WO96mUMjmxRo9Hpvm4ADz7THuuGNEMY?usp=sharing) 下载数据集。

2. 把下载的数据集放到 `data/recordings/` 下。

3. 用 `scripts/post_process.py` 处理指定的数据集，生成训练数据。

4. 可以用 `scripts/replay_demo.py` 检查某个 episode 的图像和动作序列。

5. 训练 ACT（官方参数）：

   ```bash
   python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt
   ```

6. 训练完成后，为选定的 checkpoint 导出 JIT：

   ```bash
   python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt \
                              --save_jit --resume_ckpt 25000
   ```

7. 用 `scripts/deploy_sim.py` 以数据集中的输入可视化训练好的策略，例如：

   ```bash
   python deploy_sim.py --taskid 00 --exptid 01 --resume_ckpt 25000
   ```

## 引用

如果本仓库对你的研究有帮助，请引用原论文：

```bibtex
@article{cheng2024tv,
title={Open-TeleVision: Teleoperation with Immersive Active Visual Feedback},
author={Cheng, Xuxin and Li, Jialong and Yang, Shiqi and Yang, Ge and Wang, Xiaolong},
journal={arXiv preprint arXiv:2407.01512},
year={2024}
}
```
