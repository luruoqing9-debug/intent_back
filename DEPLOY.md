# 后端部署清单

## 一、服务器信息

| 项目 | 值 |
|---|---|
| SSH 地址 | `amax@pub.hsuni.top:63022` |
| 公网 IP | `157.254.154.230` |
| 工作目录 | `/data3/lrq/` |
| GPU | RTX 3090 |
| API 端口 | 9000（Flask）、8080（Hunyuan3D） |


---

## 二、服务器端操作

### 1. 安装系统依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git ffmpeg libgl1-mesa-glx libglib2.0-0

# 如果服务器有 GPU（NVIDIA）
# 安装 CUDA + cuDNN（根据 GPU 型号选择版本）
# 验证: nvidia-smi
```

### 2. 克隆代码

```bash
cd /data3/lrq
git clone <仓库地址> intentrelay_back
cd intentrelay_back
```

### 3. 创建虚拟环境 & 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 额外依赖（requirements.txt 可能未包含）
pip install flask flask-cors ultralytics opencv-python numpy pillow python-dotenv
```

### 4. 拷贝密钥和环境配置

从本地电脑执行（或手动在服务器创建 `.env`）：

```bash
# 本地电脑执行，上传到服务器
scp -P 63022 .env amax@pub.hsuni.top:/data3/lrq/intentrelay_back/.env
```

`.env` 需要包含：
```
GEMINI_API_KEY=你的Gemini密钥
XF_APP_ID=讯飞STT应用ID
XF_API_KEY=讯飞STT密钥
XF_API_SECRET=讯飞STT密钥
# ... 其他密钥
```

### 5. 拷贝模型文件

```bash
# 本地 -> 服务器（yolov8x-seg.pt，144MB，较慢）
scp -P 63022 yolov8x-seg.pt amax@pub.hsuni.top:/data3/lrq/intentrelay_back/yolov8x-seg.pt

# CLIP 缓存（如果有的话）
scp -P 63022 -r cache/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/cache/
```

### 6. 拷贝数据目录

```bash
# 记忆数据库
scp -P 63022 object_nodes.json amax@pub.hsuni.top:/data3/lrq/intentrelay_back/object_nodes.json

# 图片目录
scp -P 63022 -r processed_images/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/
scp -P 63022 -r generated_images/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/
scp -P 63022 -r original_image/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/
scp -P 63022 -r Component_three/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/
scp -P 63022 -r generated_3d_models/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/
scp -P 63022 -r 3d_image/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/
scp -P 63022 -r Feedback_image/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/
scp -P 63022 -r Operated_image/ amax@pub.hsuni.top:/data3/lrq/intentrelay_back/
```

### 7. 部署 Hunyuan3D 服务（端口 8080）

Hunyuan3D 是独立服务，需要单独部署：

```bash
# 参考 Hunyuan3D 官方文档安装
# 通常步骤：
git clone <Hunyuan3D仓库>
cd Hunyuan3D
# 创建独立虚拟环境（避免与 Flask 冲突）
python3 -m venv venv_3d
source venv_3d/bin/activate
pip install -r requirements.txt
# 启动服务
python app.py --port 8080
```

> **注意**：Hunyuan3D 也需要 GPU，且显存占用较大。如果服务器显存不够，
> 可能需要先后启动（先 Flask，用完再启 3D；或反过来）。

### 8. 启动服务

```bash
cd /opt/intentrelay_back
source venv/bin/activate

# 后台启动 Flask API（使用 nohup）
nohup python api.py > api.log 2>&1 &

# 或使用 screen/tmux（推荐，方便查看日志）
screen -S api
python api.py
# Ctrl+A, D 分离
```

### 9. 验证

```bash
# 检查 Flask 是否启动
curl http://localhost:9000/health

# 检查 Hunyuan3D 是否启动
curl http://localhost:8080/  # 根据实际接口调整

# 查看日志
tail -f api.log
```

### 10. 设置防火墙 / 安全组

在云服务器控制台开放端口：
- 9000/tcp（Flask API）
- 8080/tcp（Hunyuan3D）

### 11. 前端配置

将前端所有 API 请求的 base URL 从 `http://localhost:9000` 改为 `http://157.254.154.230:9000`

---

## 三、本地一键打包脚本（运行在本地电脑）

> 使用 SSH 别名 `3090`（已在 `~/.ssh/config` 配置）

```bash
#!/bin/bash
# run in: /Users/chenpei/Desktop/lrq/intentrelay_back/

SERVER="amax@pub.hsuni.top -P 63022"
REMOTE_DIR="/data3/lrq/intentrelay_back"

echo "=== 打包并上传到服务器 ==="

# 1. 创建远程目录
ssh -p 63022 amax@pub.hsuni.top "mkdir -p $REMOTE_DIR"

# 2. 上传 .env
echo "[1/8] 上传 .env..."
scp -P 63022 .env amax@pub.hsuni.top:$REMOTE_DIR/.env

# 3. 上传模型
echo "[2/8] 上传 yolov8x-seg.pt（较大，请耐心等待）..."
scp -P 63022 yolov8x-seg.pt amax@pub.hsuni.top:$REMOTE_DIR/yolov8x-seg.pt

# 4. 上传缓存
echo "[3/8] 上传 cache/..."
scp -P 63022 -r cache/ amax@pub.hsuni.top:$REMOTE_DIR/cache/

# 5. 上传数据
echo "[4/8] 上传记忆数据库..."
scp -P 63022 object_nodes.json amax@pub.hsuni.top:$REMOTE_DIR/object_nodes.json

echo "[5/8] 上传图片目录..."
for dir in processed_images generated_images original_image Component_three \
           generated_3d_models 3d_image Feedback_image Operated_image; do
    if [ -d "$dir" ]; then
        echo "  → $dir/"
        scp -P 63022 -r $dir/ amax@pub.hsuni.top:$REMOTE_DIR/$dir/
    fi
done

echo "=== 上传完成 ==="
echo ""
echo "接下来在服务器上执行："
echo "  ssh -p 63022 amax@pub.hsuni.top"
echo "  cd /data3/lrq/intentrelay_back"
echo "  git clone 代码（如果还没 clone）"
echo "  创建 venv 并 pip install -r requirements.txt"
echo "  部署 Hunyuan3D 服务"
echo "  python api.py 启动"
echo "  4. python api.py 启动"
```

---

## 四、常见问题

| 问题 | 解决 |
|---|---|
| `ModuleNotFoundError` | 确保 `source venv/bin/activate` 后再 `pip install` |
| YOLO 模型加载慢/失败 | 检查 GPU 是否可用，`python -c "import torch; print(torch.cuda.is_available())"` |
| 端口被占用 | `lsof -i :9000` 查看占用进程，`kill -9 <PID>` |
| `.env` 没生效 | 确保 `.env` 在 `api.py` 同级目录 |
| Hunyuan3D 显存不足 | 降低分辨率或分批运行 |
| 图片 404 | 检查目录权限 `chmod -R 755 processed_images/` 等 |

---

## 五、文件清单（完整项目结构）

```
intentrelay_back/
├── api.py                    # Flask 入口（端口 9000）
├── main.py                   # 核心业务逻辑
├── speech.py                 # 语音识别（讯飞 STT）
├── record.py                 # 录音/语音处理
├── Memory.py                 # 记忆数据库
├── Feedback.py               # AI 反馈生成
├── generate.py               # 提示词生成
├── Generate_image.py         # ComfyUI 图像生成
├── generate_3d.py            # 3D 模型生成调度
├── viewpoint.py              # 视点检测
├── lapi_client.py            # LLM API 客户端
├── requirements.txt          # Python 依赖
├── .env                      # 环境变量（需手动拷贝）
│
├── yolov8x-seg.pt            # YOLO 模型（需手动拷贝）
├── cache/                    # CLIP 缓存（需手动拷贝）
├── object_nodes.json         # 记忆数据库（需手动拷贝）
│
├── original_image/           # 参考图（需手动拷贝）
├── processed_images/         # 已处理图片（需手动拷贝）
├── generated_images/         # 生成的图片（需手动拷贝）
├── Component_three/          # 候选图（需手动拷贝）
├── generated_3d_models/      # 3D 模型（需手动拷贝）
├── 3d_image/                 # 3D 素材图（需手动拷贝）
├── Feedback_image/           # 反馈图片（需手动拷贝）
├── Operated_image/           # 操作记录图
│
├── Component_generation.json # ComfyUI 工作流
├── Text-to-Image.json        # ComfyUI 工作流
├── google_Gemini_image.json  # ComfyUI 工作流
│
└── (其他测试/调试文件，不需要部署)
    ├── interactive_test.py
    ├── debug_glb.py
    ├── app.py
    ├── trigger.py
    ├── detection_log.json
    ├── frame_process.py
    ├── test_external_input.py
    └── server.py
```
