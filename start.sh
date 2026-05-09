#!/bin/bash
# 1. 清理旧进程（杀掉占用 9000 端口的进程）
lsof -ti :9000 | xargs kill 2>/dev/null
sleep 2

# 2. 清掉旧的代理设置
unset http_proxy https_proxy no_proxy

# 3. 设置代理（Google API 需要走代理）
#    - 本地 mihomo (127.0.0.1:7897) 赛博云节点连 Google 超时
#    - 10.195.25.233:7897 测试能通 Google API
export http_proxy=http://10.195.25.233:7897
export https_proxy=http://10.195.25.233:7897
# 旧配置（本地 mihomo）：http://127.0.0.1:7897

# 4. 以下地址不走代理：
#    - localhost/127.0.0.1: 本地回环
#    - 157.254.154.230: 服务器公网 IP
#    - 192.168.199.27: 5090 机器（3D 模型），防止请求被代理拦截
#    - 192.168.199.26: 服务器内网 IP
#    - hf-mirror.com: HuggingFace 镜像
export no_proxy="localhost,127.0.0.1,157.254.154.230,192.168.199.27,192.168.199.26,hf-mirror.com"

# CLIP 模型已缓存到本地，离线模式避免每次启动都尝试联网检查更新
export HF_HUB_OFFLINE=1

# 5. 启动 Flask
cd /data3/lrq/intentrelay_back
source venv/bin/activate
conda deactivate 2>/dev/null
python api.py
