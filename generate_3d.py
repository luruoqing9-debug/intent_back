"""
generate_3d.py - 3D 模型生成中转调度器

调用流程：
1. 翻译提示词
2. 从 3d_image/ 获取 2D 素材图
3. POST http://<server>:8080/generate → 传入 base64 图片
4. 直接接收 .glb 二进制响应
5. 用 Blender 将 .glb 转为 .usdz

文件夹说明：
├── 3d_image/              → 3D 生成素材图（始终只有一张，上传新图时自动替换旧的）
├── generated_3d_models/   → 已生成的 3D 模型文件（永久保留）
"""

import os
import json
import shutil
import base64
import tempfile
import subprocess
import requests
from typing import Optional

# ==================== 配置 ====================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

THREE_D_IMAGE_DIR = os.path.join(PROJECT_DIR, "3d_image")
os.makedirs(THREE_D_IMAGE_DIR, exist_ok=True)

MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "generated_3d_models")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

HUNYUAN3D_HOST = "192.168.199.27"
HUNYUAN3D_PORT = 8080

# Blender 路径（macOS）
BLENDER_PATH = "/data3/lrq/blender-4.2.6-linux-x64/blender"
if not os.path.exists(BLENDER_PATH):
    BLENDER_PATH = "/data3/lrq/blender-4.2.6-linux-x64/blender"

MODEL_URL_PREFIX = "http://refinity-intentrelay.pub.hsuni.top:15902/models"
THREE_D_IMAGE_URL_PREFIX = "http://refinity-intentrelay.pub.hsuni.top:15902/3d_image"


def get_3d_image() -> Optional[str]:
    """从 3d_image/ 获取唯一的素材图片"""
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    if not os.path.exists(THREE_D_IMAGE_DIR):
        os.makedirs(THREE_D_IMAGE_DIR)
        return None
    for f in os.listdir(THREE_D_IMAGE_DIR):
        if os.path.splitext(f)[1].lower() in supported_extensions:
            return os.path.join(THREE_D_IMAGE_DIR, f)
    return None


def update_3d_image(new_image_path: str) -> str:
    """更新 3d_image/ 中的图片（删除旧的，放入新的）"""
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    os.makedirs(THREE_D_IMAGE_DIR, exist_ok=True)
    for f in os.listdir(THREE_D_IMAGE_DIR):
        if os.path.splitext(f)[1].lower() in supported_extensions:
            try:
                os.remove(os.path.join(THREE_D_IMAGE_DIR, f))
            except Exception:
                pass
    new_filename = os.path.basename(new_image_path)
    final_path = os.path.join(THREE_D_IMAGE_DIR, new_filename)
    shutil.copy(new_image_path, final_path)
    print(f"[3d_image] Added new image: {final_path}")
    return final_path


def _convert_with_blender(glb_path: str, usdz_path: str) -> Optional[str]:
    """使用 Blender 转换 GLB → USDZ"""
    if not BLENDER_PATH:
        print("[3D] Blender 未安装或未配置")
        return None

    blender_script = f"""
import bpy

# 启用 USD 插件
try:
    if 'io_scene_usd' not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module="io_scene_usd")
except Exception:
    pass

# 清理场景
bpy.ops.wm.read_factory_settings(use_empty=True)

# 导入 GLB
bpy.ops.import_scene.gltf(filepath="{glb_path}")

# 导出 USDZ
bpy.ops.wm.usd_export(
    filepath="{usdz_path}", 
    check_existing=False,
    export_textures=True,
    selected_objects_only=False
)
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(blender_script)
        script_path = tmp_file.name

    try:
        cmd = [BLENDER_PATH, "-b", "-P", script_path]
        print(f"[3D] 调用 Blender 转换: {glb_path}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if os.path.exists(usdz_path):
            size = os.path.getsize(usdz_path)
            print(f"[3D] Blender USDZ 转换成功: {usdz_path} ({size} bytes)")
            return usdz_path
        else:
            print(f"[3D] Blender 转换失败:\n{result.stderr}")
            return None
    except Exception as e:
        print(f"[3D] Blender 系统错误: {e}")
        return None
    finally:
        if os.path.exists(script_path):
            try:
                os.remove(script_path)
            except Exception:
                pass


def convert_to_usdz(glb_path: str) -> Optional[str]:
    """将 .glb 转为 .usdz"""
    usdz_path = glb_path.replace('.glb', '.usdz')

    # 方法1: Blender（保留完整贴图和材质）
    result = _convert_with_blender(glb_path, usdz_path)
    if result:
        return result

    # 方法2: 无可用转换方式，返回 None
    print(f"[3D] 所有 USDZ 转换方式均不可用，将使用 .glb")
    return None


def generate_3d_model(model_name: Optional[str] = None) -> dict:
    """
    生成 3D 模型并转换为 USDZ（纯 image-to-3D 模式）。

    Args:
        model_name: 输出模型文件名（不含扩展名）

    Returns:
        {
            "success": True,
            "model_id": "xxx.usdz",
            "model_url": "http://8.211.160.130:9000/models/xxx.usdz",
            "model_path": "/local/path/to/xxx.usdz"
        }
    """
    # 1. 获取素材图
    image_path = get_3d_image()
    if not image_path:
        return {
            "success": False,
            "error": "3d_image/ 中没有素材图片，请先上传或放入 2D 图"
        }
    print(f"[3D] 使用素材图: {image_path}")

    # 2. 准备输出路径
    if not model_name:
        model_name = os.path.splitext(os.path.basename(image_path))[0]

    glb_path = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}.glb")
    usdz_path = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}.usdz")

    # 3. 调用 Hunyuan3D API 生成 GLB
    try:
        result_path = _hunyuan3d_generate(
            image_path=image_path,
            output_path=glb_path
        )

        # 4. 尝试转为 .usdz
        usdz_result = None
        if result_path:
            usdz_result = convert_to_usdz(result_path)

        # 优先返回 .usdz，否则用 .glb
        if usdz_result and os.path.exists(usdz_result):
            final_path = usdz_result
        else:
            final_path = result_path

        model_id = os.path.basename(final_path)
        return {
            "success": True,
            "model_id": model_id,
            "model_url": f"{MODEL_URL_PREFIX}/{model_id}",
            "model_path": final_path
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def _hunyuan3d_generate(
    image_path: str,
    output_path: str
) -> Optional[str]:
    """
    调用 Hunyuan3D 服务器生成 3D 模型（纯 image-to-3D）。

    POST http://<host>:8080/generate → 直接返回 .glb 二进制内容
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    base_url = f"http://{HUNYUAN3D_HOST}:{HUNYUAN3D_PORT}"

    payload = {
        "image": image_b64
    }

    print(f"[Hunyuan3D] POST {base_url}/generate ...")
    response = requests.post(
        f"{base_url}/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=1200
    )
    response.raise_for_status()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"[Hunyuan3D] GLB 模型已保存: {output_path}")
    return output_path
