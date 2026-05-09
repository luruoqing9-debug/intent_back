import json
import base64
import random
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory

from ProtoFusionAPI import Comfyui_api, Tencentcloud_api, Hunyuan3d_api
from utils import ResponseWrapper, BusinessException, BUSINESS_FAIL, Img2Img, process_image

# 配置导入
from config import (
    OUTPUT_DIR, SOURCE_IMAGE_DIR, STYLE_IMAGE_DIR, WORKFLOW_FILE,
    ENDPOINTS, DESIGNEDIT_ENDPOINT, HUNYUAN3D_HOST, HUNYUAN3D_PORT, PORT, FLASK_PORT
)

def encode_image(image_path):
    """将图片编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_base64_image(base64_str, save_path):
    """保存base64图片到指定路径"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as file:
        file.write(image_data)

def create_app():
    """创建和配置Flask应用"""
    app = Flask(__name__)
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        """健康检查端点"""
        return "OK"
    
    @app.route('/<path:filename>')
    def uploaded_file(filename):
        """提供生成的文件"""
        return send_from_directory(OUTPUT_DIR, filename)
    
    @app.route('/GenerateCandidateEntireImage', methods=['POST'])
    def generate_candidate_entire_image():
        """生成候选完整图像"""
        try:
            data = request.json
            user_token = str(data.get('user_token'))
            test_prompt = str(data.get('prompt'))
            denoising_strength = float(data.get("denoising_strength"))
            style_ids = data.get('styleId')
            base64_image = data.get('image')
            
            # 保存源图像
            prompt_safe = test_prompt.replace(" ", "_")
            source_image_path = SOURCE_IMAGE_DIR / f"{user_token}_{test_prompt}.png"
            save_base64_image(base64_image, source_image_path)
            
            # 处理图像
            process_image(str(source_image_path))
            
            # 生成风格化图像
            images_list = []
            for style_id in style_ids:
                seed = random.randint(100000, 9999999)
                style_image_path = STYLE_IMAGE_DIR / f"style{style_id}.jpg"
                
                Comfyui_api.generate_image(
                    test_prompt, denoising_strength, seed,
                    str(source_image_path), str(style_image_path),
                    user_token, style_id, WORKFLOW_FILE
                )
                
                image_id = f"{user_token}_{prompt_safe}_{style_id}.png"
                url = f"{ENDPOINTS}{PORT}/{image_id}"
                
                image = Img2Img(url, image_id, style_id)
                images_list.append(image.to_dict())
            
            return jsonify(images=images_list)
            
        except BusinessException as be:
            return jsonify(ResponseWrapper.fail(code=be.code, message=be.message))
        except Exception as e:
            return jsonify(ResponseWrapper.fail(code=BUSINESS_FAIL, message=str(e)))
    
    @app.route('/GenerateComponentImage', methods=['POST'])
    def generate_component_image():
        """生成组件图像"""
        try:
            data = request.json
            user_token = str(data.get('user_token'))
            image_id = str(data.get('imageId'))
            bounding_box_data = data.get('boundingBox')
            component_count = int(data.get('componentCount'))
            
            # 处理图像路径
            image_path = OUTPUT_DIR / image_id
            
            # 处理边界框数据
            bounding_box = [
                [[item["coordinate"][0], item["coordinate"][1]], 
                 [item["coordinate"][2], item["coordinate"][3]]]
                for item in bounding_box_data
            ]
            
            # 准备请求数据
            base_image_name = Path(image_id).stem
            request_data = {
                "user_token": user_token,
                "original_image": encode_image(str(image_path)),
                "componentCount": component_count,
                "boundingBox": bounding_box,
                "imageId": base_image_name
            }
            
            # 发送到DesignEdit服务
            response = requests.post(DESIGNEDIT_ENDPOINT, json=request_data)
            return jsonify(response.json())
            
        except BusinessException as be:
            return jsonify(ResponseWrapper.fail(code=be.code, message=be.message))
        except Exception as e:
            return jsonify(ResponseWrapper.fail(code=BUSINESS_FAIL, message=str(e)))
    
    @app.route('/GenerateComponent3DModel', methods=['POST'])
    def generate_component_3d_model():
        """生成组件3D模型"""
        try:
            data = request.json
            user_token = str(data.get('user_token'))
            component_order = str(data.get('componentOrder'))
            prompt = str(data.get('prompt'))
            image_id = str(data.get('imageId'))
            
            # 翻译提示词
            translation_response = Tencentcloud_api.TextTranslate(prompt)
            translated_prompt = json.loads(
                str(translation_response.read(), 'utf-8')
            ).get('Response').get("TargetText")
            
            # 处理图像
            image_path = OUTPUT_DIR / image_id
            if not image_path.exists():
                raise BusinessException(BUSINESS_FAIL, f"Image not found: {image_id}")
            
            # 准备输出路径
            base_image_name = Path(image_id).stem
            result_path = OUTPUT_DIR / base_image_name

            # 最终期望的文件名 (用于返回给前端)
            model_filename = f"{base_image_name}.usdz"
            
            # 调用 Hunyuan3D API
            # 传入 base_image_name 作为 ID，内部会自动生成 .glb 并转换为 .usdz
            Hunyuan3d_api.Generate3DModel(
                prompt=translated_prompt, 
                image_path=str(image_path), 
                result_path=str(result_path), 
                modelId=base_image_name 
            )
            # 构造下载链接
            model_url = f"{ENDPOINTS}{PORT}/{base_image_name}/{model_filename}"
            return jsonify(modelId=model_filename, url=model_url)
            
        except BusinessException as be:
            return jsonify(ResponseWrapper.fail(code=be.code, message=be.message))
        except Exception as e:
            return jsonify(ResponseWrapper.fail(code=BUSINESS_FAIL, message=str(e)))
    
    return app
    # @app.route('/GenerateComponent3DModel', methods=['POST'])
    # def generate_component_3d_model():
    #     """生成组件3D模型"""
    #     try:
    #         data = request.json
    #         user_token = str(data.get('user_token'))
    #         component_order = str(data.get('componentOrder'))
    #         prompt = str(data.get('prompt'))
    #         image_id = str(data.get('imageId'))
            
    #         # 翻译提示词
    #         translation_response = Tencentcloud_api.TextTranslate(prompt)
    #         translated_prompt = json.loads(
    #             str(translation_response.read(), 'utf-8')
    #         ).get('Response').get("TargetText")
            
    #         # 处理图像
    #         image_path = OUTPUT_DIR / image_id
    #         with open(image_path, 'rb') as image_file:
    #             image_data = image_file.read()
            
    #         # 准备输出路径
    #         base_image_name = Path(image_id).stem
    #         result_path = OUTPUT_DIR / base_image_name
    #         model_filename = f"{base_image_name}.usdz"
            
    #         # 生成3D模型
    #         Rodin_api.Generate3DModel(
    #             translated_prompt, str(image_path), str(result_path), model_filename
    #         )
            
    #         model_url = f"{ENDPOINTS}{PORT}/{base_image_name}/{model_filename}"
    #         return jsonify(modelId=model_filename, url=model_url)
            
    #     except BusinessException as be:
    #         return jsonify(ResponseWrapper.fail(code=be.code, message=be.message))
    #     except Exception as e:
    #         return jsonify(ResponseWrapper.fail(code=BUSINESS_FAIL, message=str(e)))
    
    # return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)