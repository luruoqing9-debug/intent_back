import base64
import requests
import json


try:
    import aspose.threed as a3d
    ASPOSE_AVAILABLE = True
except ImportError:
    ASPOSE_AVAILABLE = False
    print("⚠️ 警告: aspose.threed 未安装, 无法转换usdz格式")


class Hunyuan3DClient:
    def __init__(self, api_host="localhost", api_port=8080, timeout=1200):
        self.base_url = f"http://{api_host}:{api_port}"
        self.timeout = timeout

    def generate_3d_from_image(self, image_path, output_path=None, convert_to_usdz=False):
        if output_path is None:
            output_path = image_path.replace('.png', '.glb').replace('.jpg', '.glb')

        try:
            with open(image_path, "rb") as image_file:
                img_b64_str = base64.b64encode(image_file.read()).decode('utf-8')

            payload = {
                "image": img_b64_str
            }

            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self.timeout
            )

            response.raise_for_status()

            with open(output_path, "wb") as output_file:
                output_file.write(response.content)

            print(f"✅ 3D模型生成成功: {output_path}")

            if convert_to_usdz and ASPOSE_AVAILABLE:
                usdz_path = self.convert_to_usdz(output_path)
                if usdz_path:
                    return usdz_path

            return output_path

        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            return None

    def convert_to_usdz(self, glb_path):
        if not ASPOSE_AVAILABLE:
            print("❌ aspose.threed 未安装，无法转换")
            return None

        try:
            usdz_path = glb_path.replace('.glb', '.usdz')
            scene = a3d.Scene.from_file(glb_path)
            scene.save(usdz_path)
            print(f"✅ 转换为 USDZ 成功: {usdz_path}")
            return usdz_path
        except Exception as e:
            print(f"❌ USDZ 转换失败: {str(e)}")
            return None


if __name__ == "__main__":
    client = Hunyuan3DClient(api_host="localhost", api_port=8080)
    
    image_path = "your_image.png"
    output_path = "output_model.glb"
    
    # 生成glb并自动转换为usdz
    result = client.generate_3d_from_image(
        image_path, 
        output_path, 
        convert_to_usdz=True
    )
    
    # 单独转换已有的glb文件
    # usdz_path = client.convert_to_usdz("existing_model.glb")
