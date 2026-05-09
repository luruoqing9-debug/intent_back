"""
debug_glb.py - 诊断 GLB 文件结构，排查 UV/贴图问题
"""

import os
import sys
import json
import base64
from PIL import Image
import numpy as np

try:
    from pygltflib import GLTF2
except ImportError:
    print("请先安装 pygltflib: pip install pygltflib")
    sys.exit(1)

def diagnose_glb(glb_path):
    print(f"\n{'='*60}")
    print(f"GLB 诊断: {glb_path}")
    print(f"{'='*60}\n")

    glb_size = os.path.getsize(glb_path)
    print(f"文件大小: {glb_size:,} bytes ({glb_size/1024/1024:.1f} MB)")

    gltf = GLTF2().load(glb_path)

    # --- 1. Buffers ---
    print(f"\n--- Buffers ({len(gltf.buffers)}) ---")
    for i, buf in enumerate(gltf.buffers):
        print(f"  Buffer[{i}]: uri={buf.uri}, byteLength={buf.byteLength:,}")

    # --- 2. BufferViews ---
    print(f"\n--- BufferViews ({len(gltf.bufferViews)}) ---")
    for i, bv in enumerate(gltf.bufferViews):
        stride = bv.byteStride if bv.byteStride else "N/A"
        print(f"  BV[{i}]: buffer={bv.buffer}, offset={bv.byteOffset}, len={bv.byteLength}, stride={stride}, target={bv.target}")

    # --- 3. Accessors ---
    print(f"\n--- Accessors ({len(gltf.accessors)}) ---")
    for i, acc in enumerate(gltf.accessors):
        type_map = {
            "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
            "MAT2": 4, "MAT3": 9, "MAT4": 16
        }
        comp_size = {"BYTE": 1, "UBYTE": 1, "SHORT": 2, "USHORT": 2,
                     "INT": 4, "UINT": 4, "FLOAT": 4, "DOUBLE": 8}
        n_comp = type_map.get(acc.type, 1)
        c_size = comp_size.get(acc.componentType, 4)
        expected = acc.count * n_comp * c_size
        print(f"  ACC[{i}]: type={acc.type}, comp={acc.componentType}, "
              f"count={acc.count}, bufferView={acc.bufferView}, "
              f"expected_bytes={expected:,}")

    # --- 4. Meshes ---
    print(f"\n--- Meshes ({len(gltf.meshes)}) ---")
    for mi, mesh in enumerate(gltf.meshes):
        print(f"  Mesh[{mi}]: {mesh.name or 'unnamed'}")
        for pi, prim in enumerate(mesh.primitives):
            print(f"    Primitive[{pi}]:")
            print(f"      mode={prim.mode}")
            for attr_name, acc_idx in prim.attributes.__dict__.items():
                if acc_idx is not None:
                    acc = gltf.accessors[acc_idx]
                    print(f"      {attr_name}: accessor[{acc_idx}], type={acc.type}, count={acc.count}")
            if prim.indices is not None:
                idx_acc = gltf.accessors[prim.indices]
                print(f"      indices: accessor[{prim.indices}], type={idx_acc.type}, count={idx_acc.count}, comp={idx_acc.componentType}")
            if prim.material is not None:
                print(f"      material: {prim.material}")

    # --- 5. Materials ---
    if gltf.materials:
        print(f"\n--- Materials ({len(gltf.materials)}) ---")
        for mi, mat in enumerate(gltf.materials):
            print(f"  Material[{mi}]: {mat.name or 'unnamed'}")
            if mat.pbrMetallicRoughness:
                pbr = mat.pbrMetallicRoughness
                print(f"    metallicFactor={pbr.metallicFactor}")
                print(f"    roughnessFactor={pbr.roughnessFactor}")
                if pbr.baseColorFactor:
                    print(f"    baseColorFactor={pbr.baseColorFactor}")
                if pbr.baseColorTexture:
                    tex_idx = pbr.baseColorTexture.index
                    print(f"    baseColorTexture: tex[{tex_idx}]")
                    if gltf.textures and tex_idx < len(gltf.textures):
                        tex = gltf.textures[tex_idx]
                        print(f"      texture source: img[{tex.source}]")
                        if tex.sampler is not None:
                            samp = gltf.samplers[tex.sampler]
                            print(f"      sampler: mag={samp.magFilter}, min={samp.minFilter}, wrap={samp.wrapS}/{samp.wrapT}")
            if mat.emissiveFactor:
                print(f"    emissiveFactor={mat.emissiveFactor}")

    # --- 6. Images ---
    if gltf.images:
        print(f"\n--- Images ({len(gltf.images)}) ---")
        for i, img in enumerate(gltf.images):
            if img.bufferView is not None:
                bv = gltf.bufferViews[img.bufferView]
                print(f"  Image[{i}]: bufferView={i}, mimeType={img.mimeType}, size={bv.byteLength:,} bytes")
            elif img.uri:
                print(f"  Image[{i}]: uri={img.uri[:50]}...")

    # --- 7. UV 数据详细分析 ---
    print(f"\n--- UV 数据深度分析 ---")
    if gltf.meshes:
        mesh = gltf.meshes[0]
        prim = mesh.primitives[0]

        # 读取 binary data
        with open(glb_path, "rb") as f:
            f.read(12); f.read(4)
            json_len_val = int.from_bytes(f.read(4), 'little')
            f.read(4 + json_len_val)
            bin_len_val = int.from_bytes(f.read(4), 'little')
            f.read(4)
            bin_data = f.read(bin_len_val)

        # 获取 index accessor
        if prim.indices is not None:
            idx_acc = gltf.accessors[prim.indices]
            idx_bv = gltf.bufferViews[idx_acc.bufferView]
            idx_offset = idx_bv.byteOffset + (idx_acc.byteOffset or 0)
            idx_data = bin_data[idx_offset:idx_offset + idx_bv.byteLength]

            if idx_acc.componentType == 5125:  # UINT32
                indices = np.frombuffer(idx_data, dtype=np.uint32)
            elif idx_acc.componentType == 5123:  # UNSIGNED_SHORT
                indices = np.frombuffer(idx_data, dtype=np.uint16)
            else:
                indices = np.frombuffer(idx_data, dtype=np.uint8)

            print(f"  索引数组: dtype={indices.dtype}, len={len(indices)}, "
                  f"min={indices.min()}, max={indices.max()}, unique={len(np.unique(indices))}")

        # 获取 POSITION accessor
        pos_acc = gltf.accessors[prim.attributes.POSITION]
        pos_bv = gltf.bufferViews[pos_acc.bufferView]
        pos_offset = pos_bv.byteOffset + (pos_acc.byteOffset or 0)
        positions = np.frombuffer(bin_data[pos_offset:pos_offset + pos_bv.byteLength], dtype='<f4').reshape(-1, 3)
        print(f"  POSITION: shape={positions.shape}, min={positions.min(axis=0)}, max={positions.max(axis=0)}")

        # 获取 TEXCOORD_0 accessor
        if hasattr(prim.attributes, 'TEXCOORD_0') and prim.attributes.TEXCOORD_0 is not None:
            uv_acc = gltf.accessors[prim.attributes.TEXCOORD_0]
            uv_bv = gltf.bufferViews[uv_acc.bufferView]
            uv_offset = uv_bv.byteOffset + (uv_acc.byteOffset or 0)
            uv_data = bin_data[uv_offset:uv_offset + uv_bv.byteLength]

            if uv_acc.componentType == 5126:  # FLOAT
                uvs = np.frombuffer(uv_data, dtype='<f4').reshape(-1, 2)
            else:
                uvs = np.frombuffer(uv_data, dtype='<f2').reshape(-1, 2) / 65535.0

            print(f"  TEXCOORD_0: shape={uvs.shape}, dtype={uvs.dtype}, "
                  f"min=({uvs[:,0].min():.4f}, {uvs[:,1].min():.4f}), "
                  f"max=({uvs[:,0].max():.4f}, {uvs[:,1].max():.4f})")
            print(f"  UV count ({len(uvs)}) vs Position count ({len(positions)}) vs Index unique ({len(np.unique(indices))})")

            # 检查 UV 范围
            print(f"\n  前 10 个 UV 坐标:")
            for j in range(min(10, len(uvs))):
                print(f"    UV[{j}] = ({uvs[j][0]:.6f}, {uvs[j][1]:.6f})")

            # 检查 face-varying 展平后的 UV
            fv_uvs = uvs[indices]
            print(f"\n  Face-varying UV (按索引展平): shape={fv_uvs.shape}, "
                  f"unique={len(np.unique(fv_uvs.reshape(-1, 2), axis=0))}")
            print(f"  前 10 个 face-varying UV:")
            for j in range(min(10, len(fv_uvs))):
                print(f"    FV_UV[{j}] (from idx {indices[j]}) = ({fv_uvs[j][0]:.6f}, {fv_uvs[j][1]:.6f})")

            # 检查三角形内的 UV 一致性
            print(f"\n  前 5 个三角形的 UV:")
            for j in range(min(5, len(indices)//3)):
                i0, i1, i2 = indices[j*3], indices[j*3+1], indices[j*3+2]
                uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
                print(f"    Triangle[{j}]: idx=[{i0},{i1},{i2}]")
                print(f"      UV0=({uv0[0]:.4f},{uv0[1]:.4f}) UV1=({uv1[0]:.4f},{uv1[1]:.4f}) UV2=({uv2[0]:.4f},{uv2[1]:.4f})")

        else:
            print("  TEXCOORD_0: 不存在！模型没有 UV 坐标！")

    # --- 8. 提取并检查贴图 ---
    print(f"\n--- 贴图分析 ---")
    if gltf.images:
        img = gltf.images[0]
        if img.bufferView is not None:
            bv = gltf.bufferViews[img.bufferView]
            buf = gltf.buffers[bv.buffer]
            img_data = bin_data[bv.byteOffset:bv.byteOffset + bv.byteLength]

            # 保存贴图
            ext = "png" if (img.mimeType and "png" in img.mimeType) else "jpg"
            tex_path = glb_path.replace('.glb', f'_texture.{ext}')
            with open(tex_path, "wb") as f:
                f.write(img_data)
            print(f"  贴图已提取: {tex_path} ({len(img_data):,} bytes)")

            # 检查贴图尺寸
            try:
                im = Image.open(tex_path)
                print(f"  贴图尺寸: {im.size[0]}x{im.size[1]}, mode={im.mode}")
            except Exception as e:
                print(f"  无法打开贴图: {e}")

    print(f"\n{'='*60}")
    print("诊断完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        glb_path = sys.argv[1]
    else:
        # 查找最新的 GLB 文件
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_3d_models")
        glb_files = [f for f in os.listdir(model_dir) if f.endswith('.glb')]
        if glb_files:
            glb_path = os.path.join(model_dir, sorted(glb_files)[-1])
        else:
            print("没有找到 .glb 文件！请先生成 3D 模型。")
            sys.exit(1)

    if not os.path.exists(glb_path):
        print(f"文件不存在: {glb_path}")
        sys.exit(1)

    diagnose_glb(glb_path)
