"""
generate.py - 图像生成模块
包含：部件/整体图像生成提示词生成
"""

import sys
import os
import json

# 禁用警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import warnings
warnings.filterwarnings('ignore')

from typing import Literal
from record import client, extract_and_parse_json, text_encoder
import numpy as np

# ========== 生成模式定义 ==========

GenerateMode = Literal[1, 2]
# 1: 部件生成
# 2: 整体生成


# ========== 部件名称匹配 ==========

def find_component_in_memory(component_name: str, memory_db: dict) -> dict:
    """
    在记忆数据库中查找部件，使用 LLM 判断名称是否指向同一部件。

    Args:
        component_name: 用户输入的部件名称
        memory_db: 记忆数据库

    Returns:
        找到的部件记忆数据，没找到返回 None
    """
    # 获取所有部件节点
    component_nodes = {}
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            stored_name = data.get('component_name', '')
            component_nodes[stored_name] = data

    if not component_nodes:
        print("[Generate] No components in memory")
        return None

    # 使用 LLM 判断名称匹配
    stored_names = list(component_nodes.keys())
    match_prompt = f'''
请判断用户输入的部件名称是否与以下已存储的部件名称中的某一个指向同一个部件。

用户输入：{component_name}

已存储部件：{json.dumps(stored_names, ensure_ascii=False)}

判断规则：
1. 如果名称完全相同，返回该名称
2. 如果是同义词或方言变体（如"车座"和"车座子"、"把手"和"手柄"），返回最匹配的存储名称
3. 如果没有匹配的，返回 "无匹配"

只输出结果，不要解释。输出格式：
{{"match": "匹配的部件名称 或 无匹配"}}
'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[match_prompt]
    )

    result = extract_and_parse_json(response.text)
    if result is None:
        print("[Generate] Failed to parse LLM response")
        return None

    match_name = result.get('match', '无匹配')
    print(f"[Generate] LLM match result: '{component_name}' -> '{match_name}'")

    if match_name == '无匹配' or match_name not in component_nodes:
        return None

    return component_nodes[match_name]


def get_component_memory_text(component_data: dict) -> str:
    """
    将部件记忆数据转换为文本描述。

    Args:
        component_data: 部件节点数据

    Returns:
        部件记忆文本描述
    """
    component_name = component_data.get('component_name', '未知部件')

    # 提取各类描述
    appearance_list = component_data.get('appearance_descriptions', [])
    function_list = component_data.get('function_descriptions', [])
    structure_list = component_data.get('structure_descriptions', [])

    appearance_text = "".join([d.get('content', '') for d in appearance_list])
    function_text = "".join([d.get('content', '') for d in function_list])
    structure_text = "".join([d.get('content', '') for d in structure_list])

    return f'''
部件名称：{component_name}
外形描述：{appearance_text if appearance_text else '暂无'}
功能描述：{function_text if function_text else '暂无'}
结构描述：{structure_text if structure_text else '暂无'}
'''


def get_overall_memory_text(memory_db: dict) -> str:
    """
    获取整体产品的记忆文本。

    Args:
        memory_db: 记忆数据库

    Returns:
        整体记忆文本描述
    """
    overall_nodes = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'OVERALL':
            overall_nodes.append(data)

    if not overall_nodes:
        return "暂无整体设计记忆"

    # 合并所有整体节点信息
    overall_text = ""
    for node in overall_nodes:
        design_background = node.get('design_background', '')
        if design_background:
            overall_text += f"设计背景：{design_background}\n"

        appearance_list = node.get('overall_appearances', [])
        function_list = node.get('overall_functions', [])
        structure_list = node.get('overall_structures', [])

        for d in appearance_list:
            overall_text += f"整体外形：{d.get('content', '')}\n"
        for d in function_list:
            overall_text += f"整体功能：{d.get('content', '')}\n"
        for d in structure_list:
            overall_text += f"整体结构：{d.get('content', '')}\n"

    return overall_text if overall_text else "暂无整体设计记忆"


def get_all_components_text(memory_db: dict) -> str:
    """
    获取所有部件的记忆文本。

    Args:
        memory_db: 记忆数据库

    Returns:
        所有部件记忆文本描述
    """
    all_text = ""
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            all_text += get_component_memory_text(data) + "\n"

    return all_text if all_text else "暂无部件设计记忆"


# ========== 提示词生成 ==========

def generate_component_prompt(
    component_name: str,
    memory_db: dict,
    trigger_generate: Literal[0, 1] = 1
) -> str:
    """
    生成部件图像的提示词（t=1）。

    Args:
        component_name: 用户输入的部件名称
        memory_db: 记忆数据库
        trigger_generate: 是否触发生成（1=触发，0=不触发）

    Returns:
        图像生成提示词，不触发时返回空字符串
    """
    if trigger_generate == 0:
        print("[Generate] Not triggered (trigger_generate=0)")
        return ""

    print(f"[Generate] Component mode (t=1), searching for '{component_name}'...")

    # 查找部件记忆
    component_data = find_component_in_memory(component_name, memory_db)

    # 获取整体记忆
    overall_text = get_overall_memory_text(memory_db)

    # 构建最终提示词源
    if component_data:
        # 找到部件，结合整体记忆和部件记忆
        component_text = get_component_memory_text(component_data)

        prompt_source = f'''
## 整体产品信息
{overall_text}

## 目标部件信息
{component_text}
'''
        # DEBUG: 打印传给LLM的prompt_source
        print(f"[Generate DEBUG] prompt_source sent to LLM:\n{prompt_source}")
        print(f"[Generate] Found component '{component_data.get('component_name')}'")
    else:
        # 没找到部件，只用整体记忆
        prompt_source = f'''
## 整体产品信息
{overall_text}

## 目标部件
部件名称：{component_name}
（该部件暂无设计记忆，请根据整体风格进行设计）
'''
        print(f"[Generate] Component '{component_name}' not found in memory")
        print(f"[Generate DEBUG] prompt_source sent to LLM:\n{prompt_source}")

    # 使用 LLM 生成图像提示词
    generate_prompt = f'''
你是一个专业的产品设计图像生成提示词编写助手。请根据以下设计记忆信息，为图像生成模型编写一个详细、准确的提示词。

{prompt_source}

## 提示词编写要求
1. **必须包含记忆内容**：提示词必须自然融入上述「外形描述」「功能描述」「结构描述」中的所有已有信息，不能遗漏或忽略
2. **自然表达**：将记忆内容自然串联成流畅的描述语句，不要机械罗列
3. **适当扩展**：可在记忆内容基础上补充材质、颜色、细节等视觉特征，但不能脱离记忆的核心特征
4. **风格一致**：保持与整体产品风格的一致性
5. 使用中文编写，不超过100字
6. 直接输出提示词，不要有任何解释或前缀

## 提示词示例
假设记忆内容为「外形：科技风格、圆润流线型；功能：全地形越野」，则提示词可写：
圆润流线型履带设计，呈现科技风格，采用高强度复合材料，具备全地形越野能力，表面带有防滑纹理。

请输出图像生成提示词：
'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[generate_prompt]
    )

    image_prompt = response.text.strip()
    print(f"[Generate] Generated prompt: '{image_prompt}'")

    return image_prompt


def generate_overall_prompt(
    memory_db: dict,
    trigger_generate: Literal[0, 1] = 1,
    component_image_mapping: dict = None,
    overall_image_index: int = None
) -> str:
    """
    生成整体图像的提示词（t=2）。

    Args:
        memory_db: 记忆数据库
        trigger_generate: 是否触发生成（1=触发，0=不触发）
        component_image_mapping: 部件图片索引映射，格式如 {"车架": 1, "车轮": 2, ...}
        overall_image_index: 整体图片的索引，如 7（整体图是模型制作的粗糙图，作为结构参考）

    Returns:
        图像生成提示词，不触发时返回空字符串
    """
    if trigger_generate == 0:
        print("[Generate] Not triggered (trigger_generate=0)")
        return ""

    print("[Generate] Overall mode (t=2), combining all memories...")

    # 获取整体记忆
    overall_text = get_overall_memory_text(memory_db)

    # 获取所有部件记忆并构建图片引用信息
    all_components_text = get_all_components_text(memory_db)

    # 构建图片引用信息
    image_reference_text = ""
    if component_image_mapping:
        component_refs = []
        for comp_name, img_idx in component_image_mapping.items():
            component_refs.append(f"- {comp_name}：[@图{img_idx}]")
        image_reference_text += f"## 部件图片索引\n" + "\n".join(component_refs) + "\n\n"
        print(f"[Generate] Component image mapping: {component_image_mapping}")

    if overall_image_index is not None:
        image_reference_text += f"## 整体图片索引\n- 整体结构参考图：[@图{overall_image_index}]（此图较粗糙，仅作结构和外形参考）\n\n"
        print(f"[Generate] Overall image index: {overall_image_index}")

    prompt_source = f'''
{image_reference_text}## 整体产品信息
{overall_text}

## 所有部件信息
{all_components_text}
'''
    # DEBUG: 打印传给LLM的prompt_source
    print(f"[Generate DEBUG] prompt_source sent to LLM:\n{prompt_source}")

    # 使用 LLM 生成图像提示词
    generate_prompt = f'''
你是一个专业的产品设计图像生成提示词编写助手。

你需要根据以下信息，编写一个用于生成高质量整体产品图像的提示词。

{prompt_source}

## 关键要求
1. **必须引用图片**：提示词中必须使用 [@图N] 格式引用对应的部件图片（如 [@图0]、[@图1]）
2. **必须包含记忆内容**：必须自然融入「部件信息」和「整体产品信息」中的所有已有描述（外形、功能、结构等），不能遗漏
3. **自然表达**：将记忆内容自然串联成流畅的描述，不要机械罗列关键词
4. **部件图为主**：部件图片是高质量参考，描述时应结合记忆中的部件特征
5. **整体图为辅**：如有整体图片索引，它只是粗糙的结构参考，用于确定组装方式和整体布局

## 提示词格式示例
假设部件记忆为「履带：外形科技风格，功能全地形越野；探照灯：外形红光，功能危险时发光」，则提示词可写：
[@图0]至[@图2]的组件组合成探索机器人。[@图0]的履带呈现科技风格，具备全地形越野能力。[@图1]的探照灯发出红光，危险时可警示发光。整体布局参考[@图2]的结构，呈现硬朗的专业探索设备效果。

## 编写规则
1. 开头说明所有部件共同构成什么产品
2. 逐一描述各部件，每句引用对应图片，且必须包含该部件的记忆内容（外形/功能/结构）
3. 如有整体图片索引，提及它作为结构参考
4. 结尾描述整体效果
5. 使用中文，不超过200字
6. 直接输出提示词，不要有任何解释或前缀

请输出图像生成提示词：
'''

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[generate_prompt]
    )

    image_prompt = response.text.strip()
    print(f"[Generate] Generated prompt: '{image_prompt}'")

    return image_prompt


# ========== 统一生成接口 ==========

def process_generate_request(
    t: GenerateMode,
    component_name: str = None,
    trigger_generate: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None,
    component_image_mapping: dict = None,
    overall_image_index: int = None
) -> str:
    """
    统一图像生成接口。

    Args:
        t: 生成模式（1=部件生成，2=整体生成）
        component_name: 部件名称（t=1 时必需）
        trigger_generate: 是否触发生成（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径
        component_image_mapping: 部件图片索引映射（t=2时使用），格式如 {"车架": 1, "车轮": 2}
        overall_image_index: 整体图片索引（t=2时使用），如 7

    Returns:
        图像生成提示词，不触发时返回空字符串
    """
    # 加载记忆数据库
    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Generate] No memory file at '{memory_path}'")

    # 根据模式生成
    if t == 1:
        # 部件生成
        if component_name is None:
            print("[Generate] Error: component_name required for t=1")
            return ""

        return generate_component_prompt(component_name, memory_db, trigger_generate)

    elif t == 2:
        # 整体生成
        return generate_overall_prompt(
            memory_db,
            trigger_generate,
            component_image_mapping,
            overall_image_index
        )

    else:
        print(f"[Generate] Error: Invalid t value: {t}")
        return ""


# ========== 部件结构/功能信息提取 ==========

def get_components_structure_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> list:
    """
    提取所有部件的结构信息。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        list: 部件结构信息列表，格式如 ["履带：采用齿轮连接结构", "把手：螺纹连接", ...]
        不触发时返回空列表
    """
    if trigger == 0:
        print("[Structure Info] Not triggered (trigger=0)")
        return []

    # 加载记忆数据库
    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Structure Info] No memory file at '{memory_path}'")
            return []

    # 检索所有部件的结构描述（仅 status=1）
    structure_list = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')
            structure_descriptions = data.get('structure_descriptions', [])

            # 仅提取 status=1 的结构描述
            for desc in structure_descriptions:
                status = desc.get('status', 1)
                content = desc.get('content', '')
                if status == 1 and content and content.strip():
                    structure_list.append(f"{component_name}：{content}")

    print(f"[Structure Info] Found {len(structure_list)} structure entries")
    return structure_list


def get_components_function_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> list:
    """
    提取所有部件的功能信息。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        list: 部件功能信息列表，格式如 ["履带：用于在社区间平稳行走", "把手：方便握持", ...]
        不触发时返回空列表
    """
    if trigger == 0:
        print("[Function Info] Not triggered (trigger=0)")
        return []

    # 加载记忆数据库
    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Function Info] No memory file at '{memory_path}'")
            return []

    # 检索所有部件的功能描述（仅 status=1）
    function_list = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')
            function_descriptions = data.get('function_descriptions', [])

            # 仅提取 status=1 的功能描述
            for desc in function_descriptions:
                status = desc.get('status', 1)
                content = desc.get('content', '')
                if status == 1 and content and content.strip():
                    function_list.append(f"{component_name}：{content}")

    print(f"[Function Info] Found {len(function_list)} function entries")
    return function_list


def get_components_uncertain_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> list:
    """
    提取所有部件的待确定信息（status=0的描述）。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        list: 部件待确定信息列表，格式如 ["履带：还没想好外形设计的风格", ...]
        不触发时返回空列表
    """
    if trigger == 0:
        print("[Uncertain Info] Not triggered (trigger=0)")
        return []

    # 加载记忆数据库
    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Uncertain Info] No memory file at '{memory_path}'")
            return []

    # 检索所有节点中 status=0 的描述（部件 + 整体）
    uncertain_list = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')

            # 检查三类描述中 status=0 的内容
            desc_type_map = {
                'appearance_descriptions': '外形',
                'function_descriptions': '功能',
                'structure_descriptions': '结构'
            }
            for desc_type_key, desc_type_label in desc_type_map.items():
                descriptions = data.get(desc_type_key, [])
                for desc in descriptions:
                    status = desc.get('status', 1)  # 默认为确定状态
                    content = desc.get('content', '')
                    if status == 0 and content and content.strip():
                        uncertain_list.append({
                            "target": component_name,
                            "content": content,
                            "desc_type": desc_type_label
                        })

        elif data.get('node_type') == 'OVERALL':
            # 检查整体节点的 status=0 内容
            desc_type_map = {
                'overall_appearances': '外形',
                'overall_functions': '功能',
                'overall_structures': '结构'
            }
            for desc_type_key, desc_type_label in desc_type_map.items():
                descriptions = data.get(desc_type_key, [])
                for desc in descriptions:
                    status = desc.get('status', 1)
                    content = desc.get('content', '')
                    if status == 0 and content and content.strip():
                        uncertain_list.append({
                            "target": "整体",
                            "content": content,
                            "desc_type": desc_type_label
                        })

    print(f"[Uncertain Info] Found {len(uncertain_list)} uncertain entries")
    return uncertain_list


def get_components_appearance_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> list:
    """
    提取所有部件的外形信息（仅 status=1）。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        list: 部件外形信息列表，格式如 ["履带：采用黑色金属材质", ...]
        不触发时返回空列表
    """
    if trigger == 0:
        print("[Appearance Info] Not triggered (trigger=0)")
        return []

    if memory_db is None:
        if memory_path is None:
            memory_path = os.path.join(os.path.dirname(__file__), "object_nodes.json")

        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_db = json.load(f)
        else:
            memory_db = {}
            print(f"[Appearance Info] No memory file at '{memory_path}'")
            return []

    appearance_list = []
    for node_id, data in memory_db.items():
        if data.get('node_type') == 'COMPONENT':
            component_name = data.get('component_name', '未知部件')
            descriptions = data.get('appearance_descriptions', [])

            for desc in descriptions:
                status = desc.get('status', 1)
                content = desc.get('content', '')
                if status == 1 and content and content.strip():
                    appearance_list.append(f"{component_name}：{content}")

    print(f"[Appearance Info] Found {len(appearance_list)} appearance entries")
    return appearance_list


def get_components_info(
    trigger: Literal[0, 1] = 1,
    memory_db: dict = None,
    memory_path: str = None
) -> dict:
    """
    统一接口：同时获取部件外形、结构、功能信息和待确定信息。

    Args:
        trigger: 是否触发（1=触发，0=不触发）
        memory_db: 记忆数据库（为 None 则从文件加载）
        memory_path: 记忆文件路径

    Returns:
        dict: {
            "appearance_info": ["履带：采用黑色金属材质", ...],
            "structure_info": ["履带：采用齿轮连接结构", ...],
            "function_info": ["履带：用于在社区间平稳行走", ...],
            "uncertain_info": ["履带：还没想好外形设计的风格", ...]
        }
        不触发时返回空列表
    """
    appearance_info = get_components_appearance_info(trigger, memory_db, memory_path)
    structure_info = get_components_structure_info(trigger, memory_db, memory_path)
    function_info = get_components_function_info(trigger, memory_db, memory_path)
    uncertain_info = get_components_uncertain_info(trigger, memory_db, memory_path)

    return {
        "appearance_info": appearance_info,
        "structure_info": structure_info,
        "function_info": function_info,
        "uncertain_info": uncertain_info
    }


# ========== 从提示词变化更新记忆 ==========

def update_memory_from_prompt_change(
    mode: int,
    component_name: str,
    ai_prompt: str,
    user_prompt: str,
    memory_db: dict
) -> dict:
    """
    对比 AI 生成的提示词和用户修改后的提示词，提取设计变化并自动更新记忆。

    Args:
        mode: 1=部件生成，2=整体生成
        component_name: 部件名称（mode=1 时必需）
        ai_prompt: AI 生成的原始提示词
        user_prompt: 用户修改后的提示词
        memory_db: 记忆数据库

    Returns:
        {
            "success": true/false,
            "changes_detected": true/false,
            "detected_changes": {...},
            "actions_taken": [...],
            "new_component_created": true/false,
            "new_component_name": str 或 None
        }
    """
    from Memory import make_description, text_encoder, ComponentNode

    print(f"\n[UpdateMemory] 对比提示词变化...")
    print(f"  模式: {'部件' if mode == 1 else '整体'}")
    if mode == 1:
        print(f"  部件名: {component_name}")

    # 1. LLM 对比差异 — 返回结构化变更映射
    # 收集所有部件名（mode=2 时需要路由到对应部件）
    component_names = []
    if mode == 2:
        for node_id, data in memory_db.items():
            if data.get('node_type') == 'COMPONENT':
                component_names.append(data.get('component_name', ''))

    diff_prompt = f'''
你是一个专业的设计信息分析助手。请对比以下两个图像生成提示词，找出用户修改后所体现的设计信息变化。

【AI生成的原始提示词】
{ai_prompt}

【用户修改后的提示词】
{user_prompt}

请逐条分析变化，按以下分类提取变更：
1. replace（替换）：原始提示词中某个特征被新的特征替代（如材质从金属变为塑料）
2. add（新增）：原始提示词中没有，修改后的提示词新增的特征
3. remove（移除）：原始提示词中有，修改后的提示词不再包含的特征

分类：
- appearance：外形、颜色、材质、风格等视觉特征
- function：功能用途、使用方式、性能要求
- structure：部件关系、连接方式、布局结构

{"以下是当前产品包含的部件：\\n" + "、".join(component_names) if component_names else ""}

返回JSON格式：
{{
    "changes": [
        {{"type": "replace", "category": "appearance", "old": "原始特征原文", "new": "新特征描述", "assigned_to": "部件名 或 overall"}},
        {{"type": "add", "category": "function", "new": "新增功能描述", "assigned_to": "部件名 或 overall"}},
        {{"type": "remove", "category": "structure", "old": "被移除的结构特征原文", "assigned_to": "部件名 或 overall"}}
    ],
    {"\"implied_component_name\": \"新部件名称 或 null\"," if mode == 1 else ""}
}}

注意：
- type=replace 时必须同时提供 old 和 new
- type=add 只提供 new
- type=remove 只提供 old
- old/new 应该是完整的设计描述语句，不是差异说明
- category 必须是 appearance/function/structure 之一
- assigned_to 用于指定该变化属于哪个部件或整体：
  - 如果是某个特定部件的变化（如"履带采用深蓝色金属材质"），assigned_to 设为对应部件名
  - 如果是整体产品的变化（如整体风格、整体功能），assigned_to 设为 "overall"
- mode=1（部件生成）时，assigned_to 可以省略
- 如果没有任何变化，changes 返回空数组 []
- 只返回JSON，不要其他解释
'''

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[diff_prompt]
        )
        diff_result = extract_and_parse_json(response.text)
    except Exception as e:
        print(f"[UpdateMemory] LLM 对比失败: {e}")
        return {"success": False, "error": f"LLM 对比失败: {str(e)}"}

    if diff_result is None:
        print("[UpdateMemory] LLM 解析失败")
        return {"success": False, "error": "LLM 解析失败"}

    changes = diff_result.get("changes", [])
    implied_name = diff_result.get("implied_component_name") if mode == 1 else None

    # 按 category 和 type 分组
    replace_items = {"appearance": [], "function": [], "structure": []}
    add_items = {"appearance": [], "function": [], "structure": []}
    remove_items = {"appearance": [], "function": [], "structure": []}

    for change in changes:
        cat = change.get("category", "")
        ctype = change.get("type", "")
        if cat not in replace_items:
            continue
        if ctype == "replace":
            replace_items[cat].append({"old": change.get("old", ""), "new": change.get("new", "")})
        elif ctype == "add":
            add_items[cat].append({"new": change.get("new", "")})
        elif ctype == "remove":
            remove_items[cat].append({"old": change.get("old", "")})

    total_changes = len(replace_items["appearance"]) + len(replace_items["function"]) + len(replace_items["structure"]) \
                  + len(add_items["appearance"]) + len(add_items["function"]) + len(add_items["structure"]) \
                  + len(remove_items["appearance"]) + len(remove_items["function"]) + len(remove_items["structure"])

    if total_changes == 0:
        print("[UpdateMemory] 未检测到设计变化")
        return {
            "success": True,
            "changes_detected": False,
            "detected_changes": {"changes": []},
            "actions_taken": [],
            "new_component_created": False,
            "new_component_name": None
        }

    print(f"[UpdateMemory] 检测到 {len(changes)} 条变化:")
    for c in changes:
        assigned = c.get("assigned_to", "")
        print(f"  {c['type']} ({c['category']}): {c.get('old', '—')} → {c.get('new', '—')} [→ {assigned if assigned else component_name}]")
    if implied_name:
        print(f"[UpdateMemory] 暗示新部件名: {implied_name}")

    # 3. 按 assigned_to 路由变化
    from Memory import ComponentNode, make_description, text_encoder

    # 构建目标 -> 变化列表映射
    targets = {}  # target_name -> {"replace_items": {...}, "add_items": {...}, "remove_items": {...}}

    for change in changes:
        cat = change.get("category", "")
        ctype = change.get("type", "")
        if cat not in ("appearance", "function", "structure"):
            continue

        # 确定归属
        if mode == 1:
            target = component_name
        else:
            target = change.get("assigned_to", "overall")
            # 如果 LLM 没提供 assigned_to，用启发式判断
            if not target or target == "":
                target = _infer_target(change, component_names)

        if target not in targets:
            targets[target] = {"replace_items": {"appearance": [], "function": [], "structure": []},
                               "add_items": {"appearance": [], "function": [], "structure": []},
                               "remove_items": {"appearance": [], "function": [], "structure": []}}

        if ctype == "replace":
            targets[target]["replace_items"][cat].append({"old": change.get("old", ""), "new": change.get("new", "")})
        elif ctype == "add":
            targets[target]["add_items"][cat].append({"new": change.get("new", "")})
        elif ctype == "remove":
            targets[target]["remove_items"][cat].append({"old": change.get("old", "")})

    actions_taken = []
    new_component_created = False
    new_component_name = None
    name_changed = False

    # 对每个目标应用变化
    for target_name, t_items in targets.items():
        actions_taken.extend(_apply_changes_to_target(
            target_name, t_items["replace_items"], t_items["add_items"], t_items["remove_items"],
            memory_db, mode, component_name, implied_name
        ))

    return {
        "success": True,
        "changes_detected": True,
        "detected_changes": {
            "changes": changes,
            "component_name_changed": name_changed
        },
        "actions_taken": actions_taken,
        "new_component_created": new_component_created,
        "new_component_name": new_component_name
    }


# ========== 辅助函数 ==========

def _infer_target(change: dict, component_names: list) -> str:
    """当 LLM 没有提供 assigned_to 时，用启发式推断变化归属"""
    text = change.get("old", "") + " " + change.get("new", "")
    for name in component_names:
        if name in text:
            return name
    return "overall"


def _apply_changes_to_target(
    target_name: str,
    replace_items: dict,
    add_items: dict,
    remove_items: dict,
    memory_db: dict,
    mode: int,
    component_name: str,
    implied_name: str
) -> list:
    """将变化应用到指定目标（部件或整体），返回 actions_taken 列表"""
    from Memory import ComponentNode, OverallNode, make_description, text_encoder

    actions = []

    if target_name.lower() == "overall":
        # 找到或创建 OVERALL 节点
        overall_data = None
        for node_id, data in memory_db.items():
            if data.get('node_type') == 'OVERALL':
                overall_data = data
                break

        if overall_data is None:
            print(f"[UpdateMemory] 创建新整体节点")
            all_new = []
            for cat in ["appearance", "function", "structure"]:
                for item in replace_items[cat]:
                    if item.get("new"):
                        all_new.append((cat, item["new"]))
                for item in add_items[cat]:
                    if item.get("new"):
                        all_new.append((cat, item["new"]))

            new_node = OverallNode(
                overall_appearances=[make_description(c, 1) for cat, c in all_new if cat == "appearance"],
                overall_structures=[make_description(c, 1) for cat, c in all_new if cat == "structure"],
                overall_functions=[make_description(c, 1) for cat, c in all_new if cat == "function"],
                overall_appearance_embeddings=[text_encoder(c).tolist() for cat, c in all_new if cat == "appearance"],
                overall_structure_embeddings=[text_encoder(c).tolist() for cat, c in all_new if cat == "structure"],
                overall_function_embeddings=[text_encoder(c).tolist() for cat, c in all_new if cat == "function"],
            )
            memory_db[new_node.node_id] = new_node.model_dump()
            actions.append("created_overall")
            return actions

        print(f"[UpdateMemory] 更新整体节点")
        _update_node_descriptions(overall_data, replace_items, add_items, remove_items,
                                   "appearance", "overall_appearances", "overall_appearance_embeddings", actions, "overall")
        _update_node_descriptions(overall_data, replace_items, add_items, remove_items,
                                   "function", "overall_functions", "overall_function_embeddings", actions, "overall")
        _update_node_descriptions(overall_data, replace_items, add_items, remove_items,
                                   "structure", "overall_structures", "overall_structure_embeddings", actions, "overall")

        from datetime import datetime, timezone
        overall_data['timestamp_last_accessed'] = datetime.now(timezone.utc).isoformat()
        return actions

    # 部件节点
    existing = find_component_in_memory(target_name, memory_db)

    if existing is None:
        # 创建新部件
        print(f"[UpdateMemory] 创建新部件: {target_name}")
        all_new = []
        for cat in ["appearance", "function", "structure"]:
            for item in replace_items[cat]:
                if item.get("new"):
                    all_new.append((cat, item["new"]))
            for item in add_items[cat]:
                if item.get("new"):
                    all_new.append((cat, item["new"]))

        new_node = ComponentNode(
            component_name=target_name,
            appearance_descriptions=[make_description(c, 1) for cat, c in all_new if cat == "appearance"],
            structure_descriptions=[make_description(c, 1) for cat, c in all_new if cat == "structure"],
            function_descriptions=[make_description(c, 1) for cat, c in all_new if cat == "function"],
            appearance_embeddings=[text_encoder(c).tolist() for cat, c in all_new if cat == "appearance"],
            structure_embeddings=[text_encoder(c).tolist() for cat, c in all_new if cat == "structure"],
            function_embeddings=[text_encoder(c).tolist() for cat, c in all_new if cat == "function"],
        )
        memory_db[new_node.node_id] = new_node.model_dump()
        actions.append(f"created_component: {target_name}")
        return actions

    existing_name = existing.get('component_name', target_name)
    print(f"[UpdateMemory] 更新部件: {existing_name}")

    _update_node_descriptions(existing, replace_items, add_items, remove_items,
                               "appearance", "appearance_descriptions", "appearance_embeddings", actions, existing_name)
    _update_node_descriptions(existing, replace_items, add_items, remove_items,
                               "function", "function_descriptions", "function_embeddings", actions, existing_name)
    _update_node_descriptions(existing, replace_items, add_items, remove_items,
                               "structure", "structure_descriptions", "structure_embeddings", actions, existing_name)

    from datetime import datetime, timezone
    existing['timestamp_last_accessed'] = datetime.now(timezone.utc).isoformat()
    return actions


def _update_node_descriptions(
    data: dict,
    replace_items: dict,
    add_items: dict,
    remove_items: dict,
    cat: str,
    desc_key: str,
    emb_key: str,
    actions: list,
    label: str
):
    """更新单个节点指定类别的描述（基于快照的替换逻辑）"""
    from Memory import make_description, text_encoder

    desc_list = [make_description(d.get('content', ''), d.get('status', 1)) for d in data.get(desc_key, [])]
    emb_list = data.get(emb_key, [])

    if len(emb_list) != len(desc_list):
        emb_list = [text_encoder(d.content).tolist() for d in desc_list]

    # 创建快照
    snapshot = [(d.content, emb_list[i]) for i, d in enumerate(desc_list)]
    replace_targets = {}
    unmatched_replaces = []

    # replace
    for item in replace_items[cat]:
        old_content = item.get("old", "")
        new_content = item.get("new", "")
        if not old_content or not new_content:
            continue

        matched = False
        for i, (orig_content, _) in enumerate(snapshot):
            if orig_content == old_content:
                replace_targets.setdefault(i, []).append(new_content)
                matched = True
                break

        if not matched and snapshot:
            old_emb = text_encoder(old_content)
            embs = np.array([e for _, e in snapshot])
            sims = np.dot(embs, old_emb)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            print(f"[UpdateMemory] {cat} replace similarity: {best_sim:.4f} for '{snapshot[best_idx][0][:30]}...'")

            if best_sim > 0.75:
                replace_targets.setdefault(best_idx, []).append(new_content)
                matched = True

        if not matched:
            unmatched_replaces.append(new_content)

    # 应用替换
    for idx, new_contents in replace_targets.items():
        merged = "，".join(new_contents) if len(new_contents) > 1 else new_contents[0]
        desc_list[idx].content = merged
        desc_list[idx].status = 1
        emb_list[idx] = text_encoder(merged).tolist()
        print(f"[UpdateMemory] {cat} replaced: '{new_contents}'")

    for new_content in unmatched_replaces:
        desc_list.append(make_description(new_content, 1))
        emb_list.append(text_encoder(new_content).tolist())
        print(f"[UpdateMemory] {cat} appended: '{new_content[:30]}...'")

    # add
    for item in add_items[cat]:
        new_content = item.get("new", "")
        if new_content:
            desc_list.append(make_description(new_content, 1))
            emb_list.append(text_encoder(new_content).tolist())
            print(f"[UpdateMemory] {cat} added: '{new_content[:30]}...'")

    # remove
    for item in remove_items[cat]:
        old_content = item.get("old", "")
        if old_content:
            for i, (orig_content, _) in enumerate(snapshot):
                if orig_content == old_content:
                    desc_list[i].status = 0
                    print(f"[UpdateMemory] {cat} removed: '{old_content[:30]}...'")
                    break
            else:
                if snapshot:
                    old_emb = text_encoder(old_content)
                    embs = np.array([e for _, e in snapshot])
                    sims = np.dot(embs, old_emb)
                    best_idx = int(np.argmax(sims))
                    if float(sims[best_idx]) > 0.75:
                        desc_list[best_idx].status = 0
                        print(f"[UpdateMemory] {cat} removed (vector): '{old_content[:30]}...'")

    data[desc_key] = [d.model_dump() for d in desc_list]
    data[emb_key] = emb_list
    if replace_items[cat] or add_items[cat] or remove_items[cat]:
        actions.append(f"updated {cat} for {label}")


# ========== 测试 ==========

if __name__ == "__main__":
    print("=== Generate Module Test ===")

    # 测试部件生成
    print("\n--- Test 1: Component generation (t=1) ---")
    prompt1 = process_generate_request(
        t=1,
        component_name="履带",
        trigger_generate=1
    )
    print(f"Result: {prompt1}")

    # 测试整体生成（带图片索引）
    print("\n--- Test 2: Overall generation with image mapping (t=2) ---")
    prompt2 = process_generate_request(
        t=2,
        trigger_generate=1,
        component_image_mapping={"车架": 1, "车轮": 2, "车座": 3, "车篮": 4, "水杯架": 5, "车把手": 6},
        overall_image_index=7
    )
    print(f"Result: {prompt2}")

    # 测试整体生成（不带图片索引）
    print("\n--- Test 2b: Overall generation without image mapping ---")
    prompt2b = process_generate_request(
        t=2,
        trigger_generate=1
    )
    print(f"Result: {prompt2b}")

    # 测试部件结构/功能/待确定信息
    print("\n--- Test 3: Components info ---")
    info = get_components_info(trigger=1)
    print(f"Structure info: {info['structure_info']}")
    print(f"Function info: {info['function_info']}")
    print(f"Uncertain info: {info['uncertain_info']}")

    # 测试不触发
    print("\n--- Test 4: Not triggered ---")
    info2 = get_components_info(trigger=0)
    print(f"Result: {info2}")