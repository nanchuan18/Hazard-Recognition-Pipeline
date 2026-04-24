import sys

sys.path.append("./segment-anything-main")

import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from qwen_vl_module import qwen_vl_infer
import tempfile
import os
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


# ===================== 内存优化配置 =====================
# 强制使用 CPU 或限制 GPU 内存使用
if torch.cuda.is_available():
    # 设置 CUDA 内存分配策略，避免内存碎片化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    # 限制 PyTorch 使用的显存比例
    torch.cuda.set_per_process_memory_fraction(0.8, 0)  # 限制使用 80% 显存
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"当前设备：{DEVICE}")
if DEVICE == "cuda":
    print(f"GPU 显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# ===================== 模型初始化（全局单例）=====================
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
sam = None
mask_generator = None
yolo_model = None


def _init_sam():
    """内部函数：延迟加载 SAM 模型"""
    global sam, mask_generator
    if sam is None:
        print("正在加载 SAM 模型...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
        print("SAM 模型加载完成")
    return mask_generator


def _init_yolo():
    """内部函数：延迟加载 YOLO 模型"""
    global yolo_model
    if yolo_model is None:
        print("正在加载 YOLOv8 模型...")
        # 方案1：临时修改torch.load参数（仅兼容PyTorch 2.6+）
        import torch
        original_load = torch.load

        def custom_load(*args, **kwargs):
            kwargs.setdefault('weights_only', False)  # 强制关闭weights_only
            return original_load(*args, **kwargs)

        torch.load = custom_load
        try:
            yolo_model = YOLO("yolov8x.pt")
        finally:
            torch.load = original_load  # 恢复原始函数

        # 方案2：如果已注册安全类，直接加载（注释掉上面，启用下面）
        # yolo_model = YOLO("yolov8x.pt")

        print("YOLOv8 模型加载完成")
    return yolo_model


# ===================== 内存清理工具 =====================
def clear_memory():
    """清理 GPU 内存"""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"已清理 GPU 内存，当前分配：{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


# ===================== SAM 真实区域分割 =====================
def sam_segment(image_path):
    """
    使用 SAM 模型对图片进行区域分割

    Args:
        image_path: 图片路径

    Returns:
        list: 区域列表，每个区域包含 ID、坐标、置信度等信息
    """
    print("\n===== 1. 启动 SAM 区域像素级分割 =====")

    # 初始化 SAM 模型
    mg = _init_sam()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 降低分辨率以减少内存使用（可选）
    max_size = 1280
    scale = 1.0  # 默认缩放比例为1（不缩放）
    original_shape = image_rgb.shape[:2]  # 记录原始尺寸
    
    if max(image_rgb.shape[:2]) > max_size:
        scale = max_size / max(image_rgb.shape[:2])
        image_rgb = cv2.resize(image_rgb, None, fx=scale, fy=scale)
        print(f"图片已缩放至：{image_rgb.shape[1]}x{image_rgb.shape[0]} (缩放比例: {scale:.3f})")

    masks = mg.generate(image_rgb)

    regions = []
    for idx, mask in enumerate(masks[:5]):  # 限制区域数量
        x, y, w, h = mask["bbox"]
        # 将缩放后的坐标还原为原图坐标
        x_orig = int(x / scale)
        y_orig = int(y / scale)
        w_orig = int(w / scale)
        h_orig = int(h / scale)
        
        regions.append(
            {
                "region_id": idx + 1,
                "bbox": [x_orig, y_orig, x_orig + w_orig, y_orig + h_orig],
                "confidence": round(mask["stability_score"], 2),
                "region_type": "生产作业区域",
            }
        )
    print(f"SAM 完成，共识别 {len(regions)} 个区域")

    return regions


# ===================== YOLO 真实目标检测 =====================
def yolo_detect(image_path):
    """
    使用 YOLOv8 模型检测图片中的目标对象

    Args:
        image_path: 图片路径

    Returns:
        list: 检测到的目标列表，每个目标包含名称、坐标、置信度
    """
    print("\n===== 2. 启动 YOLOv8 目标检测 =====")

    # 初始化 YOLO 模型
    model = _init_yolo()

    # 使用较小的图片尺寸进行检测
    results = model(image_path, imgsz=640, verbose=False)
    objects = []

    for res in results:
        for box in res.boxes:
            cls_name = res.names[int(box.cls)]
            conf = round(float(box.conf), 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            objects.append({
                "object_name": cls_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

    print(f"YOLO 完成，共检测 {len(objects)} 个目标")

    return objects


# ===================== 跨模态对齐（核心联动） =====================
def convert_bbox_to_percentage(bbox_json: str, img_width: int, img_height: int) -> str:
    """
    将像素坐标转换为百分比坐标
    
    Args:
        bbox_json: 边界框 JSON 字符串，格式如 {"x": 100, "y": 200, "w": 50, "h": 60}
        img_width: 图片宽度（像素）
        img_height: 图片高度（像素）
    
    Returns:
        str: 转换后的百分比坐标 JSON 字符串，格式如 {"x": 10.5, "y": 20.3, "w": 5.2, "h": 6.1}
    """
    import re as re_module
    
    try:
        # 解析 JSON 字符串
        if isinstance(bbox_json, str):
            bbox = json.loads(bbox_json)
        else:
            bbox = bbox_json
        
        # 转换为百分比（保留两位小数）
        x_pct = round((bbox['x'] / img_width) * 100, 2)
        y_pct = round((bbox['y'] / img_height) * 100, 2)
        w_pct = round((bbox['w'] / img_width) * 100, 2)
        h_pct = round((bbox['h'] / img_height) * 100, 2)
        
        # 返回新的 JSON 字符串
        return json.dumps({"x": x_pct, "y": y_pct, "w": w_pct, "h": h_pct})
        
    except json.JSONDecodeError as e:
        # JSON 解析失败，尝试修复不规范格式
        print(f"   ⚠️ bboxJson 格式错误，尝试修复: {bbox_json}")
        try:
            if isinstance(bbox_json, str):
                # 使用更智能的正则表达式提取 x, y, w, h
                x_match = re_module.search(r'"x"\s*:\s*(\d+)', bbox_json)
                y_match = re_module.search(r'"y"\s*:\s*(\d+)', bbox_json)
                w_match = re_module.search(r'"w"\s*:\s*(\d+)', bbox_json)
                h_match = re_module.search(r'"h"\s*:\s*(\d+)', bbox_json)
                
                if x_match and y_match and w_match and h_match:
                    x = int(x_match.group(1))
                    y = int(y_match.group(1))
                    w = int(w_match.group(1))
                    h = int(h_match.group(1))
                    
                    # 转换为百分比
                    x_pct = round((x / img_width) * 100, 2)
                    y_pct = round((y / img_height) * 100, 2)
                    w_pct = round((w / img_width) * 100, 2)
                    h_pct = round((h / img_height) * 100, 2)
                    
                    print(f"   ✅ 修复成功: x={x}, y={y}, w={w}, h={h} -> x={x_pct}%, y={y_pct}%, w={w_pct}%, h={h_pct}%")
                    return json.dumps({"x": x_pct, "y": y_pct, "w": w_pct, "h": h_pct})
                else:
                    # 如果键名匹配失败，尝试提取前4个数字（兼容旧格式）
                    print(f"   ⚠️ 无法通过键名提取，尝试按位置提取数字...")
                    all_numbers = re_module.findall(r'\d+', bbox_json)
                    if len(all_numbers) >= 4:
                        # 过滤掉过小的数字（可能是版本号或其他干扰）
                        valid_numbers = [int(n) for n in all_numbers if int(n) > 10]
                        if len(valid_numbers) >= 4:
                            x, y, w, h = valid_numbers[:4]
                            
                            # 转换为百分比
                            x_pct = round((x / img_width) * 100, 2)
                            y_pct = round((y / img_height) * 100, 2)
                            w_pct = round((w / img_width) * 100, 2)
                            h_pct = round((h / img_height) * 100, 2)
                            
                            print(f"   ✅ 按位置修复成功: [{x}, {y}, {w}, {h}] -> [{x_pct}%, {y_pct}%, {w_pct}%, {h_pct}%]")
                            return json.dumps({"x": x_pct, "y": y_pct, "w": w_pct, "h": h_pct})
                    
                    print(f"   ❌ 修复失败: 无法提取足够的坐标数据")
            
        except Exception as fix_error:
            print(f"   ❌ 修复失败: {fix_error}")
            import traceback
            traceback.print_exc()
        
        # 返回原始数据
        return bbox_json if isinstance(bbox_json, str) else json.dumps(bbox_json)
        
    except Exception as e:
        print(f"   ⚠️ 坐标转换失败: {e}，保持原始坐标")
        return bbox_json if isinstance(bbox_json, str) else json.dumps(bbox_json)


def is_inside_region(obj_box, region_box):
    """
    判断目标是否在区域内

    Args:
        obj_box: 目标边界框 [x1, y1, x2, y2]
        region_box: 区域边界框 [x1, y1, x2, y2]

    Returns:
        bool: 目标是否在区域内
    """
    ox1, oy1, ox2, oy2 = obj_box
    rx1, ry1, rx2, ry2 = region_box
    return ox1 >= rx1 and oy1 >= ry1 and ox2 <= rx2 and oy2 <= ry2


def calculate_hazard_score_and_confidence(hazard_level_id: int) -> tuple:
    """
    根据隐患等级计算 score 和 confidence
    
    Args:
        hazard_level_id: 隐患等级ID（1=一般隐患, 2=较大隐患, 3=重大隐患）
    
    Returns:
        tuple: (score, confidence)
            - score: 隐患评分（0-100，越高风险越低）
            - confidence: 置信度（0-1）
    """
    # 等级映射到分数和置信度
    level_config = {
        1: {"score": 85.0, "confidence": 0.85},  # 一般隐患（低）
        2: {"score": 60.0, "confidence": 0.75},  # 较大隐患（中）
        3: {"score": 30.0, "confidence": 0.90}   # 重大隐患（高）
    }
    
    config = level_config.get(hazard_level_id, level_config[2])  # 默认较大隐患
    return config["score"], config["confidence"]


def filter_objects_in_regions(objects, regions):
    """
    筛选位于区域内的有效目标

    Args:
        objects: 目标列表
        regions: 区域列表

    Returns:
        list: 位于区域内的有效目标列表
    """
    print("\n===== 3. 跨模态对齐：筛选区域内目标 =====")
    filtered = []
    for obj in objects:
        for reg in regions:
            if is_inside_region(obj["bbox"], reg["bbox"]):
                filtered.append(obj)
                break
    print(f"筛选后有效目标：{len(filtered)} 个")
    return filtered


# ===================== Qwen-VL 隐患推理 =====================
def parse_qwen_vl_result(qwen_text: str, image_path: str = None) -> List[Dict[str, Any]]:
    """
    解析 Qwen-VL 返回的文本，提取结构化隐患数据
    
    Args:
        qwen_text: Qwen-VL 返回的文本（可能是 JSON 字符串或普通文本）
        image_path: 图片路径，用于获取图片尺寸以转换坐标为百分比
    
    Returns:
        list: 隐患列表，每个隐患包含 hazardTypeId, hazardLevelId, bboxJson, reasonText
    """
    import re
    import cv2
    
    # 获取图片尺寸用于坐标转换
    img_width, img_height = None, None
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            img_height, img_width = img.shape[:2]
            print(f"   图片尺寸: {img_width}x{img_height}")
    
    try:
        # 尝试直接解析 JSON
        qwen_text = qwen_text.strip()
        
        # 如果返回的是 JSON 数组格式
        if qwen_text.startswith('['):
            results = json.loads(qwen_text)
            # 转换为标准格式
            if isinstance(results, list):
                cleaned_results = []
                for item in results:
                    if isinstance(item, dict):
                        # 直接使用 ID 格式
                        bbox_json = item.get('bboxJson')
                        hazard_level_id = item.get('hazardLevelId', 2)
                        
                        # 如果提供了图片尺寸且 bboxJson 存在，转换像素坐标为百分比
                        if bbox_json and img_width and img_height:
                            bbox_json = convert_bbox_to_percentage(bbox_json, img_width, img_height)
                        
                        # 计算 score 和 confidence
                        score, confidence = calculate_hazard_score_and_confidence(hazard_level_id)
                        
                        cleaned_item = {
                            "hazardTypeId": item.get('hazardTypeId', 9),
                            "hazardLevelId": hazard_level_id,
                            "score": score,
                            "confidence": confidence,
                            "bboxJson": bbox_json,
                            "segmentJson": None,
                            "reasonText": item.get('reasonText'),
                            "ruleMatchJson": None
                        }
                        cleaned_results.append(cleaned_item)
                return cleaned_results
            return []
        
        # 如果返回的是带代码块的 JSON
        if '```json' in qwen_text:
            match = re.search(r'```json\s*(.*?)\s*```', qwen_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                results = json.loads(json_str)
                # 转换为标准格式
                if isinstance(results, list):
                    cleaned_results = []
                    for item in results:
                        if isinstance(item, dict):
                            # 直接使用 ID 格式
                            bbox_json = item.get('bboxJson')
                            hazard_level_id = item.get('hazardLevelId', 2)
                            
                            # 如果提供了图片尺寸且 bboxJson 存在，转换像素坐标为百分比
                            if bbox_json and img_width and img_height:
                                bbox_json = convert_bbox_to_percentage(bbox_json, img_width, img_height)
                            
                            # 计算 score 和 confidence
                            score, confidence = calculate_hazard_score_and_confidence(hazard_level_id)
                            
                            cleaned_item = {
                                "hazardTypeId": item.get('hazardTypeId', 9),
                                "hazardLevelId": hazard_level_id,
                                "score": score,
                                "confidence": confidence,
                                "bboxJson": bbox_json,
                                "segmentJson": None,
                                "reasonText": item.get('reasonText'),
                                "ruleMatchJson": None
                            }
                            cleaned_results.append(cleaned_item)
                    return cleaned_results
            return []
        
        # 如果是旧格式的文本，转换为新格式
        # 尝试从文本中提取信息
        hazards = []
        
        # 检测是否有违规
        if '违规：是' in qwen_text or '违规: 是' in qwen_text:
            # 提取违规类型
            type_match = re.search(r'违规类型[：:](.+?)(?:\n|$)', qwen_text)
            level_match = re.search(r'等级[：:](.+?)(?:\n|$)', qwen_text)
            desc_match = re.search(r'描述[：:](.+?)(?:\n|$)', qwen_text)
            
            hazard_type = type_match.group(1).strip() if type_match else "未知隐患"
            hazard_level = level_match.group(1).strip() if level_match else "中"
            description = desc_match.group(1).strip() if desc_match else qwen_text
            
            # 根据等级设置分数
            score_map = {"高": 30.0, "中": 60.0, "低": 85.0}
            confidence_map = {"高": 0.90, "中": 0.75, "低": 0.85}
            score = score_map.get(hazard_level, 60.0)
            confidence = confidence_map.get(hazard_level, 0.75)
            
            hazards.append({
                "hazardTypeId": HAZARD_TYPE_MAP.get(hazard_type, 9),  # 默认值9（缺少安全警示标志）
                "hazardLevelId": HAZARD_LEVEL_MAP.get(hazard_level, 2),  # 默认值2（较大隐患）
                "score": score,
                "confidence": confidence,
                "bboxJson": json.dumps({"x": 31, "y": 43, "w": 25, "h": 18}),
                "segmentJson": None,
                "reasonText": f"建议整改：{description}",
                "ruleMatchJson": None
            })
        
        return hazards
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败: {e}")
        print(f"   原始内容: {qwen_text[:300]}")
        
        # 尝试修复常见的 JSON 格式问题
        try:
            # 0. 先尝试修复 bboxJson 中的引号问题
            # 将 "bboxJson": "{\"x":123,"y":456,...}" 修复为正确格式
            import re as re_module
            
            # 匹配 bboxJson 字段并修复内部的引号
            def fix_bbox_json(match):
                bbox_content = match.group(1)
                # 如果 bbox_content 看起来像是一个未正确转义的 JSON 字符串
                if '"x":' in bbox_content or "'x':" in bbox_content:
                    # 提取 x, y, w, h 的值
                    x_match = re_module.search(r'["\']?x["\']?\s*[:=]\s*(\d+)', bbox_content)
                    y_match = re_module.search(r'["\']?y["\']?\s*[:=]\s*(\d+)', bbox_content)
                    w_match = re_module.search(r'["\']?w["\']?\s*[:=]\s*(\d+)', bbox_content)
                    h_match = re_module.search(r'["\']?h["\']?\s*[:=]\s*(\d+)', bbox_content)
                    
                    if x_match and y_match and w_match and h_match:
                        x, y, w, h = x_match.group(1), y_match.group(1), w_match.group(1), h_match.group(1)
                        # 返回正确转义的 JSON 字符串
                        return f'"bboxJson": "{{\\"x\\":{x},\\"y\\":{y},\\"w\\":{w},\\"h\\":{h}}}"'
                return match.group(0)
            
            # 修复 bboxJson 字段
            qwen_text_fixed = re_module.sub(
                r'"bboxJson":\s*"([^"]*(?:"[^"]*)*[^"]*)"',
                fix_bbox_json,
                qwen_text
            )
            
            # 1. 尝试直接解析修复后的 JSON
            if qwen_text_fixed.startswith('['):
                try:
                    results = json.loads(qwen_text_fixed)
                    if isinstance(results, list):
                        cleaned_results = []
                        for item in results:
                            if isinstance(item, dict):
                                hazard_type_name = item.get('hazardTypeName', '')
                                hazard_level_name = item.get('hazardLevelName', '')
                                
                                bbox_json = item.get('bboxJson')
                                # 如果提供了图片尺寸且 bboxJson 存在，转换像素坐标为百分比
                                if bbox_json and img_width and img_height:
                                    bbox_json = convert_bbox_to_percentage(bbox_json, img_width, img_height)
                                
                                cleaned_item = {
                                    "hazardTypeId": HAZARD_TYPE_MAP.get(hazard_type_name, 9),
                                    "hazardLevelId": HAZARD_LEVEL_MAP.get(hazard_level_name, 2),
                                    "bboxJson": bbox_json,
                                    "segmentJson": None,
                                    "reasonText": item.get('advice'),
                                    "ruleMatchJson": None
                                }
                                cleaned_item["score"], cleaned_item["confidence"] = calculate_hazard_score_and_confidence(
                                    cleaned_item["hazardLevelId"]
                                )
                                cleaned_results.append(cleaned_item)
                        if cleaned_results:
                            print(f"✅ 修复后成功解析 {len(cleaned_results)} 个隐患")
                            return cleaned_results
                except:
                    pass
            
            # 2. 尝试从截断的 JSON 中提取有用的部分
            # 提取所有完整的隐患对象（支持 ID 格式和名称格式）
            hazard_pattern = r'\{[^}]*(?:"hazardTypeId"|"hazardTypeName")[^}]*(?:"hazardLevelId"|"hazardLevelName")[^}]*\}'
            matches = re_module.findall(hazard_pattern, qwen_text_fixed, re_module.DOTALL)
            
            if matches:
                print(f"   尝试从截断的 JSON 中提取 {len(matches)} 个隐患对象...")
                cleaned_results = []
                for match in matches:
                    try:
                        # 先修复单个对象中的 bboxJson
                        match_fixed = re_module.sub(
                            r'"bboxJson"\s*:\s*"([^"]*(?:"[^"]*)*[^"]*)"',
                            fix_bbox_json,
                            match
                        )
                        hazard = json.loads(match_fixed)
                        
                        # 支持两种格式：ID 格式和名称格式
                        if 'hazardTypeId' in hazard:
                            # 新格式：直接使用 ID
                            bbox_json = hazard.get('bboxJson')
                            # 如果提供了图片尺寸且 bboxJson 存在，转换像素坐标为百分比
                            if bbox_json and img_width and img_height:
                                bbox_json = convert_bbox_to_percentage(bbox_json, img_width, img_height)
                            
                            cleaned_item = {
                                "hazardTypeId": hazard.get('hazardTypeId', 9),
                                "hazardLevelId": hazard.get('hazardLevelId', 2),
                                "bboxJson": bbox_json,
                                "segmentJson": None,
                                "reasonText": hazard.get('reasonText') or hazard.get('advice'),
                                "ruleMatchJson": None
                            }
                        else:
                            # 旧格式：需要转换名称为 ID
                            hazard_type_name = hazard.get('hazardTypeName', '')
                            hazard_level_name = hazard.get('hazardLevelName', '')
                            
                            bbox_json = hazard.get('bboxJson')
                            # 如果提供了图片尺寸且 bboxJson 存在，转换像素坐标为百分比
                            if bbox_json and img_width and img_height:
                                bbox_json = convert_bbox_to_percentage(bbox_json, img_width, img_height)
                            
                            cleaned_item = {
                                "hazardTypeId": HAZARD_TYPE_MAP.get(hazard_type_name, 9),
                                "hazardLevelId": HAZARD_LEVEL_MAP.get(hazard_level_name, 2),
                                "bboxJson": bbox_json,
                                "segmentJson": None,
                                "reasonText": hazard.get('reasonText') or hazard.get('advice'),
                                "ruleMatchJson": None
                            }
                        cleaned_results.append(cleaned_item)
                    except Exception as obj_err:
                        print(f"   跳过无法解析的对象: {obj_err}")
                        continue
                
                if cleaned_results:
                    print(f"✅ 成功从截断 JSON 中提取 {len(cleaned_results)} 个隐患")
                    return cleaned_results
            
            # 2. 尝试找到最接近的完整 JSON 数组
            # 从 '[' 到最后一个完整的 '}' 后跟 ']' 或 ', ]' 等
            array_match = re.search(r'\[.*?\}(?:\s*,\s*\{[^}]*\})*\s*\]', qwen_text, re.DOTALL)
            if array_match:
                try:
                    partial_json = array_match.group(0)
                    results = json.loads(partial_json)
                    if isinstance(results, list):
                        cleaned_results = []
                        for item in results:
                            if isinstance(item, dict):
                                # 转换为新格式
                                hazard_type_name = item.get('hazardTypeName', '')
                                hazard_level_name = item.get('hazardLevelName', '')
                                
                                bbox_json = item.get('bboxJson')
                                # 如果提供了图片尺寸且 bboxJson 存在，转换像素坐标为百分比
                                if bbox_json and img_width and img_height:
                                    bbox_json = convert_bbox_to_percentage(bbox_json, img_width, img_height)
                                
                                cleaned_item = {
                                    "hazardTypeId": HAZARD_TYPE_MAP.get(hazard_type_name, 9),
                                    "hazardLevelId": HAZARD_LEVEL_MAP.get(hazard_level_name, 2),
                                    "bboxJson": bbox_json,
                                    "segmentJson": None,
                                    "reasonText": item.get('advice'),
                                    "ruleMatchJson": None
                                }
                                cleaned_item["score"], cleaned_item["confidence"] = calculate_hazard_score_and_confidence(
                                    cleaned_item["hazardLevelId"]
                                )
                                cleaned_results.append(cleaned_item)
                        print(f"✅ 从部分 JSON 中解析成功，共 {len(cleaned_results)} 个隐患")
                        return cleaned_results
                except:
                    pass
        except Exception as repair_error:
            print(f"   JSON 修复也失败: {repair_error}")
        
        # 返回默认提示
        return [{
            "hazardTypeId": 9,  # 默认值9（缺少安全警示标志）
            "hazardLevelId": 2,  # 默认值2（较大隐患）
            "score": 60.0,
            "confidence": 0.75,
            "bboxJson": json.dumps({"x": 50, "y": 50, "w": 20, "h": 20}),
            "segmentJson": None,
            "reasonText": f"AI 分析结果解析失败，请人工复核。原始输出：{qwen_text[:100]}",
            "ruleMatchJson": None
        }]
    except Exception as e:
        print(f"⚠️ 解析 Qwen-VL 结果失败: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认提示
        return [{
            "hazardTypeId": 9,  # 默认值9（缺少安全警示标志）
            "hazardLevelId": 2,  # 默认值2（较大隐患）
            "score": 60.0,
            "confidence": 0.75,
            "bboxJson": json.dumps({"x": 50, "y": 50, "w": 20, "h": 20}),
            "segmentJson": None,
            "reasonText": f"AI 分析结果解析失败，请人工复核。原始输出：{qwen_text[:100]}",
            "ruleMatchJson": None
        }]


def safety_check(image_path, regions, objects):
    """
    使用 Qwen-VL 进行安全隐患分析

    Args:
        image_path: 图片路径
        regions: 区域列表
        objects: 有效目标列表

    Returns:
        list: 隐患列表，每个隐患包含 hazardTypeName, hazardLevelName, bboxJson, advice
    """
    print("\n===== 4. Qwen-VL-Max 隐患识别推理 =====")
    rules = """
1. 作业区域内人员必须佩戴安全帽，未佩戴属于重大隐患
2. 禁止在作业区域吸烟、使用明火
3. 禁止违规攀爬、跨越设备
4. 必须穿戴反光背心
    """
    qwen_text = qwen_vl_infer(image_path, regions, objects, rules)
    
    # 解析 Qwen-VL 返回的结果为结构化数据
    hazards = parse_qwen_vl_result(qwen_text, image_path)
    
    print(f"Qwen-VL 分析完成，发现 {len(hazards)} 个隐患")
    return hazards


# ===================== 主流程：完整的隐患识别流程 =====================
def hazard_recognition_pipeline(image_path):
    """
    完整的安全生产隐患三级识别流程

    Args:
        image_path: 输入图片路径

    Returns:
        dict: 包含以下字段的字典
            - status (str): 处理状态 ("success" 或 "error")
            - message (str): 处理消息
            - data (dict): 详细数据
                - regions (list): 识别到的作业区域
                - detected_objects (list): 检测到的所有目标
                - valid_objects_in_regions (list): 区域内的有效目标
                - hazard_analysis (str): 隐患分析结果
    """
    print("\n" + "=" * 60)
    print("开始处理图片：", image_path)
    print("=" * 60)

    try:
        # 检查图片是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在：{image_path}")

        # 检查图片大小
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
        print(f"图片大小：{file_size:.2f} MB")
        if file_size > 10:
            print("⚠️ 警告：图片较大，建议压缩后上传")

        # 1. 区域识别
        regions = sam_segment(image_path)

        # 2. 对象识别
        objects = yolo_detect(image_path)

        # 3. 跨模态对齐
        valid_objects = filter_objects_in_regions(objects, regions)

        # 4. 行为识别 & 隐患输出
        hazard_result = safety_check(image_path, regions, valid_objects)

        # 构建响应数据 - 使用新的 JSON 格式
        response_data = {
            "status": "success",
            "message": "隐患识别完成",
            "data": {
                "results": hazard_result  # 直接使用隐患列表
            }
        }

        print(f"\n{'=' * 60}")
        print("隐患识别完成")
        print(f"{'=' * 60}\n")

        return response_data

    except Exception as e:
        error_msg = f"隐患识别失败：{str(e)}"
        print(f"\n❌ {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "data": None
        }
    finally:
        # 清理内存
        clear_memory()


# ===================== 映射配置 =====================
# 隐患类型映射
HAZARD_TYPE_MAP = {
    "未戴安全帽": 1,
    "未穿反光背心": 2,
    "禁烟区吸烟": 3,
    "消防通道堵塞": 4,
    "灭火器缺失": 5,
    "灭火器过期": 6,
    "私拉乱接电线": 7,
    "电线裸露": 8,
    "缺少安全警示标志": 9,
    "物料堆放混乱": 10,
    "叉车占用通道": 11,
    "危险化学品泄漏": 12,
    "缺少防护栏": 13,
    "安全网破损": 14,
    "梯子使用不规范": 15,
    "气瓶摆放不规范": 16,
    "未使用个人防护装备": 17,
    "工具散落地面": 18,
    "疏散通道堵塞": 19,
    "危险品存放不规范": 20,
    "粉尘堆积严重": 21,
    "高处坠落风险": 22,
    "叉车碰撞风险": 23,
    "人员疲劳作业": 24,
    "车辆超速": 25,
    "人员聚集风险": 26,
    "设备超负荷运行": 27,
    "违规操作设备": 28,
    "动火作业风险": 29,
    "高空坠物风险": 30,
    "未经授权进入危险区域": 31,
    "车辆倒车盲区风险": 32
}

# 隐患等级映射
HAZARD_LEVEL_MAP = {
    "一般隐患": 1,
    "较大隐患": 2,
    "重大隐患": 3,
    # 兼容旧格式
    "低": 1,
    "中": 2,
    "高": 3
}


# ===================== 数据模型定义 =====================
class HazardResult(BaseModel):
    """隐患结果模型"""
    hazardTypeId: int = Field(..., description="隐患类型ID")
    hazardLevelId: int = Field(..., description="隐患等级ID")
    score: float = Field(0.0, description="隐患评分（0-100，越高风险越低）")
    confidence: float = Field(0.0, description="置信度（0-1）")
    bboxJson: Optional[str] = Field(None, description="隐患位置边界框 JSON 字符串")
    segmentJson: Optional[str] = Field(None, description="分割信息（固定为null）")
    reasonText: Optional[str] = Field(None, description="安全建议")
    ruleMatchJson: Optional[str] = Field(None, description="规则匹配信息（固定为null）")


class RecognitionData(BaseModel):
    """识别数据模型"""
    results: List[HazardResult] = Field(..., description="隐患结果列表")


class RecognitionSummary(BaseModel):
    """识别统计信息"""
    regions_count: int = Field(..., description="识别区域数")
    objects_count: int = Field(..., description="检测目标数")
    valid_objects_count: int = Field(..., description="区域内有效目标数")
    hazards_count: int = Field(0, description="发现的隐患数")


class RecognitionResult(BaseModel):
    """识别结果模型"""
    status: str = Field(..., description="处理状态: success/error")
    message: str = Field(..., description="处理消息")
    data: Optional[RecognitionData] = Field(None, description="详细数据")
    summary: Optional[RecognitionSummary] = Field(None, description="识别统计")


# ===================== 业务逻辑层 =====================

class HazardRecognitionService:
    """隐患识别业务服务类"""

    def __init__(self):
        """初始化服务"""
        pass

    def process_image(self, image_path: str) -> RecognitionResult:
        """
        处理图片并进行隐患识别

        Args:
            image_path: 图片文件路径

        Returns:
            RecognitionResult: 识别结果对象
        """
        # 调用原有的识别流程
        result_dict = hazard_recognition_pipeline(image_path)

        # 转换为标准数据模型
        if result_dict["status"] == "success":
            hazards = result_dict["data"]["results"]
            
            # 将字典列表转换为 HazardResult 对象列表
            hazard_objects = [HazardResult(**hazard) for hazard in hazards]
            
            recognition_data = RecognitionData(results=hazard_objects)
            
            summary = RecognitionSummary(
                regions_count=0,  # 新格式不再返回区域信息
                objects_count=0,  # 新格式不再返回目标信息
                valid_objects_count=0,
                hazards_count=len(hazards)
            )

            return RecognitionResult(
                status="success",
                message="隐患识别完成",
                data=recognition_data,
                summary=summary
            )
        else:
            return RecognitionResult(
                status="error",
                message=result_dict["message"],
                data=None
            )


# 创建全局服务实例
recognition_service = HazardRecognitionService()


# ===================== 工具函数（保持向后兼容）=====================
def interactive_mode():
    """
    交互式命令行模式，用户可以上传图片进行隐患识别
    """
    print("\n" + "=" * 60)
    print("🔍 安全生产隐患识别系统 - 交互模式")
    print("=" * 60)
    print("\n使用说明：")
    print("1. 输入图片路径进行隐患识别")
    print("2. 输入 'help' 查看帮助信息")
    print("3. 输入 'quit' 或 'exit' 退出程序")
    print("4. 输入 'status' 查看系统状态")
    print("5. 直接按回车可测试默认图片 (test.jpg)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("请输入图片路径: ").strip()

            # 去除可能存在的引号（单引号或双引号）
            if user_input and user_input[0] in ['"', "'"] and user_input[-1] in ['"', "'"]:
                user_input = user_input[1:-1].strip()

            # 退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用，再见！👋\n")
                break

            # 帮助信息
            elif user_input.lower() in ['help', 'h', '?']:
                print("\n" + "-" * 60)
                print("📖 帮助信息")
                print("-" * 60)
                print("支持的图片格式: JPG, JPEG, PNG")
                print("建议图片大小: 不超过 10MB")
                print("\n可用命令:")
                print("  - help/h/?    : 显示此帮助信息")
                print("  - status      : 查看系统状态")
                print("  - quit/q      : 退出程序")
                print("  - 直接回车    : 使用默认测试图片 test.jpg")
                print("-" * 60 + "\n")
                continue

            # 查看系统状态
            elif user_input.lower() == 'status':
                info = get_system_info()
                print("\n" + "-" * 60)
                print("📊 系统状态")
                print("-" * 60)
                print(f"运行设备: {info['device'].upper()}")
                if info['gpu_memory_gb']:
                    print(f"GPU 显存: {info['gpu_memory_gb']} GB")
                print(f"SAM 模型: {'✅ 已加载' if info['sam_loaded'] else '❌ 未加载'}")
                print(f"YOLO 模型: {'✅ 已加载' if info['yolo_loaded'] else '❌ 未加载'}")
                print("-" * 60 + "\n")
                continue

            # 使用默认测试图片
            elif user_input == '':
                image_path = 'test.jpg'
                if not os.path.exists(image_path):
                    print(f"\n⚠️ 警告: 默认测试图片 {image_path} 不存在\n")
                    continue
            else:
                image_path = user_input

            # 验证文件是否存在
            if not os.path.exists(image_path):
                print(f"\n❌ 错误: 文件不存在 - {image_path}\n")
                continue

            # 验证文件类型
            valid_extensions = ['.jpg', '.jpeg', '.png']
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in valid_extensions:
                print(f"\n❌ 错误: 不支持的文件格式 '{file_ext}'")
                print(f"支持的格式: {', '.join(valid_extensions)}\n")
                continue

            # 执行隐患识别
            print(f"\n🔄 开始处理图片: {image_path}")
            result_obj = recognition_service.process_image(image_path)

            # 转换为字典格式用于显示
            result = {
                'status': result_obj.status,
                'message': result_obj.message,
                'data': result_obj.data,
                'summary': result_obj.summary
            }

            # 显示结果
            if result['status'] == 'success':
                data = result['data']  # RecognitionData 对象
                summary = result['summary']  # RecognitionSummary 对象
                
                print("\n" + "=" * 60)
                print("✅ 识别完成 - 结果摘要")
                print("=" * 60)
                print(f"🎯 发现隐患数: {summary.hazards_count}")
                
                # 显示隐患列表
                if data and data.results:
                    print(f"\n📋 隐患详情 (共 {len(data.results)} 个):")
                    print("-" * 60)
                    for idx, hazard in enumerate(data.results, 1):
                        print(f"\n隐患 {idx}:")
                        print(f"  类型ID: {hazard.hazardTypeId}")
                        print(f"  等级ID: {hazard.hazardLevelId}")
                        print(f"  建议: {hazard.reasonText}")
                        print(f"  位置: {hazard.bboxJson}")
                    print("-" * 60)
                else:
                    print("\n✅ 未发现安全隐患，现场安全状况良好！")

                # 询问是否保存详细结果
                save_option = input("\n是否保存详细结果到 JSON 文件? (y/n): ").strip().lower()
                if save_option == 'y':
                    import json
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"result_{timestamp}.json"
                    # 将 Pydantic 模型转换为字典
                    result_dict = {
                        'status': result_obj.status,
                        'message': result_obj.message,
                        'data': result_obj.data.model_dump() if result_obj.data else None,
                        'summary': result_obj.summary.model_dump() if result_obj.summary else None
                    }
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result_dict, f, ensure_ascii=False, indent=2)
                    print(f"✅ 结果已保存到: {output_file}\n")
            else:
                print(f"\n❌ 识别失败: {result['message']}\n")

            print()

        except KeyboardInterrupt:
            print("\n\n感谢使用，再见！👋\n")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}\n")
            import traceback
            traceback.print_exc()


# ===================== 创建 FastAPI 应用 =====================
app = FastAPI(
    title="安全生产隐患识别系统",
    description="""
## 系统简介
基于 **SAM（Segment Anything）+ YOLOv8 + Qwen-VL** 三大模型的安全生产隐患三级识别系统。
### 核心功能
1. **区域识别** - SAM 自动分割生产作业区域
2. **目标检测** - YOLOv8 检测人员、设备等目标  
3. **跨模态对齐** - 筛选区域内的有效目标
4. **隐患分析** - Qwen-VL 进行违规行为推理
### API 接口
- `POST /api/v1/hazard-recognition` - 上传图片进行隐患识别
- `GET /api/v1/health` - 健康检查
### 返回数据格式

```json
{
    "results": [
        {
            "hazardTypeId": 9,
            "hazardLevelId": 1,
            "score": 85.0,
            "confidence": 0.85,
            "bboxJson": "{\"x\":31,\"y\":43,\"w\":25,\"h\":18}",
            "segmentJson": null,
            "reasonText": "建议补充安全警示标识",
            "ruleMatchJson": null
        }
    ]
}
```

**字段说明**：
- **hazardTypeId**: 隐患类型ID（1-32）
- **hazardLevelId**: 隐患等级ID（1=一般隐患, 2=较大隐患, 3=重大隐患）
- **score**: 隐患评分（0-100，分数越高风险越低）
  - 一般隐患：85.0
  - 较大隐患：60.0
  - 重大隐患：30.0
- **confidence**: 置信度（0-1）
  - 一般隐患：0.85
  - 较大隐患：0.75
  - 重大隐患：0.90
- **bboxJson**: 隐患位置边界框（百分比坐标）
- **segmentJson**: 分割信息（固定为null）
- **reasonText**: 安全建议
- **ruleMatchJson**: 规则匹配信息（固定为null）
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ===================== 工具函数：获取系统状态 =====================
def get_system_info():
    """
    获取系统运行状态信息
    Returns:
        dict: 系统信息，包括设备类型、显存、模型加载状态等
    """
    return {
        "device": DEVICE,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
                               2) if DEVICE == "cuda" else None,
        "sam_loaded": sam is not None,
        "yolo_loaded": yolo_model is not None
    }


# ===================== API 路由注册 =====================
@app.get("/api/v1/health", summary="健康检查", tags=["系统管理"])
async def health_check():
    """
    检查服务运行状态和模型加载情况

    返回:
        - device: 当前运行设备 (cuda/cpu)
        - gpu_memory_gb: GPU 显存大小
        - sam_loaded: SAM 模型是否已加载
        - yolo_loaded: YOLO 模型是否已加载
    """
    return get_system_info()


@app.post("/api/v1/hazard-recognition", summary="上传图片进行隐患识别", tags=["核心业务"])
async def recognize_hazard(
        file: UploadFile = File(
            ...,
            description="现场图片文件（支持 JPG/PNG 格式，最大 10MB）"
        )
):
    """
    ## 功能说明
    接收上传的安全生产现场图片，自动进行三级隐患识别分析。
    ## 处理流程
    1. **区域分割**: 使用 SAM 识别作业区域
    2. **目标检测**: 使用 YOLOv8 检测人员、设备等
    3. **跨模态对齐**: 筛选区域内的有效目标
    4. **隐患推理**: 使用 Qwen-VL 分析违规行为

    ## 请求参数

    - **file**: 上传的图片文件
      - 支持格式：JPG, JPEG, PNG
      - 建议大小：不超过 10MB
      - Content-Type: multipart/form-data

    ## 返回结果

    ```json
    {
        "results": [
            {
                "hazardTypeId": 1,
                "hazardLevelId": 3,
                "bboxJson": "{\"x\": 87, \"y\": 248, \"w\": 19, \"h\": 16}",
                "segmentJson": null,
                "reasonText": "立即要求作业人员佩戴符合标准的安全帽。",
                "ruleMatchJson": null
            },
            {
                "hazardTypeId": 9,
                "hazardLevelId": 1,
                "bboxJson": "{\"x\": 31, \"y\": 43, \"w\": 25, \"h\": 18}",
                "segmentJson": null,
                "reasonText": "建议补充安全警示标识，保持现场风险提示清晰可见。",
                "ruleMatchJson": null
            }
        ]
    }
    ```

    ## 错误码

    - **400**: 不支持的文件类型
    - **500**: 内部处理错误
    """
    # 验证文件类型
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FILE_TYPE",
                "message": f"不支持的文件类型：{file.content_type}",
                "allowed_types": ["image/jpeg", "image/jpg", "image/png"]
            }
        )

    tmp_path = None
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        print(f"\n{'=' * 60}")
        print(f"收到图片：{file.filename}")
        print(f"文件大小：{len(contents) / 1024:.2f} KB")
        print(f"{'=' * 60}")

        # 调用服务类处理图片
        result_obj = recognition_service.process_image(tmp_path)

        # 转换为字典
        result = result_obj.model_dump()
        
        # 返回包含 results 包装的格式
        if result["status"] == "success" and result["data"]:
            return {
                "results": result["data"]["results"]
            }
        else:
            # 失败情况返回空 results
            return {
                "results": []
            }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"识别失败：{str(e)}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RECOGNITION_ERROR",
                "message": error_msg
            }
        )

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"已清理临时文件：{tmp_path}")





# ===================== 主程序入口 =====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="安全生产隐患识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python pipeline.py                          # 启动交互模式
  python pipeline.py --interactive            # 启动交互模式
  python pipeline.py --image test.jpg         # 识别指定图片
  python pipeline.py --api                    # 启动 API 服务
        """
    )

    parser.add_argument('--interactive', '-i', action='store_true',
                        help='启动交互式命令行模式')
    parser.add_argument('--image', type=str,
                        help='指定要识别的图片路径')
    parser.add_argument('--api', action='store_true',
                        help='启动 FastAPI 服务')

    args = parser.parse_args()

    # 默认启动交互模式
    if args.interactive or (not args.image and not args.api):
        interactive_mode()
    elif args.image:
        # 单次识别模式
        if os.path.exists(args.image):
            result = hazard_recognition_pipeline(args.image)
            import json

            print("\n识别结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"❌ 错误: 文件不存在 - {args.image}")
    elif args.api:
        # 启动 API 服务（需要 uvicorn）
        print("\n启动 API 服务...")
        print("访问 http://127.0.0.1:8000/docs 查看 API 文档\n")
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
