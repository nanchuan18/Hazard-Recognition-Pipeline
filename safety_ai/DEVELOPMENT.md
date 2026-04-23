# 安全生产隐患识别系统 - 开发文档

## 📖 文档说明

本文档面向开发人员，详细介绍系统的架构设计、核心模块、API 接口、数据流程和扩展开发指南。

**版本**: v1.0.0  
**最后更新**: 2026-04-23

---

## 📋 目录

- [1. 系统架构](#1-系统架构)
- [2. 技术栈](#2-技术栈)
- [3. 项目结构](#3-项目结构)
- [4. 核心模块详解](#4-核心模块详解)
- [5. 数据流与处理流程](#5-数据流与处理流程)
- [6. API 接口规范](#6-api-接口规范)
- [7. 数据模型](#7-数据模型)
- [8. 配置说明](#8-配置说明)
- [9. 开发环境搭建](#9-开发环境搭建)
- [10. 测试指南](#10-测试指南)
- [11. 部署指南](#11-部署指南)
- [12. 扩展开发](#12-扩展开发)
- [13. 常见问题](#13-常见问题)

---

## 1. 系统架构

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                    │
│                   (pipeline.py: 8000)                    │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│              HazardRecognitionService                    │
│           (业务逻辑层 - 服务编排)                         │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│          Hazard Recognition Pipeline                     │
│              (三级识别流程)                               │
├─────────────────────────────────────────────────────────┤
│  Stage 1: SAM 区域分割                                   │
│  Stage 2: YOLOv8 目标检测                                │
│  Stage 3: 跨模态对齐                                     │
│  Stage 4: Qwen-VL 隐患推理                               │
└──────────────┬──────────────────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────────┐
│  Local AI   │  │  Cloud AI        │
│  Models     │  │  Service         │
├─────────────┤  ├──────────────────┤
│ • SAM       │  │ • Qwen-VL-Max    │
│ • YOLOv8    │  │   (阿里云 API)    │
└─────────────┘  └──────────────────┘
```

### 1.2 设计模式

- **单例模式**: SAM 和 YOLO 模型全局单例，避免重复加载
- **延迟加载**: 模型在首次使用时才加载，减少启动时间
- **服务层模式**: `HazardRecognitionService` 封装业务逻辑
- **管道模式**: 四级处理流程（区域→目标→对齐→推理）

---

## 2. 技术栈

### 2.1 后端框架

| 技术 | 版本 | 用途 |
|------|------|------|
| FastAPI | 0.104.1 | Web 框架，提供 RESTful API |
| Uvicorn | 0.24.0 | ASGI 服务器 |
| Pydantic | 内置 | 数据验证和序列化 |

### 2.2 AI 模型

| 模型 | 版本 | 用途 | 来源 |
|------|------|------|------|
| SAM (Segment Anything) | vit_h | 图像区域分割 | Facebook Research |
| YOLOv8 | x (extra large) | 目标检测 | Ultralytics |
| Qwen-VL-Max | - | 多模态隐患推理 | 阿里云通义千问 |

### 2.3 数据处理

| 库 | 版本 | 用途 |
|----|------|------|
| OpenCV | 4.8.1.78 | 图像预处理 |
| NumPy | 1.24.3 | 数值计算 |
| PyTorch | 2.1.0 | 深度学习框架 |
| Torchvision | 0.16.0 | 计算机视觉工具 |

### 2.4 其他依赖

| 库 | 版本 | 用途 |
|----|------|------|
| Requests | 2.31.0 | HTTP 请求（调用 Qwen-VL API） |
| Python-multipart | 0.0.6 | 文件上传支持 |

---

## 3. 项目结构

```
safety_ai/
├── pipeline.py                  # 主程序 & API 服务（核心）
├── qwen_vl_module.py            # Qwen-VL API 调用模块
├── network_diagnosis.py         # 网络诊断工具
├── test_api.py                  # API 接口测试脚本
├── test_sam.py                  # SAM 模型测试脚本
├── test_yolo.py                 # YOLO 模型测试脚本
├── requirements.txt             # Python 依赖清单
├── sam_vit_h_4b8939.pth         # SAM 预训练模型（~2.4GB）
├── yolov8x.pt                   # YOLOv8x 预训练模型（~130MB）
├── segment-anything-main/       # SAM 源码仓库
│   ├── segment_anything/        # SAM 核心代码
│   ├── demo/                    # React 前端演示
│   └── notebooks/               # Jupyter 示例
├── result_*.json                # 识别结果输出文件
└── test.jpg                     # 测试图片
```

### 3.1 核心文件说明

#### `pipeline.py` (1085 行)
**职责**: 系统入口、API 路由、业务编排

**主要组件**:
- **内存优化配置** (23-36 行): GPU 显存管理
- **模型初始化** (38-81 行): SAM 和 YOLO 延迟加载
- **SAM 区域分割** (94-143 行): `sam_segment()`
- **YOLO 目标检测** (147-179 行): `yolo_detect()`
- **跨模态对齐** (183-218 行): `filter_objects_in_regions()`
- **Qwen-VL 推理** (465-490 行): `safety_check()`
- **JSON 解析容错** (222-462 行): `parse_qwen_vl_result()`
- **完整流程** (494-563 行): `hazard_recognition_pipeline()`
- **数据模型** (616-645 行): Pydantic 模型定义
- **业务服务** (649-700 行): `HazardRecognitionService`
- **FastAPI 应用** (851-1035 行): API 路由注册
- **交互模式** (704-848 行): CLI 命令行界面
- **主程序入口** (1041-1085 行): argparse 参数解析

#### `qwen_vl_module.py` (126 行)
**职责**: 阿里云 Qwen-VL API 调用封装

**主要函数**:
- `check_network()`: 网络连接检查
- `qwen_vl_infer()`: 调用 Qwen-VL API 进行隐患分析

**关键配置**:
- `API_KEY`: 阿里云 API 密钥
- `API_URL`: DashScope API 端点

#### `network_diagnosis.py` (162 行)
**职责**: 网络问题诊断工具

**诊断项**:
- DNS 解析检查
- 网络连通性测试
- API Key 配置验证
- 代理设置检查

---

## 4. 核心模块详解

### 4.1 模型管理模块

#### 4.1.1 延迟加载机制

```python
# 全局变量
sam = None
mask_generator = None
yolo_model = None

def _init_sam():
    """内部函数：延迟加载 SAM 模型"""
    global sam, mask_generator
    if sam is None:  # 仅在首次调用时加载
        print("正在加载 SAM 模型...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
        print("SAM 模型加载完成")
    return mask_generator
```

**优点**:
- 减少启动时间（无需预先加载所有模型）
- 节省内存（未使用的模型不占用资源）
- 线程安全（Python GIL 保证）

#### 4.1.2 内存优化策略

```python
if torch.cuda.is_available():
    # 防止内存碎片化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    # 限制显存使用比例（80%）
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
```

**优化效果**:
- 避免 OOM (Out Of Memory) 错误
- 提高 GPU 利用率
- 支持并发请求

### 4.2 SAM 区域分割模块

#### 4.2.1 功能说明

使用 Segment Anything Model (SAM) 自动识别图像中的作业区域。

#### 4.2.2 实现细节

```python
def sam_segment(image_path):
    # 1. 读取图片并转换为 RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 图片缩放（最大边长 1280px）
    max_size = 1280
    scale = 1.0
    if max(image_rgb.shape[:2]) > max_size:
        scale = max_size / max(image_rgb.shape[:2])
        image_rgb = cv2.resize(image_rgb, None, fx=scale, fy=scale)
    
    # 3. 生成掩码
    masks = mg.generate(image_rgb)
    
    # 4. 提取前 5 个区域，还原坐标到原图尺寸
    regions = []
    for idx, mask in enumerate(masks[:5]):
        x, y, w, h = mask["bbox"]
        x_orig = int(x / scale)
        y_orig = int(y / scale)
        w_orig = int(w / scale)
        h_orig = int(h / scale)
        
        regions.append({
            "region_id": idx + 1,
            "bbox": [x_orig, y_orig, x_orig + w_orig, y_orig + h_orig],
            "confidence": round(mask["stability_score"], 2),
            "region_type": "生产作业区域",
        })
    
    return regions
```

**关键点**:
- 图片缩放以提高处理速度
- 坐标还原确保位置准确
- 限制区域数量（最多 5 个）避免过多干扰

### 4.3 YOLO 目标检测模块

#### 4.3.1 功能说明

使用 YOLOv8x 检测图像中的人员、设备、安全装备等目标。

#### 4.3.2 实现细节

```python
def yolo_detect(image_path):
    # 1. 初始化模型（延迟加载）
    model = _init_yolo()
    
    # 2. 执行检测（imgsz=640 平衡速度和精度）
    results = model(image_path, imgsz=640, verbose=False)
    
    # 3. 解析检测结果
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
    
    return objects
```

**支持的类别**:
YOLOv8x 基于 COCO 数据集，可检测 80 种常见物体，包括：
- person (人员)
- backpack (背包)
- handbag (手提包)
- suitcase (行李箱)
- 等等...

**注意**: 如需检测安全帽等专业类别，需要重新训练模型。

### 4.4 跨模态对齐模块

#### 4.4.1 功能说明

筛选出位于作业区域内的有效目标，排除背景干扰。

#### 4.4.2 实现逻辑

```python
def is_inside_region(obj_box, region_box):
    """判断目标是否在区域内"""
    ox1, oy1, ox2, oy2 = obj_box
    rx1, ry1, rx2, ry2 = region_box
    return ox1 >= rx1 and oy1 >= ry1 and ox2 <= rx2 and oy2 <= ry2

def filter_objects_in_regions(objects, regions):
    """筛选位于区域内的有效目标"""
    filtered = []
    for obj in objects:
        for reg in regions:
            if is_inside_region(obj["bbox"], reg["bbox"]):
                filtered.append(obj)
                break
    return filtered
```

**算法复杂度**: O(n × m)，其中 n 为目标数，m 为区域数

### 4.5 Qwen-VL 隐患推理模块

#### 4.5.1 功能说明

调用阿里云 Qwen-VL-Max API，结合图像和结构化数据进行安全隐患分析。

#### 4.5.2 Prompt 工程

```python
prompt = f"""
你是安全生产隐患识别 AI。
区域：{regions}
目标：{objects}
规则：{safety_rules}

请分析图片中的安全隐患，并以 JSON 数组格式返回结果。
每个隐患对象包含以下字段：
- hazardTypeName: 隐患类型名称（从指定列表中选择）
- hazardLevelName: 隐患等级（高/中/低）
- bboxJson: 隐患位置的边界框
- advice: 整改建议

如果没有发现隐患，返回空数组 []。
请只返回 JSON 数组，不要添加其他文字说明。
"""
```

**设计要点**:
- 提供明确的字段定义
- 限定可选值范围（枚举约束）
- 要求纯 JSON 输出（便于解析）
- 给出示例格式

#### 4.5.3 JSON 容错解析

由于 LLM 可能返回不规范的 JSON，系统实现了多层容错机制：

```python
def parse_qwen_vl_result(qwen_text: str) -> List[Dict[str, Any]]:
    try:
        # 1. 尝试直接解析
        if qwen_text.startswith('['):
            results = json.loads(qwen_text)
            return convert_to_new_format(results)
        
        # 2. 提取代码块中的 JSON
        if '```json' in qwen_text:
            match = re.search(r'```json\s*(.*?)\s*```', qwen_text, re.DOTALL)
            if match:
                results = json.loads(match.group(1))
                return convert_to_new_format(results)
        
        # 3. 修复常见的 JSON 格式问题
        #    - 引号转义问题
        #    - 截断的 JSON
        #    - 部分完整的对象
        
        # 4. 文本解析（旧格式兼容）
        #    从自然语言中提取隐患信息
        
        # 5. 返回默认提示
        return [{
            "hazardTypeId": 9,
            "hazardLevelId": 2,
            "reasonText": f"AI 分析结果解析失败，请人工复核。原始输出：{qwen_text[:100]}"
        }]
        
    except Exception as e:
        # 记录错误日志
        traceback.print_exc()
        return default_fallback()
```

**容错层级**:
1. 标准 JSON 解析
2. Markdown 代码块提取
3. 正则表达式修复
4. 部分 JSON 提取
5. 自然语言解析
6. 默认降级方案

### 4.6 映射配置模块

#### 4.6.1 隐患类型映射

```python
HAZARD_TYPE_MAP = {
    "未戴安全帽": 1,
    "未穿反光背心": 2,
    "禁烟区吸烟": 3,
    # ... 共 32 种类型
}
```

**设计原因**:
- API 返回字符串类型名称
- 数据库存储整数 ID
- 便于统计和查询

#### 4.6.2 隐患等级映射

```python
HAZARD_LEVEL_MAP = {
    "一般隐患": 1,
    "较大隐患": 2,
    "重大隐患": 3,
    # 兼容旧格式
    "低": 1,
    "中": 2,
    "高": 3
}
```

**兼容性设计**: 同时支持新旧两种命名方式

---

## 5. 数据流与处理流程

### 5.1 完整处理流程

```
用户上传的图片
      │
      ▼
┌─────────────────┐
│ 1. 图片验证      │ ← 检查文件格式、大小
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. SAM 区域分割  │ → 输出: regions[]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. YOLO 目标检测 │ → 输出: objects[]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. 跨模态对齐    │ → 输出: valid_objects[]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Qwen-VL 推理  │ → 输出: hazards[]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. 结果格式化    │ → 输出: JSON 响应
└────────┬────────┘
         │
         ▼
    返回给用户
```

### 5.2 数据结构流转

#### 输入
```python
image_path: str  # 图片文件路径
```

#### Stage 1: SAM 输出
```python
regions = [
    {
        "region_id": 1,
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95,
        "region_type": "生产作业区域"
    }
]
```

#### Stage 2: YOLO 输出
```python
objects = [
    {
        "object_name": "person",
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.92
    }
]
```

#### Stage 3: 对齐后
```python
valid_objects = [
    # 过滤后的对象（位于区域内）
]
```

#### Stage 4: Qwen-VL 输出（原始）
```json
[
  {
    "hazardTypeName": "未戴安全帽",
    "hazardLevelName": "高",
    "bboxJson": "{\"x\":87,\"y\":248,\"w\":19,\"h\":16}",
    "advice": "立即要求作业人员佩戴符合标准的安全帽。"
  }
]
```

#### 最终输出（转换后）
```json
{
  "results": [
    {
      "hazardTypeId": 1,
      "hazardLevelId": 3,
      "bboxJson": "{\"x\":87,\"y\":248,\"w\":19,\"h\":16}",
      "segmentJson": null,
      "reasonText": "立即要求作业人员佩戴符合标准的安全帽。",
      "ruleMatchJson": null
    }
  ]
}
```

### 5.3 错误处理流程

```python
try:
    # 正常处理流程
    result = process_image(image_path)
except FileNotFoundError:
    return {"status": "error", "message": "文件不存在"}
except Exception as e:
    log_error(e)
    return {"status": "error", "message": str(e)}
finally:
    clear_memory()  # 清理 GPU 内存
```

---

## 6. API 接口规范

### 6.1 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json` (响应)
- **认证**: 暂无（生产环境需添加）

### 6.2 接口列表

#### 6.2.1 健康检查

**请求**:
```http
GET /api/v1/health
```

**响应** (200 OK):
```json
{
  "device": "cuda",
  "gpu_memory_gb": 16.0,
  "sam_loaded": true,
  "yolo_loaded": true
}
```

**字段说明**:
- `device`: 运行设备 (cuda/cpu)
- `gpu_memory_gb`: GPU 显存大小（仅 CUDA）
- `sam_loaded`: SAM 模型是否已加载
- `yolo_loaded`: YOLO 模型是否已加载

---

#### 6.2.2 隐患识别

**请求**:
```http
POST /api/v1/hazard-recognition
Content-Type: multipart/form-data

file: <binary>
```

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件（JPG/PNG，≤10MB） |

**响应** (200 OK):
```json
{
  "results": [
    {
      "hazardTypeId": 1,
      "hazardLevelId": 3,
      "bboxJson": "{\"x\":87,\"y\":248,\"w\":19,\"h\":16}",
      "segmentJson": null,
      "reasonText": "立即要求作业人员佩戴符合标准的安全帽。",
      "ruleMatchJson": null
    }
  ]
}
```

**响应字段说明**:
| 字段 | 类型 | 说明 |
|------|------|------|
| hazardTypeId | int | 隐患类型 ID（1-32） |
| hazardLevelId | int | 隐患等级 ID（1-3） |
| bboxJson | string | 边界框 JSON 字符串 |
| segmentJson | null | 保留字段（暂未使用） |
| reasonText | string | 整改建议 |
| ruleMatchJson | null | 保留字段（暂未使用） |

**错误响应**:

**400 Bad Request** - 不支持的文件类型
```json
{
  "detail": {
    "code": "INVALID_FILE_TYPE",
    "message": "不支持的文件类型：text/plain",
    "allowed_types": ["image/jpeg", "image/jpg", "image/png"]
  }
}
```

**500 Internal Server Error** - 处理失败
```json
{
  "detail": {
    "code": "RECOGNITION_ERROR",
    "message": "识别失败：具体错误信息"
  }
}
```

### 6.3 cURL 示例

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 上传图片
curl -X POST "http://localhost:8000/api/v1/hazard-recognition" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```

### 6.4 Python 客户端示例

```python
import requests

# 健康检查
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json())

# 隐患识别
with open("test.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/v1/hazard-recognition",
        files=files
    )
    print(response.json())
```

---

## 7. 数据模型

### 7.1 Pydantic 模型定义

#### HazardResult
```python
class HazardResult(BaseModel):
    """隐患结果模型"""
    hazardTypeId: int = Field(..., description="隐患类型ID")
    hazardLevelId: int = Field(..., description="隐患等级ID")
    bboxJson: Optional[str] = Field(None, description="边界框 JSON 字符串")
    segmentJson: Optional[str] = Field(None, description="分割信息（固定为null）")
    reasonText: Optional[str] = Field(None, description="安全建议")
    ruleMatchJson: Optional[str] = Field(None, description="规则匹配信息（固定为null）")
```

#### RecognitionData
```python
class RecognitionData(BaseModel):
    """识别数据模型"""
    results: List[HazardResult] = Field(..., description="隐患结果列表")
```

#### RecognitionSummary
```python
class RecognitionSummary(BaseModel):
    """识别统计信息"""
    regions_count: int = Field(..., description="识别区域数")
    objects_count: int = Field(..., description="检测目标数")
    valid_objects_count: int = Field(..., description="区域内有效目标数")
    hazards_count: int = Field(0, description="发现的隐患数")
```

#### RecognitionResult
```python
class RecognitionResult(BaseModel):
    """识别结果模型"""
    status: str = Field(..., description="处理状态: success/error")
    message: str = Field(..., description="处理消息")
    data: Optional[RecognitionData] = Field(None, description="详细数据")
    summary: Optional[RecognitionSummary] = Field(None, description="识别统计")
```

### 7.2 模型使用示例

```python
# 创建隐患结果
hazard = HazardResult(
    hazardTypeId=1,
    hazardLevelId=3,
    bboxJson='{"x":87,"y":248,"w":19,"h":16}',
    reasonText="立即要求作业人员佩戴符合标准的安全帽。"
)

# 序列化为字典
data_dict = hazard.model_dump()

# 序列化为 JSON
json_str = hazard.model_dump_json()
```

**注意**: Pydantic v2 使用 `model_dump()` 而非 `dict()`

---

## 8. 配置说明

### 8.1 环境变量

```bash
# GPU 配置（可选）
export CUDA_VISIBLE_DEVICES=0  # 指定使用的 GPU

# 代理配置（如需要访问外网）
export HTTP_PROXY=http://proxy_server:port
export HTTPS_PROXY=http://proxy_server:port

# PyTorch 配置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 8.2 模型配置

```python
# pipeline.py 中的配置项
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"  # SAM 模型路径
SAM_MODEL_TYPE = "vit_h"                  # SAM 模型类型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 图片处理配置
max_size = 1280  # SAM 处理的最大边长
imgsz = 640      # YOLO 检测的图片尺寸
```

### 8.3 API 配置

```python
# qwen_vl_module.py
API_KEY = "sk-your-api-key-here"  # 阿里云 API Key
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
```

### 8.4 服务配置

```python
# FastAPI 应用配置
app = FastAPI(
    title="安全生产隐患识别系统",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc
)

# Uvicorn 启动配置
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 9. 开发环境搭建

### 9.1 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **GPU** (可选): NVIDIA GPU with CUDA 11.x+
- **内存**: ≥8GB RAM (推荐 16GB+)
- **存储**: ≥10GB 可用空间（模型文件较大）

### 9.2 安装步骤

#### Step 1: 克隆项目

```bash
git clone <repository-url>
cd safety_ai
```

#### Step 2: 创建虚拟环境（推荐）

```bash
# 使用 Conda
conda create -n safety_ai python=3.10
conda activate safety_ai

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### Step 3: 安装依赖

```bash
pip install -r requirements.txt
```

**CUDA 版本选择**:
```bash
# CPU 版本（无 GPU）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

#### Step 4: 下载模型文件

```bash
# 方法 1: 手动下载
# SAM 模型: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# 放置到项目根目录

# YOLO 模型: 首次运行时自动下载
# 或手动下载: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

#### Step 5: 配置 API Key

编辑 `qwen_vl_module.py`:
```python
API_KEY = "sk-your-actual-api-key"
```

获取 API Key: [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)

#### Step 6: 验证安装

```bash
# 测试 SAM 模型
python test_sam.py

# 测试 YOLO 模型
python test_yolo.py

# 运行网络诊断
python network_diagnosis.py
```

### 9.3 IDE 配置

#### VS Code

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "black"
}
```

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. 添加虚拟环境解释器
3. 安装 requirements.txt 中的依赖

---

## 10. 测试指南

### 10.1 单元测试

#### 测试 SAM 模型加载

```bash
python test_sam.py
```

预期输出:
```
SAM 下载成功！可以正常使用
```

#### 测试 YOLO 模型

```bash
python test_yolo.py
```

#### 测试 API 接口

```bash
python test_api.py
```

测试内容:
1. ✅ 健康检查接口
2. ✅ 隐患识别接口
3. ✅ 无效文件格式处理

### 10.2 集成测试

#### 启动服务

```bash
python pipeline.py --api
```

#### 交互式测试

```bash
python pipeline.py --interactive
```

输入图片路径进行测试。

#### 单次识别测试

```bash
python pipeline.py --image test.jpg
```

### 10.3 性能测试

#### 测试图片处理时间

```python
import time
from pipeline import hazard_recognition_pipeline

start = time.time()
result = hazard_recognition_pipeline("test.jpg")
end = time.time()

print(f"处理时间: {end - start:.2f} 秒")
```

**预期性能** (RTX 3090):
- SAM 区域分割: ~2-5 秒
- YOLO 目标检测: ~0.5-1 秒
- Qwen-VL 推理: ~3-8 秒（取决于网络）
- **总计**: ~6-15 秒

### 10.4 压力测试

使用 `ab` 或 `wrk` 进行并发测试：

```bash
# Apache Bench
ab -n 100 -c 10 -p test.jpg -T image/jpeg \
   http://localhost:8000/api/v1/hazard-recognition
```

**注意**: 由于模型推理耗时较长，建议限制并发数。

---

## 11. 部署指南

### 11.1 本地部署

#### 开发环境

```bash
python pipeline.py --api
```

#### 生产环境（使用 Gunicorn）

```bash
pip install gunicorn

gunicorn pipeline:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### 11.2 Docker 部署

#### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 安装 Python 和依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python3", "pipeline.py", "--api"]
```

#### 构建和运行

```bash
# 构建镜像
docker build -t safety-ai .

# 运行容器
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  safety-ai
```

### 11.3 云服务器部署

#### Nginx 反向代理配置

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 10M;  # 限制上传文件大小
    }
}
```

#### Systemd 服务配置

`/etc/systemd/system/safety-ai.service`:
```ini
[Unit]
Description=Safety AI Recognition Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/safety_ai
Environment="PATH=/opt/safety_ai/venv/bin"
ExecStart=/opt/safety_ai/venv/bin/gunicorn pipeline:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

启动服务:
```bash
sudo systemctl enable safety-ai
sudo systemctl start safety-ai
```

### 11.4 监控和日志

#### 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

#### 健康监控

定期调用 `/api/v1/health` 检查服务状态。

---

## 12. 扩展开发

### 12.1 添加新的隐患类型

#### Step 1: 更新映射表

在 `pipeline.py` 中添加：

```python
HAZARD_TYPE_MAP = {
    # ... 现有类型
    "新隐患类型名称": 33,  # 分配新 ID
}
```

#### Step 2: 更新 Prompt

在 `qwen_vl_module.py` 的 prompt 中添加新类型到可选值列表：

```python
隐患类型可选值：..., 新隐患类型名称
```

#### Step 3: 测试

上传包含新隐患类型的图片进行测试。

### 12.2 自定义安全规则

编辑 `pipeline.py` 中的 `safety_check` 函数：

```python
def safety_check(image_path, regions, objects):
    rules = """
1. 作业区域内人员必须佩戴安全帽，未佩戴属于重大隐患
2. 禁止在作业区域吸烟、使用明火
3. 必须穿戴反光背心
4. 【新增规则】禁止在消防通道堆放物品
5. 【新增规则】高空作业必须系安全带
    """
    qwen_text = qwen_vl_infer(image_path, regions, objects, rules)
    # ...
```

### 12.3 添加新的 API 端点

#### 示例：批量识别接口

```python
@app.post("/api/v1/batch-recognition", summary="批量隐患识别")
async def batch_recognize(files: List[UploadFile] = File(...)):
    """
    批量处理多张图片
    """
    results = []
    for file in files:
        # 保存临时文件
        tmp_path = save_temp_file(file)
        
        # 调用识别服务
        result_obj = recognition_service.process_image(tmp_path)
        
        # 收集结果
        results.append({
            "filename": file.filename,
            "result": result_obj.model_dump()
        })
        
        # 清理临时文件
        os.unlink(tmp_path)
    
    return {"results": results}
```

### 12.4 替换 YOLO 模型

如果需要检测专业类别（如安全帽、反光背心），需要重新训练模型：

#### Step 1: 准备数据集

标注包含安全帽、反光背心等类别的图片。

#### Step 2: 训练模型

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8x.pt')

# 训练
model.train(
    data='safety_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='safety_yolo'
)
```

#### Step 3: 替换模型文件

将训练好的模型替换 `yolov8x.pt`。

### 12.5 添加数据库支持

#### 安装 SQLAlchemy

```bash
pip install sqlalchemy psycopg2-binary
```

#### 创建数据库模型

```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class HazardRecord(Base):
    __tablename__ = 'hazard_records'
    
    id = Column(Integer, primary_key=True)
    image_path = Column(String(500))
    hazard_type_id = Column(Integer)
    hazard_level_id = Column(Integer)
    bbox_json = Column(String(200))
    reason_text = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

# 创建表
engine = create_engine('postgresql://user:password@localhost/safety_db')
Base.metadata.create_all(engine)
```

#### 保存识别结果

```python
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()

for hazard in hazards:
    record = HazardRecord(
        image_path=image_path,
        hazard_type_id=hazard.hazardTypeId,
        hazard_level_id=hazard.hazardLevelId,
        bbox_json=hazard.bboxJson,
        reason_text=hazard.reasonText
    )
    session.add(record)

session.commit()
```

### 12.6 前端集成示例

#### Vue.js 示例

```vue
<template>
  <div>
    <input type="file" @change="handleFileUpload" accept="image/*" />
    <div v-if="loading">识别中...</div>
    <div v-if="results">
      <h3>识别结果</h3>
      <ul>
        <li v-for="hazard in results" :key="hazard.hazardTypeId">
          {{ getHazardTypeName(hazard.hazardTypeId) }} - 
          {{ hazard.reasonText }}
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      loading: false,
      results: null
    }
  },
  methods: {
    async handleFileUpload(event) {
      const file = event.target.files[0]
      if (!file) return
      
      this.loading = true
      const formData = new FormData()
      formData.append('file', file)
      
      try {
        const response = await fetch('/api/v1/hazard-recognition', {
          method: 'POST',
          body: formData
        })
        const data = await response.json()
        this.results = data.results
      } catch (error) {
        console.error('识别失败:', error)
      } finally {
        this.loading = false
      }
    },
    getHazardTypeName(id) {
      const types = {
        1: '未戴安全帽',
        2: '未穿反光背心',
        // ...
      }
      return types[id] || '未知'
    }
  }
}
</script>
```

---

## 13. 常见问题

### 13.1 安装问题

#### Q: PyTorch CUDA 版本不匹配

**A**: 确认 CUDA 版本：
```bash
nvcc --version
```

安装对应版本的 PyTorch：
```bash
# CUDA 11.8
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### Q: 缺少系统依赖

**A**: 安装 OpenCV 依赖：
```bash
# Ubuntu
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# CentOS
sudo yum install mesa-libGL
```

### 13.2 运行时问题

#### Q: GPU 显存不足 (OOM)

**A**: 
1. 降低图片分辨率：
```python
max_size = 1024  # 从 1280 降到 1024
```

2. 限制显存使用：
```python
torch.cuda.set_per_process_memory_fraction(0.6, 0)  # 从 0.8 降到 0.6
```

3. 使用 CPU 模式：
```bash
export CUDA_VISIBLE_DEVICES=""
```

#### Q: Qwen-VL API 连接超时

**A**: 
1. 检查网络连接：
```bash
python network_diagnosis.py
```

2. 配置代理：
```bash
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

3. 增加超时时间：
```python
response = requests.post(API_URL, json=data, headers=headers, timeout=60)
```

#### Q: JSON 解析失败

**A**: 系统已有容错机制，会自动降级处理。查看日志了解原始输出：
```python
print(f"原始输出: {qwen_text}")
```

### 13.3 性能问题

#### Q: 处理速度慢

**A**: 
1. 使用 GPU 加速
2. 降低图片分辨率
3. 减少 SAM 生成的区域数量：
```python
masks[:3]  # 从 5 个降到 3 个
```

4. 使用较小的 YOLO 模型：
```python
yolov8m.pt  # 或 yolov8s.pt
```

#### Q: 并发请求导致显存溢出

**A**: 
1. 限制并发数
2. 使用队列处理请求
3. 增加请求间隔

```python
import asyncio
from asyncio import Semaphore

semaphore = Semaphore(2)  # 最多 2 个并发

@app.post("/api/v1/hazard-recognition")
async def recognize_hazard(file: UploadFile = File(...)):
    async with semaphore:
        # 处理逻辑
        pass
```

### 13.4 模型问题

#### Q: SAM 分割不准确

**A**: 
1. 调整稳定性分数阈值
2. 使用不同的 SAM 模型（vit_b, vit_l, vit_h）
3. 提供提示点（points）引导分割

#### Q: YOLO 检测不到特定目标

**A**: 
1. 使用更大的模型（yolov8x）
2. 降低置信度阈值
3. 重新训练模型（添加特定类别）

```python
results = model(image_path, conf=0.3, imgsz=640)  # 降低 conf
```

### 13.5 部署问题

#### Q: 服务启动失败

**A**: 检查端口占用：
```bash
# Windows
netstat -ano | findstr :8000

# Linux
lsof -i :8000
```

更换端口：
```python
uvicorn.run(app, host="0.0.0.0", port=8080)
```

#### Q: 跨域问题 (CORS)

**A**: 添加 CORS 中间件：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📚 参考资料

### 官方文档

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Segment Anything Model](https://segment-anything.com/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Qwen-VL API 文档](https://help.aliyun.com/zh/dashscope/developer-reference/qwen-vl-api)
- [PyTorch Documentation](https://pytorch.org/docs/)

### 相关论文

- **SAM**: "Segment Anything" - Kirillov et al., 2023
- **YOLOv8**: "YOLOv8 Technical Report" - Ultralytics, 2023
- **Qwen-VL**: "Qwen-VL: A Versatile Vision-Language Model" - Alibaba, 2023

### 开源项目

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [DashScope SDK](https://github.com/aliyun/alibabacloud-dashscope-python-sdk)

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 提交 Bug

1. 搜索现有 Issue
2. 创建新 Issue，包含：
   - 问题描述
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（OS、Python 版本、GPU 等）
   - 错误日志

### 提交功能请求

1. 描述功能需求
2. 说明使用场景
3. 提供可能的实现思路

### 提交代码

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 风格指南
- 添加必要的注释和文档字符串
- 编写单元测试
- 确保通过所有测试

---

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

## 📞 技术支持

如有问题，请通过以下方式联系：

- **Issue**: [GitHub Issues](https://github.com/your-repo/issues)
- **Email**: support@example.com
- **文档**: 查看本文档和在线 API 文档

---

**最后更新**: 2026-04-23  
**维护者**: Safety AI Team
