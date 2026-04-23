# 安全生产隐患识别系统

基于 **SAM（Segment Anything）+ YOLOv8 + Qwen-VL** 三大 AI 模型的智能安全生产隐患三级识别系统。

## 📋 项目简介

本系统通过多模态 AI 技术，自动识别安全生产现场的安全隐患，包括未佩戴安全帽、缺少安全警示标志、违规操作等行为。系统采用三级识别流程：区域分割 → 目标检测 → 隐患推理，实现高精度的安全隐患自动识别。

### 核心功能

- ✅ **区域识别** - SAM 自动分割生产作业区域
- ✅ **目标检测** - YOLOv8 检测人员、设备等目标  
- ✅ **跨模态对齐** - 筛选区域内的有效目标
- ✅ **隐患分析** - Qwen-VL 进行违规行为推理
- ✅ **RESTful API** - 提供标准的 HTTP 接口
- ✅ **交互式命令行** - 支持本地图片测试

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────┐
│           FastAPI Web Service               │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│        Hazard Recognition Pipeline          │
├─────────────────────────────────────────────┤
│  1. SAM 区域分割     (像素级区域识别)       │
│  2. YOLOv8 目标检测  (对象识别与定位)       │
│  3. 跨模态对齐       (区域内目标筛选)       │
│  4. Qwen-VL 推理     (隐患分析与建议)       │
└─────────────────────────────────────────────┘
```

## 📦 依赖环境

### 系统要求

- Python 3.8+
- CUDA 11.x (可选，用于 GPU 加速)
- 网络连接（访问阿里云 Qwen-VL API）

### Python 依赖

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
opencv-python==4.8.1.78
torch==2.1.0
torchvision==0.16.0
ultralytics==8.0.200
segment-anything==1.0
requests==2.31.0
numpy==1.24.3
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd safety_ai

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 准备模型文件

需要下载以下预训练模型并放置在项目根目录：

- **SAM 模型**: `sam_vit_h_4b8939.pth` 
  - 下载地址: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
  
- **YOLOv8 模型**: `yolov8x.pt`
  - 首次运行时会自动下载，或手动从 Ultralytics 官网下载

### 3. 配置 API Key

编辑 `qwen_vl_module.py` 文件，设置你的阿里云 API Key：

```python
API_KEY = "sk-your-api-key-here"
```

获取 API Key: [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)

### 4. 启动服务

#### 方式一：启动 FastAPI 服务（推荐）

```bash
python pipeline.py
```

服务将在 `http://localhost:8000` 启动

#### 方式二：交互式命令行模式

```bash
python pipeline.py --interactive
```

或直接运行后在交互界面输入图片路径进行测试。

## 📡 API 接口文档

### 健康检查

```bash
GET /api/v1/health
```

**响应示例：**
```json
{
  "device": "cuda",
  "gpu_memory_gb": 16.0,
  "sam_loaded": true,
  "yolo_loaded": true
}
```

### 隐患识别

```bash
POST /api/v1/hazard-recognition
Content-Type: multipart/form-data
```

**请求参数：**
- `file`: 上传的图片文件（JPG/PNG，最大 10MB）

**响应示例：**
```json
{
  "status": "success",
  "message": "隐患识别完成",
  "data": {
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
  },
  "summary": {
    "regions_count": 0,
    "objects_count": 0,
    "valid_objects_count": 0,
    "hazards_count": 2
  }
}
```

### 在线文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 测试

### 使用测试脚本

```bash
python test_api.py
```

该脚本会自动测试：
1. 健康检查接口
2. 隐患识别接口（需要 `test.jpg` 文件）
3. 无效文件格式处理

### 手动测试

```bash
curl -X POST "http://localhost:8000/api/v1/hazard-recognition" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```

## 🔧 工具脚本

### 网络诊断工具

如果遇到 Qwen-VL API 连接问题，运行：

```bash
python network_diagnosis.py
```

该工具会检查：
- DNS 解析
- 网络连通性
- API Key 配置
- 代理设置
- 提供解决方案

## 📊 支持的隐患类型

系统可识别 32 种安全隐患类型：

| ID | 隐患类型 | ID | 隐患类型 |
|----|---------|----|---------|
| 1 | 未戴安全帽 | 17 | 未使用个人防护装备 |
| 2 | 未穿反光背心 | 18 | 工具散落地面 |
| 3 | 禁烟区吸烟 | 19 | 疏散通道堵塞 |
| 4 | 消防通道堵塞 | 20 | 危险品存放不规范 |
| 5 | 灭火器缺失 | 21 | 粉尘堆积严重 |
| 6 | 灭火器过期 | 22 | 高处坠落风险 |
| 7 | 私拉乱接电线 | 23 | 叉车碰撞风险 |
| 8 | 电线裸露 | 24 | 人员疲劳作业 |
| 9 | 缺少安全警示标志 | 25 | 车辆超速 |
| 10 | 物料堆放混乱 | 26 | 人员聚集风险 |
| 11 | 叉车占用通道 | 27 | 设备超负荷运行 |
| 12 | 危险化学品泄漏 | 28 | 违规操作设备 |
| 13 | 缺少防护栏 | 29 | 动火作业风险 |
| 14 | 安全网破损 | 30 | 高空坠物风险 |
| 15 | 梯子使用不规范 | 31 | 未经授权进入危险区域 |
| 16 | 气瓶摆放不规范 | 32 | 车辆倒车盲区风险 |

### 隐患等级

- **一般隐患** (ID: 1) - 低风险，建议整改
- **较大隐患** (ID: 2) - 中风险，需及时整改
- **重大隐患** (ID: 3) - 高风险，立即整改

## ⚙️ 配置说明

### 内存优化

系统已内置内存优化策略：

```python
# GPU 显存限制（默认 80%）
torch.cuda.set_per_process_memory_fraction(0.8, 0)

# 图片缩放（最大边长 1280px）
max_size = 1280
```

如需调整，编辑 `pipeline.py` 中的相关配置。

### 设备选择

系统自动检测并使用 GPU（如果可用），否则使用 CPU：

```python
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
```

## 📁 项目结构

```
safety_ai/
├── pipeline.py                  # 主程序 & API 服务
├── qwen_vl_module.py            # Qwen-VL API 调用模块
├── network_diagnosis.py         # 网络诊断工具
├── test_api.py                  # API 测试脚本
├── requirements.txt             # Python 依赖
├── sam_vit_h_4b8939.pth         # SAM 模型文件
├── yolov8x.pt                   # YOLOv8 模型文件
├── segment-anything-main/       # SAM 源码
├── result_*.json                # 识别结果文件
└── test.jpg                     # 测试图片
```

## 🐛 常见问题

### 1. GPU 显存不足

**解决方案：**
- 降低图片分辨率（修改 `pipeline.py` 中的 `max_size`）
- 减少同时处理的图片数量
- 使用 CPU 模式（设置环境变量 `CUDA_VISIBLE_DEVICES=""`）

### 2. Qwen-VL API 连接失败

**解决方案：**
```bash
# 运行网络诊断
python network_diagnosis.py

# 检查代理设置
echo $HTTP_PROXY
echo $HTTPS_PROXY

# 配置代理（如需要）
export HTTP_PROXY=http://proxy_server:port
export HTTPS_PROXY=http://proxy_server:port
```

### 3. 模型加载缓慢

**解决方案：**
- 首次加载需要时间，后续请求会使用缓存的模型
- 确保模型文件完整且路径正确
- 使用 SSD 存储模型文件

### 4. JSON 解析失败

Qwen-VL 返回的内容可能包含格式问题，系统已内置容错解析机制。如需调试，查看控制台输出的原始内容。

## 📝 开发指南

### 添加新的隐患类型

1. 在 `pipeline.py` 中更新 `HAZARD_TYPE_MAP`：

```python
HAZARD_TYPE_MAP = {
    # ... 现有类型
    "新隐患类型": 33,  # 分配新的 ID
}
```

2. 在 `qwen_vl_module.py` 的 prompt 中添加新类型到可选值列表

### 自定义安全规则

编辑 `pipeline.py` 中的 `safety_check` 函数：

```python
rules = """
1. 作业区域内人员必须佩戴安全帽，未佩戴属于重大隐患
2. 禁止在作业区域吸烟、使用明火
3. 添加你的自定义规则...
"""
```

### 扩展 API 功能

在 `pipeline.py` 中添加新的路由：

```python
@app.post("/api/v1/custom-endpoint")
async def custom_endpoint():
    # 你的业务逻辑
    pass
```

## 📄 许可证

本项目遵循 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

如有问题或建议，请提交 Issue 或通过以下方式联系：

- 项目地址: [GitHub Repository]
- 邮箱: [your-email@example.com]

## 🙏 致谢

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) - Facebook AI Research
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics
- [Qwen-VL](https://help.aliyun.com/zh/dashscope/developer-reference/qwen-vl-api) - 阿里云通义千问
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架

---

**注意**: 使用本系统前，请确保已获得阿里云 DashScope API 的使用权限，并遵守相关法律法规和安全规范。
