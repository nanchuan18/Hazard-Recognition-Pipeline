import requests
import base64
import socket

# 填入你从阿里云获取的 API KEY
API_KEY = "sk-024e17b7809046f28fa883e762b4e450"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def check_network():
    """检查网络连接状态"""
    try:
        # 尝试解析域名
        socket.getaddrinfo('dashscope.aliyuncs.com', 443)
        return True, "网络连接正常"
    except Exception as e:
        return False, f"网络异常：{str(e)}"


def qwen_vl_infer(image_path, regions, objects, safety_rules):
    # 首先检查网络连接
    network_ok, network_msg = check_network()
    if not network_ok:
        print(f"⚠️ {network_msg}")
        print("⚠️ 无法连接到阿里云 Qwen-VL API，将返回预设的安全提示")
        
        # 返回预设的安全提示
        return f"""
⚠️ **网络提示**: 无法连接到 Qwen-VL API ({network_msg})

**检测到的目标**:
{chr(10).join([f'- {obj["object_name"]} (置信度：{obj["confidence"]})' for obj in objects])}

**建议检查**:
1. 检查网络连接是否正常
2. 检查 DNS 设置
3. 检查防火墙是否阻止访问
4. 如果使用代理，请确保代理配置正确

**基础安全提示**:
- 请确保作业人员佩戴安全帽
- 禁止在作业区域吸烟或使用明火
- 注意现场安全规范
        """

    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        return f"图片读取失败：{str(e)}"

    prompt = f"""
你是安全生产隐患识别 AI。
区域：{regions}
目标：{objects}
规则：{safety_rules}

请分析图片中的安全隐患，并以 JSON 数组格式返回结果。
每个隐患对象包含以下字段：
- hazardTypeName: 隐患类型名称（必须从下方隐患类型可选值中选择）
- hazardLevelName: 隐患等级（必须从下方隐患等级可选值中选择）
- bboxJson: 隐患位置的边界框，JSON字符串格式 {{"x": x坐标百分比, "y": y坐标百分比, "w": 宽度百分比, "h": 高度百分比}}
- advice: 整改建议

隐患类型可选值：未戴安全帽、未穿反光背心、禁烟区吸烟、消防通道堵塞、灭火器缺失、灭火器过期、私拉乱接电线、电线裸露、缺少安全警示标志、物料堆放混乱、叉车占用通道、危险化学品泄漏、缺少防护栏、安全网破损、梯子使用不规范、气瓶摆放不规范、未使用个人防护装备、工具散落地面、疏散通道堵塞、危险品存放不规范、粉尘堆积严重、高处坠落风险、叉车碰撞风险、人员疲劳作业、车辆超速、人员聚集风险、设备超负荷运行、违规操作设备、动火作业风险、高空坠物风险、未经授权进入危险区域、车辆倒车盲区风险

隐患等级可选值：高（重大隐患）、中（较大隐患）、低（一般隐患）

如果没有发现隐患，返回空数组 []。

示例输出格式：
[
  {{
    "hazardTypeName": "缺少安全警示标志",
    "hazardLevelName": "低",
    "bboxJson": "{{\"x\":31,\"y\":43,\"w\":25,\"h\":18}}",
    "advice": "建议补充安全警示标识，保持现场风险提示清晰可见。"
  }}
]

请只返回 JSON 数组，不要添加其他文字说明。
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "qwen-vl-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{image_base64}"},
                        {"text": prompt},
                    ],
                }
            ]
        },
        "parameters": {"result_format": "message"},
    }

    try:
        print("正在调用 Qwen-VL API...")
        response = requests.post(API_URL, json=data, headers=headers, timeout=30)
        
        print(f"Qwen 返回状态码：{response.status_code}")
        
        if response.status_code == 200:
            res = response.json()
            return res["output"]["choices"][0]["message"]["content"][0]["text"]
        elif response.status_code == 401:
            return "API Key 验证失败，请检查 API_KEY 是否正确"
        elif response.status_code == 429:
            return "API 调用频率超限，请稍后重试"
        else:
            return f"API 调用失败 (状态码：{response.status_code})：{response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return "API 调用超时，请检查网络连接或稍后重试"
    except requests.exceptions.ConnectionError as e:
        return f"网络连接失败：{str(e)}\n请检查网络连接和 DNS 设置"
    except Exception as e:
        return f"调用失败：{str(e)}\n返回内容：{response.text if 'response' in locals() else '无'}"