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
- hazardTypeId: 隐患类型ID（整数，必须从下方隐患类型映射表中选择）
- hazardLevelId: 隐患等级ID（整数，1=一般隐患, 2=较大隐患, 3=重大隐患）
- bboxJson: 隐患位置的边界框，JSON字符串格式 {{"x": x坐标像素值, "y": y坐标像素值, "w": 宽度像素值, "h": 高度像素值}}
- reasonText: 整改建议

【隐患类型映射表】（必须使用左侧的 ID）：
1=未戴安全帽
2=未穿反光背心
3=禁烟区吸烟
4=消防通道堵塞
5=灭火器缺失
6=灭火器过期
7=私拉乱接电线
8=电线裸露
9=缺少安全警示标志
10=物料堆放混乱
11=叉车占用通道
12=危险化学品泄漏
13=缺少防护栏
14=安全网破损
15=梯子使用不规范
16=气瓶摆放不规范
17=未使用个人防护装备
18=工具散落地面
19=疏散通道堵塞
20=危险品存放不规范
21=粉尘堆积严重
22=高处坠落风险
23=叉车碰撞风险
24=人员疲劳作业
25=车辆超速
26=人员聚集风险
27=设备超负荷运行
28=违规操作设备
29=动火作业风险
30=高空坠物风险
31=未经授权进入危险区域
32=车辆倒车盲区风险

【隐患等级映射】：
1=一般隐患（低）
2=较大隐患（中）
3=重大隐患（高）

如果没有发现隐患，返回空数组 []。

示例输出格式：
[
  {{
    "hazardTypeId": 9,
    "hazardLevelId": 1,
    "bboxJson": "{{\"x\":120,\"y\":85,\"w\":350,\"h\":280}}",
    "reasonText": "建议补充安全警示标识，保持现场风险提示清晰可见。"
  }},
  {{
    "hazardTypeId": 1,
    "hazardLevelId": 3,
    "bboxJson": "{{\"x\":520,\"y\":150,\"w\":180,\"h\":200}}",
    "reasonText": "立即要求作业人员佩戴符合标准的安全帽。"
  }}
]

重要提示：
1. 必须返回 hazardTypeId 和 hazardLevelId（整数），不要返回字符串名称
2. bboxJson 必须是转义后的 JSON 字符串，使用像素坐标（不是百分比）
3. x, y 表示左上角坐标，w, h 表示宽度和高度，单位都是像素
4. 请根据图片实际尺寸给出合理的像素坐标值
5. 只返回 JSON 数组，不要添加其他文字说明
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