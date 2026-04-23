"""
网络诊断工具
用于检查 Qwen-VL API 的网络连接问题
"""

import socket
import requests
import sys


def check_dns():
    """检查 DNS 解析"""
    print("=" * 60)
    print("1. 检查 DNS 解析")
    print("=" * 60)
    
    domains = [
        'dashscope.aliyuncs.com',
        'dashscope.aliyuncs.com.w.kunlun.com',
        'api.aliyuncs.com'
    ]
    
    for domain in domains:
        try:
            addr_info = socket.getaddrinfo(domain, 443)
            print(f"✓ {domain}")
            print(f"  IP 地址：{addr_info[0][4][0]}")
        except Exception as e:
            print(f"✗ {domain}")
            print(f"  错误：{e}")
        print()


def check_network_speed():
    """检查网络连通性"""
    print("=" * 60)
    print("2. 检查网络连通性")
    print("=" * 60)
    
    urls = [
        'https://www.aliyun.com',
        'https://dashscope.aliyuncs.com'
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"✓ {url}")
            print(f"  状态码：{response.status_code}")
            print(f"  响应时间：{response.elapsed.total_seconds():.2f}秒")
        except Exception as e:
            print(f"✗ {url}")
            print(f"  错误：{e}")
        print()


def check_api_key():
    """检查 API Key 有效性"""
    print("=" * 60)
    print("3. 检查 API Key 配置")
    print("=" * 60)
    
    try:
        from qwen_vl_module import API_KEY, API_URL
        
        print(f"API Key: {API_KEY[:10]}...{API_KEY[-10:]}")
        print(f"API URL: {API_URL}")
        
        # 简单验证 API Key 格式
        if API_KEY.startswith("sk-") and len(API_KEY) >= 30:
            print("✓ API Key 格式正确")
        else:
            print("✗ API Key 格式可能不正确")
            
    except Exception as e:
        print(f"✗ 读取配置失败：{e}")
    
    print()


def check_proxy():
    """检查代理设置"""
    print("=" * 60)
    print("4. 检查代理设置")
    print("=" * 60)
    
    import os
    
    proxies = {
        'http': os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy'),
        'https': os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    }
    
    if proxies['http'] or proxies['https']:
        print("检测到代理配置:")
        if proxies['http']:
            print(f"  HTTP 代理：{proxies['http']}")
        if proxies['https']:
            print(f"  HTTPS 代理：{proxies['https']}")
        print("\n提示：如果使用代理，请确保代理服务器正常工作")
    else:
        print("未检测到系统代理配置")
        print("提示：如果需要代理，请设置环境变量:")
        print("  set HTTP_PROXY=http://proxy_server:port")
        print("  set HTTPS_PROXY=http://proxy_server:port")
    
    print()


def provide_solutions():
    """提供解决方案"""
    print("=" * 60)
    print("建议的解决方案")
    print("=" * 60)
    print("""
1. 【检查网络】确保网络连接正常，可以访问阿里云网站
   
2. 【修改 DNS】如果 DNS 解析失败，尝试修改 DNS 服务器：
   - 首选：8.8.8.8 (Google)
   - 备选：1.1.1.1 (Cloudflare)
   - 国内：114.114.114.114
   
3. 【配置代理】如果使用代理上网，设置环境变量：
   set HTTP_PROXY=http://你的代理服务器：端口
   set HTTPS_PROXY=http://你的代理服务器：端口
   
4. 【检查防火墙】确保防火墙没有阻止 Python 访问网络
   
5. 【API Key】确认 API Key 有效且有足够的额度
   
6. 【临时方案】如果网络问题无法解决，可以：
   - 使用本地规则引擎替代云端 API
   - 使用其他多模态模型服务
   - 使用 CPU 模式运行本地模型
    """)


def main():
    """运行所有诊断"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "网络诊断工具" + " " * 29 + "║")
    print("║" + " " * 10 + "Qwen-VL API 连接问题排查" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    check_dns()
    check_network_speed()
    check_api_key()
    check_proxy()
    provide_solutions()
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
    print("\n如果以上检查都失败，请联系网络管理员或阿里云支持")
    print()


if __name__ == "__main__":
    main()
