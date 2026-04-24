"""
批量识别功能快速演示脚本
展示如何使用三种不同的方式进行批量识别
"""
import os
import sys


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_requirements():
    """检查必要的环境"""
    print("🔍 检查环境要求...")
    
    # 检查 Python 版本
    import sys
    if sys.version_info < (3, 8):
        print("❌ Python 版本过低，需要 3.8 或更高版本")
        return False
    
    print(f"✅ Python 版本: {sys.version.split()[0]}")
    
    # 检查必要的库
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("❌ 未安装 PyTorch，请运行: pip install torch")
        return False
    
    try:
        import fastapi
        print(f"✅ FastAPI 版本: {fastapi.__version__}")
    except ImportError:
        print("❌ 未安装 FastAPI，请运行: pip install fastapi uvicorn")
        return False
    
    try:
        import requests
        print(f"✅ Requests 版本: {requests.__version__}")
    except ImportError:
        print("❌ 未安装 Requests，请运行: pip install requests")
        return False
    
    print("\n✅ 所有依赖检查通过！\n")
    return True


def demo_command_line():
    """演示命令行批量模式"""
    print_header("方式一：命令行批量模式")
    
    print("使用步骤：")
    print("1. 准备图片文件夹")
    print("   mkdir test_images")
    print("   # 将图片放入 test_images 文件夹\n")
    
    print("2. 运行批量识别命令")
    print("   python pipeline.py --batch ./test_images\n")
    
    print("3. 查看结果")
    print("   - 控制台显示实时进度")
    print("   - 自动生成 batch_result_*.json 文件\n")
    
    print("💡 提示：这是最简单的使用方式，适合本地批量处理\n")


def demo_api_mode():
    """演示 API 批量模式"""
    print_header("方式二：API 批量接口")
    
    print("使用步骤：")
    print("1. 启动 API 服务")
    print("   python pipeline.py --api\n")
    
    print("2. 访问 API 文档")
    print("   浏览器打开: http://127.0.0.1:8000/docs\n")
    
    print("3. 测试批量接口")
    print("   在 Swagger UI 中找到 /api/v1/batch-hazard-recognition")
    print("   点击 'Try it out'，上传多张图片进行测试\n")
    
    print("4. 或使用 cURL 测试")
    print('   curl -X POST "http://127.0.0.1:8000/api/v1/batch-hazard-recognition" \\')
    print('     -F "files=@image1.jpg" \\')
    print('     -F "files=@image2.jpg"\n')
    
    print("💡 提示：适合集成到其他系统或远程调用\n")


def demo_test_script():
    """演示测试脚本"""
    print_header("方式三：批量测试脚本")
    
    print("使用步骤：")
    print("1. 准备图片文件夹")
    print("   mkdir test_images")
    print("   # 将图片放入 test_images 文件夹\n")
    
    print("2. 启动 API 服务（新终端）")
    print("   python pipeline.py --api\n")
    
    print("3. 运行测试脚本")
    print("   python test_batch_api.py -f ./test_images\n")
    
    print("4. 查看结果")
    print("   - 控制台显示详细统计")
    print("   - 自动生成 batch_result_*.json 文件\n")
    
    print("💡 提示：适合自动化测试和批量验证\n")


def create_sample_folder():
    """创建示例文件夹结构"""
    print_header("创建示例文件夹")
    
    folder_name = "test_images"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ 已创建文件夹: {folder_name}/")
        print(f"\n请将需要识别的图片放入该文件夹")
        print(f"支持的格式: JPG, JPEG, PNG\n")
    else:
        print(f"⚠️  文件夹已存在: {folder_name}/")
        
        # 统计现有图片
        image_count = len([f for f in os.listdir(folder_name) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   当前包含 {image_count} 张图片\n")


def show_comparison():
    """对比三种方式"""
    print_header("三种方式对比")
    
    print("┌──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│    特性      │  命令行模式  │   API 模式   │  测试脚本    │")
    print("├──────────────┼──────────────┼──────────────┼──────────────┤")
    print("│  使用难度    │    ⭐         │    ⭐⭐⭐     │    ⭐⭐       │")
    print("│  灵活性      │    ⭐⭐       │    ⭐⭐⭐⭐⭐   │    ⭐⭐⭐     │")
    print("│  自动化程度  │    ⭐⭐⭐     │    ⭐⭐⭐⭐⭐   │    ⭐⭐⭐⭐   │")
    print("│  适用场景    │ 本地批量处理 │ 系统集成     │ 测试验证     │")
    print("│  网络要求    │ 不需要       │ 需要         │ 需要         │")
    print("│  结果保存    │ 自动 JSON    │ 手动处理     │ 自动 JSON    │")
    print("└──────────────┴──────────────┴──────────────┴──────────────┘\n")


def main():
    """主函数"""
    print("\n" + "🎯" * 35)
    print(" " * 20 + "安全生产隐患识别系统")
    print(" " * 25 + "批量识别功能演示")
    print("🎯" * 35)
    
    # 检查环境
    if not check_requirements():
        print("\n❌ 环境检查失败，请先安装必要的依赖")
        sys.exit(1)
    
    # 创建示例文件夹
    create_sample_folder()
    
    # 演示三种方式
    demo_command_line()
    demo_api_mode()
    demo_test_script()
    
    # 对比表格
    show_comparison()
    
    # 快速开始指南
    print_header("🚀 快速开始")
    
    print("推荐新手使用流程：\n")
    print("1️⃣  准备测试图片")
    print("   将图片放入 test_images/ 文件夹\n")
    
    print("2️⃣  运行命令行批量模式")
    print("   python pipeline.py --batch ./test_images\n")
    
    print("3️⃣  查看生成的 JSON 结果文件")
    print("   文件名格式: batch_result_YYYYMMDD_HHMMSS.json\n")
    
    print("📖 更多详细信息请查看: BATCH_RECOGNITION_GUIDE.md\n")
    
    print("=" * 70)
    print("  祝您使用愉快！如有问题请参考文档或提交 Issue")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
