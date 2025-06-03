import os
from PIL import Image
from torchvision import transforms
import argparse

def resize_images(input_dir, output_dir, target_size=(256, 256)):
    """
    调整文件夹内所有图像到指定尺寸
    
    参数:
        input_dir (str): 输入图像文件夹路径
        output_dir (str): 输出图像文件夹路径
        target_size (tuple): 目标尺寸 (width, height)
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义图像转换流程
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img)
    ])
    
    # 处理所有图像文件
    processed = 0
    for filename in os.listdir(input_dir):
        # 只处理常见图像格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # 打开图像
                img_path = os.path.join(input_dir, filename)
                with Image.open(img_path) as img:
                    # 调整尺寸并保存
                    output_path = os.path.join(output_dir, filename)
                    transform(img).save(output_path)
                processed += 1
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
    
    print(f"处理完成！共转换 {processed} 张图像到 {target_size} 分辨率")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量调整图像尺寸')

    parser.add_argument('--size', type=int, default=256, help='目标分辨率（默认256）')
    
    args = parser.parse_args()
    
    # 执行调整
    resize_images(
        input_dir=f'B',
        output_dir=f'BB',
        target_size=(args.size, args.size)
    )