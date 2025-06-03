import argparse
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import torch.nn as nn
from model.GAN import *
# 配置参数（与训练时一致）
parser = argparse.ArgumentParser()
parser.add_argument("--img_height", type=int, default=256)
parser.add_argument("--img_width", type=int, default=256)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--n_residual_blocks", type=int, default=9)
opt = parser.parse_args([])


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.cuda() if torch.cuda.is_available() else image

# 主函数
def style_transfer_B_to_A(input_dir, output_dir, model_path):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预训练模型
    G_BA = GeneratorResNet((opt.channels, opt.img_height, opt.img_width), opt.n_residual_blocks)
    G_BA.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    G_BA.eval()
    
    if torch.cuda.is_available():
        G_BA = G_BA.cuda()

    # 处理目录中所有图像
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 加载图像
                input_path = os.path.join(input_dir, filename)
                real_B = load_image(input_path)
                
                # 风格迁移
                with torch.no_grad():
                    fake_A = G_BA(real_B)
                
                # 保存结果（反归一化）
                output_path = os.path.join(output_dir, f"styled_{filename}")
                save_image(fake_A*0.5 + 0.5, output_path, normalize=False)
                
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_dir = "./A_B"    # 存放B风格图像的目录
    output_dir = "./A_B_A" # 输出目录
    model_path = "save/facades/G_BA_99.pth"  # 训练好的G_BA模型路径
    
    style_transfer_B_to_A(input_dir, output_dir, model_path)