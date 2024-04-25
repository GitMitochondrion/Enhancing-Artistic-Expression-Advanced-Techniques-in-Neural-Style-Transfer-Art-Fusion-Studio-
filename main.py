from __future__ import print_function

import arguments as args
import classes as cls
import torch
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms as transforms
import torchvision.models as models


matplotlib.use('TkAgg')

# 调用gpu，没有则调用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出图像的所需尺寸
imsize = args.img_size_with_cuda if torch.cuda.is_available() else args.img_size_without_cuda  # 如果没有GPU，使用小尺寸

# 设定加载器：缩放到设定的尺寸，并转换为tensor
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

# 加载图片
style_img = cls.image_loader("C:/Users/teyal/Downloads/Final Project/002.png", loader, device)
content_img = cls.image_loader("C:/Users/teyal/Downloads/Final Project/001.jpg", loader, device)
assert style_img.size() == content_img.size(), \
    "请导入相同大小的样式和内容图像"

# 设定逆向加载器：重新转换为PIL图像
unloader = transforms.ToPILImage()

# 开启交互模式
plt.ion()

# 显示style和content图像
plt.figure()
cls.imshow(style_img, unloader, title='Style Image')
plt.figure()
cls.imshow(content_img, unloader, title='Content Image')

# 加载预训练过的vgg模型
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)

cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 设定输入图像
if args.noise_input == 1:
    input_img = content_img.clone()
else:
    input_img = torch.randn(content_img.data.size(), device=device)

# 将原始输入图像添加到图中：
plt.figure()
cls.imshow(input_img, unloader, title='Input Image')

# 风格迁移并得到输出图像
output = cls.run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, device)

plt.figure()
cls.imshow(output, unloader, title='Output Image')

# 关闭交互模式
plt.ioff()
plt.show()
