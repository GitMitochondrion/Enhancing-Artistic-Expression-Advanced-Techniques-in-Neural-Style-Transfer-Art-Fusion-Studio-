from __future__ import print_function

import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

import arguments as args

style_weight = 1000000


def image_loader(image_name, loader, device):
    image = Image.open(image_name)
    image = image.convert("RGB")
    image = loader(image).unsqueeze(0)  # 需要伪造的批次尺寸以适合网络的输入尺寸:(3, x, x) --> (1, 3, x, x)
    return image.to(device, torch.float)


def imshow(tensor, unloader, title=None):
    image = tensor.cpu().clone()  # 克隆张量不对其进行更改
    image = image.squeeze(0)  # 删除假批次尺寸(1, 3, x, x) --> (3, x, x)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 稍停一下，以便更新


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()

        # 将目标内容与所使用的树“分离”
        # 动态计算梯度：这是一个规定值，
        # 不是变量。 否则将引发错误。
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=特征图数量
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # 将FXML调整为\ hat FXML

    G = torch.matmul(features, features.t())

    # 将gram矩阵的值“规范化”
    # 除以每个要素图中的元素数量。
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        g = gram_matrix(input)
        self.loss = F.mse_loss(g, self.target)
        return input


# 创建一个模块来标准化输入图像，以便可以轻松地将其放入
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # 查看均值和标准差以使其为[C x 1 x 1]，以便它们可以
        # 直接使用形状为[B x C x H x W]的图像张量。
        # B是批量大小。 C是通道数。 H是高度，W是宽度。
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)

        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# 所需的深度层以计算样式/内容损失：
content_layers_default = args.content_layers_default
style_layers_default = args.style_layers_default


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, device="cpu",
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # 标准化模块
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # 只是为了获得对内容/样式的可迭代访问或列表
    # losses
    content_losses = []
    style_losses = []

    # 假设cnn是nn.Sequential，那么我们创建一个新的nn.Sequential
    # 放入应该顺序激活的模块
    model = nn.Sequential(normalization)
    
    i = 0  # 每当转换时就增加
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):  # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # 增加内容损失:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # 增加样式损失:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 舍弃无需计算的额外层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # 此行显示输入的图片是待优化的参数, 并使用拟牛顿法来进行参数优化
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, style_weight, device="cpu", num_steps=args.num_steps,
                       content_weight=args.content_weight):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img, device)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # 更正更新后的输入图像的值
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # 最后更正...
    input_img.data.clamp_(0, 1)

    return input_img
