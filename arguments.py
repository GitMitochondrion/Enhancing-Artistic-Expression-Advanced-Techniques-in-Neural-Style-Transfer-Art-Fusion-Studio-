# 图片的大小（n*n）
img_size_with_cuda = 512
img_size_without_cuda = 128

# 是否使用白噪声作为输入图片，一般会有噪点；若为0则使用content_image作为输入图片
noise_input = 0

# 训练步长，可以适当增加以便充分训练
num_steps = 600

# 风格和内容的权重，需要重点调试
style_weight = 10000
content_weight = 10000

# 风格和内容在卷积层上的定义，可以调整内容层的位置
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_layers_default = ['conv_2']
# content_layers_default = ['conv_4']
