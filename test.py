#Scenario1:

img_size_with_cuda = 512
img_size_without_cuda = 128

noise_input = 1

num_steps = 600

style_weight = 0
content_weight = 10000

style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_layers_default = ['conv_2']

#Scenario2:
img_size_with_cuda = 512
img_size_without_cuda = 128

noise_input = 1

num_steps = 600

style_weight = 10000
content_weight = 0

style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_layers_default = ['conv_2']


#Scenario3:
img_size_with_cuda = 512
img_size_without_cuda = 128

noise_input = 1

num_steps = 600

style_weight = 10000
content_weight = 10000

style_layers_default = ['conv_11', 'conv_12', 'conv_13', 'conv_14', 'conv_15']
content_layers_default = ['conv_2']


#Scenario3:
img_size_with_cuda = 512
img_size_without_cuda = 128

noise_input = 1

num_steps = 600

style_weight = 10000
content_weight = 10000

style_layers_default = ['conv_1', 'conv_3', 'conv_5', 'conv_7', 'conv_9']
content_layers_default = ['conv_2']

#Scenario4:
img_size_with_cuda = 512
img_size_without_cuda = 128

noise_input = 0

num_steps = 600

style_weight = 10000
content_weight = 10000

style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_layers_default = ['conv_2']

