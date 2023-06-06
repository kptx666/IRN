from FLOPs.profile import profile
from model import ConvNeXt

# width = 720 // 4
# height = 1280 // 4
model = ConvNeXt(upscale_factor=4)
#
# flops, params = profile(model, input_size=(1, 3, height, width))
# print('MSR: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format(height, width, flops / (1e9), params))
# from ptflops import get_model_complexity_info
#
# macs, params = get_model_complexity_info(model, (3, height, width), as_strings=True, print_per_layer_stat=True, verbose=True)
# print(macs)
# print(params)
# from torchstat import stat
#
# stat(model, (3, height, width))
# from torchsummaryX import summary
# import torch
#
# img = torch.zeros((1, 3, height, width), device=next(model.parameters()).device)
# summary(model, img)
import torch
from torchsummaryX import summary

# summary(model, torch.zeros((1, 3, 32, 32)))

# input LR x2, HR size is 720p
# summary(model, torch.zeros((1, 3, 640, 360)))

# input LR x3, HR size is 720p
# summary(model, torch.zeros((1, 3, 426, 240)))

# input LR x4, HR size is 720p
# summary(model, torch.zeros((1, 3, 320, 180)))
from thop import profile
import torch
input = torch.zeros(1, 3, 640, 360)
flops, params = profile(model, inputs=(input,))
print(flops, params)
# 71688544824.0
# 49593079112.0
# from torchsummary import summary
# summary(model, input_size=(3,320,180), batch_size=-1)
# def print_network(net):
#     num_params = 0
#     for param in net.parameters():
#         num_params += param.numel()
#     print(net)
#     print('Total number of parameters: %d' % num_params)
# print_network(model)
from model_summary import get_model_flops
flops = get_model_flops(model, (3, 320, 180), False)
flops = flops / 10 ** 9
print('{}G'.format(flops))