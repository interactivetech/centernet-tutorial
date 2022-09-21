import torch
from torchsummary import torchsummary
import torch.nn as nn
import torchvision
# from model4 import centernet
# We also need to replace Mobilenet's ReLU6 activations with ReLU. 
# There is no noticeable difference in quality, but this will
# allow us to use CoreML for mobile inference on iOS devices.
def replace_relu6_with_relu(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_relu6_with_relu(model=module)
        if isinstance(module, nn.ReLU6):
            model._modules[name] = nn.ReLU()
    return model

def init_conv_layers(layer):
    for m in layer.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.normal_(m.weight,std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)

class EfficientCenternet(nn.Module):
    def __init__(self,num_classes = 2):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(width_mult=0.5)
        self.num_classes = num_classes
        # We reuse state dict from mobilenet v2 width width_mult == 1.0.
        # This is not the optimal way to use pretrained models, but in this case
        # it gives us good initialization for faster convergence.
        state_dict = torchvision.models.mobilenet_v2(pretrained=True).state_dict()
        target_dict = mobilenet.state_dict()
        for k in target_dict.keys():
            if len(target_dict[k].size()) == 0:
                continue
            state_dict[k] = state_dict[k][:target_dict[k].size(0)]
            if len(state_dict[k].size()) > 1:
                state_dict[k] = state_dict[k][:, :target_dict[k].size(1)]

        mobilenet.load_state_dict(state_dict)

        # weight = mobilenet.features[0][0].weight.detach()
        # mobilenet.features[0][0].weight = nn.Parameter(data=weight / 255.)

        mobilenet = replace_relu6_with_relu(mobilenet)

        self.features = mobilenet.features[:-2]
        self.upscale0 = nn.Sequential(
            nn.Conv2d(80, 48, 1, 1, 0, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        init_conv_layers(self.upscale0)
        self.upscale1 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        init_conv_layers(self.upscale1)
        self.upscale2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        init_conv_layers(self.upscale2)
        self.upscale3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        init_conv_layers(self.upscale3)
        self.upscale4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        init_conv_layers(self.upscale4)
        # self.upscale5 = nn.Conv2d(4, 1, 3, 1, 1, bias=True)
        self.outc = nn.Conv2d(4, self.num_classes, 3, 1, 1, bias=True)
        self.outr = nn.Conv2d(4, 2, 3, 1, 1, bias=True)
        self.outc.bias.data.fill_(-2.19)
        # output residue
        self.outr.bias.data.fill_(-2.19)

        # self.outc = 
    def forward(self, x):
        out = x
        skip_outs = []
        for i in range(len(self.features)):
            out = self.features[i](out)
            if i in {1, 3, 6, 13}:
                skip_outs.append(out)
        out = self.upscale0(out)
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale1(out + skip_outs[3])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale2(out + skip_outs[2])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale3(out + skip_outs[1])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale4(out + skip_outs[0])
        # print("out.shape: ",out.shape )
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        # out = self.upscale5(out)
        out_c = self.outc(out)
        out_r = self.outr(out)
        # print("out_c.shape: ",out_c.shape )
        # print("out_r.shape: ",out_r.shape )
        return out_c, out_r
class EfficientCenternet2(nn.Module):
    def __init__(self,num_classes = 2):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(width_mult=0.5)
        self.num_classes = num_classes
        # We reuse state dict from mobilenet v2 width width_mult == 1.0.
        # This is not the optimal way to use pretrained models, but in this case
        # it gives us good initialization for faster convergence.
        state_dict = torchvision.models.mobilenet_v2(pretrained=True).state_dict()
        target_dict = mobilenet.state_dict()
        for k in target_dict.keys():
            if len(target_dict[k].size()) == 0:
                continue
            state_dict[k] = state_dict[k][:target_dict[k].size(0)]
            if len(state_dict[k].size()) > 1:
                state_dict[k] = state_dict[k][:, :target_dict[k].size(1)]

        mobilenet.load_state_dict(state_dict)

        # weight = mobilenet.features[0][0].weight.detach()
        # mobilenet.features[0][0].weight = nn.Parameter(data=weight / 255.)

        mobilenet = replace_relu6_with_relu(mobilenet)

        self.features = mobilenet.features[:-2]
        self.upscale0 = nn.Sequential(
            nn.Conv2d(80, 48, 1, 1, 0, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        init_conv_layers(self.upscale0)
        self.upscale1 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        init_conv_layers(self.upscale1)
        self.upscale2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        init_conv_layers(self.upscale2)
        self.upscale3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        init_conv_layers(self.upscale3)
        self.upscale4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        init_conv_layers(self.upscale4)
        # self.upscale5 = nn.Conv2d(4, 1, 3, 1, 1, bias=True)
        self.outc = nn.Conv2d(8, self.num_classes, 3, 1, 1, bias=True)
        self.outr = nn.Conv2d(8, 2, 3, 1, 1, bias=True)
        self.outc.bias.data.fill_(-2.19)
        # output residue
        self.outr.bias.data.fill_(-2.19)

        # self.outc = 
    def forward(self, x):
        out = x
        skip_outs = []
        for i in range(len(self.features)):
            out = self.features[i](out)
            if i in {1, 3, 6, 13}:
                skip_outs.append(out)
        out = self.upscale0(out)
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale1(out + skip_outs[3])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale2(out + skip_outs[2])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale3(out + skip_outs[1])
        # out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        # out = self.upscale4(out + skip_outs[0])
        # print("out.shape: ",out.shape )
        # out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        # out = self.upscale5(out)
        out_c = self.outc(out)
        out_r = self.outr(out)
        # print("out_c.shape: ",out_c.shape )
        # print("out_r.shape: ",out_r.shape )
        return out_c, out_r
if __name__ == '__main__':
    model = EfficientCenternet()
    print(torchsummary.summary(model,input_size=(3,256,256)))
    out_c, out_r = model(torch.rand(1,3,256,256))
    print(out_c.shape)

    model = EfficientCenternet2()
    print(torchsummary.summary(model,input_size=(3,256,256)))
    out_c, out_r = model(torch.rand(1,3,256,256))
    print(out_c.shape)
    # model = centernet(2,model_name='mv2')
    # print(torchsummary.summary(model,input_size=(3,256,256)))
    # outc, outr = model(torch.rand(1,3,256,256))
    # print(outc.shape)

