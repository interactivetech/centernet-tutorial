import torch
from torchsummary import torchsummary
import torch.nn as nn
import torchvision

class Conv1x1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1x1, self).__init__()
        self.conv1x1 =  nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(output_channels),
                                      nn.ReLU(inplace=True)
                                     )
    
    def forward(self, x):
        return self.conv1x1(x)

class Head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Head, self).__init__()
        self.conv5x5 = nn.Sequential(nn.Conv2d(input_channels, input_channels, 5, 1, 2, groups = input_channels, bias = True),
                                     nn.BatchNorm2d(input_channels),
                                     nn.ReLU(inplace=True),
                                    ) 
        init_conv_layers(self.conv5x5)
        self.conv_f = nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=True)
        # self.bn = nn.BatchNorm2d(output_channels)
        self.conv_f.bias.data.fill_(-2.19)
    def forward(self, x):
        return self.conv_f(self.conv5x5(x))
class SPP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPP, self).__init__()
        self.Conv1x1 = Conv1x1(input_channels, output_channels)

        self.S1 =  nn.Sequential(nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True)
                                 )

        self.S2 =  nn.Sequential(nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True)
                                 )

        self.S3 =  nn.Sequential(nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups = output_channels, bias = False),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True)
                                 )

        self.output = nn.Sequential(nn.Conv2d(output_channels * 3, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                   )
                                   
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):   
        x = self.Conv1x1(x)#[2, 336, 22, 22]->[2, 96, 22, 22]

        y1 = self.S1(x)#[2, 96, 22, 22] ->[2, 96, 22, 22]
        y2 = self.S2(x)#[2, 96, 22, 22]->[2, 96, 22, 22]
        y3 = self.S3(x)#[2, 96, 22, 22]->[2, 96, 22, 22]

        y = torch.cat((y1, y2, y3), dim=1)#[2, 96, 22, 22]->[2, 288, 22, 22]
        y = self.relu(x + self.output(y))# [2, 288, 22, 22]->[2, 96, 22, 22]

        return y


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
            nn.Conv2d(80, 48, 1, 1, 0, bias=True),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        init_conv_layers(self.upscale0)
        self.upscale1 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        init_conv_layers(self.upscale1)
        self.upscale2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        init_conv_layers(self.upscale2)
        self.upscale3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        init_conv_layers(self.upscale3)
        self.upscale4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1, bias=True),
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
class EfficientCenternet3(nn.Module):
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
            nn.Conv2d(80, 48, 1, 1, 0, bias=True),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        init_conv_layers(self.upscale0)
        self.upscale1 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        init_conv_layers(self.upscale1)
        self.upscale2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        init_conv_layers(self.upscale2)
        self.upscale3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        init_conv_layers(self.upscale3)
        self.upscale4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1, bias=True),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        init_conv_layers(self.upscale4)
        # self.upscale5 = nn.Conv2d(4, 1, 3, 1, 1, bias=True)
        self.outc1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride = 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride = 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride = 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        init_conv_layers(self.outc1)
        self.outc= nn.Conv2d(64, self.num_classes, 3, padding= 1, bias=True)
        # in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t | str = 0
        self.outr1 =  nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64,kernel_size= 3,stride = 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size= 3,stride = 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,kernel_size= 3,stride = 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        init_conv_layers(self.outr1)

        self.outr = nn.Conv2d(64, 2, 3, stride = 1,padding= 1, bias=True)
            
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
        print([i.shape for i in skip_outs])
        # [torch.Size([1, 8, 128, 128]), torch.Size([1, 16, 64, 64]), torch.Size([1, 16, 32, 32]), torch.Size([1, 48, 16, 16])]
        out_channels = [128,64,32,16]
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
        print(out.shape)
        # out = self.upscale5(out)
        out_c1 = self.outc1(out)
        out_r1 = self.outr1(out)
        out_c = self.outc(out_c1)
        out_r = self.outr(out_r1)
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
class EfficientCenterDet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes = num_classes
        mobilenet = torchvision.models.mobilenet_v2(width_mult=0.5)

        state_dict = torchvision.models.mobilenet_v2(pretrained=True).state_dict()
        target_dict = mobilenet.state_dict()
        for k in target_dict.keys():
            if len(target_dict[k].size()) == 0:
                continue
            state_dict[k] = state_dict[k][:target_dict[k].size(0)]
            if len(state_dict[k].size()) > 1:
                state_dict[k] = state_dict[k][:, :target_dict[k].size(1)]

        mobilenet.load_state_dict(state_dict)

        # mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)
        mobilenet = replace_relu6_with_relu(mobilenet)
        # print("mobilenet.featuresL ", len(mobilenet.features))
        self.features = mobilenet.features[:14]
        # print("mobilenet.featuresL ", len(mobilenet.features))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_out_channels = [16,16,48]

        self.SPP = SPP(sum(self.stage_out_channels),128)
        init_conv_layers(self.SPP)
        self.conv1x1_1 = Conv1x1(128,128)
        init_conv_layers(self.conv1x1_1)
        self.conv1x1_2 = Conv1x1(128,128)
        init_conv_layers(self.conv1x1_2)

        self.conv1x1_3 = Conv1x1(128,128)
        init_conv_layers(self.conv1x1_3)

        self.cls_head  = Head(128,self.num_classes)
        self.reg_head = Head(128,2)

    def forward(self,x):
        '''
        torch.Size([1, 16, 64, 64])
        torch.Size([1, 16, 32, 32])
        torch.Size([1, 48, 16, 16])
        '''
        skip_outs = []
        out = x
        for i in range(len(self.features)):
            out = self.features[i](out)
            # print(i," out: ",out.shape)
            if i in { 3, 6}:
                skip_outs.append(out)
        P1 = self.avg_pool(skip_outs[0]) #[1, 16, 64, 64] -> [1, 16, 32, 32]
        P3 = self.upsample(out)# [1, 48, 16, 16] ->[1, 48, 32, 32]
        P = torch.cat((P1, skip_outs[1], P3), dim=1)#[1, 336, 22, 22]

        y = self.SPP(P)#[2, 336, 22, 22]->[1, 96, 32, 32]
        # print("y: ",y.shape)
        y = self.conv1x1_1(y)# -> [1, 96, 32, 32])
        # print("y: ",y.shape)
        y = self.upsample(y)
        y = self.conv1x1_2(y)
        y = self.upsample(y)
        y = self.conv1x1_3(y)
        outc = self.cls_head(y)
        outr = self.reg_head(y)
        return outc, outr
if __name__ == '__main__':
    # model = EfficientCenternet()
    # print(torchsummary.summary(model,input_size=(3,256,256)))
    # out_c, out_r = model(torch.rand(1,3,256,256))
    # print(out_c.shape)

    # model = EfficientCenternet3()
    # print(torchsummary.summary(model,input_size=(3,256,256)))
    # out_c, out_r = model(torch.rand(1,3,256,256))
    # print(out_c.shape)

    model = EfficientCenterDet(2)
    print(torchsummary.summary(model,input_size=(3,256,256)))
    out_c, out_r = model(torch.rand(1,3,256,256))
    print(out_c.shape)
    # model = centernet(2,model_name='mv2')
    # print(torchsummary.summary(model,input_size=(3,256,256)))
    # outc, outr = model(torch.rand(1,3,256,256))
    # print(outc.shape)

