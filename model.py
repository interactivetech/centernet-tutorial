from statistics import mode
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# remove operators that are not supported by SnapML
def replace_hardsigmoid(model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, nn.Hardsigmoid):
            model._modules[name] = nn.Sigmoid()
        if len(list(module.children())) > 0:
            model._modules[name] = replace_hardsigmoid(model=module)
    return model
def replace_hardswish_with_relu(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_hardswish_with_relu(model=module)
        if isinstance(module, nn.Hardswish):
            model._modules[name] = nn.ReLU(True)
    return model


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        '''
        Author of focal loss:  all new conv layers except the 
        final layer, subnets are initalized  with bias b=0 and
        a Gaussian weight filled with sigma=0.01

        Weight init improves training stability for both the 
        cross entropy and focal loss in the case of heavy class
        imbalance
        '''
        for m in self.conv.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
        
    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # if x2 is not None:
        #     x = torch.cat([x2, x1], dim=1)
        #     # input is CHW
        #     diffY = x2.size()[2] - x1.size()[2]
        #     diffX = x2.size()[3] - x1.size()[3]

        #     x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                     diffY // 2, diffY - diffY//2))
        # else:
        x = x1
        x = self.conv(x)
        return x

class centernet(nn.Module):
    def __init__(self, 
                 n_classes=1, 
                 model_name="resnet18",
                 ckpt='/Users/mendeza/Downloads/mobileone_s0_unfused.pth.tar'):
        super(centernet, self).__init__()
        # create backbone.
        if model_name == 'mv3':
            basemodel = torchvision.models.mobilenet_v3_small(pretrained=True)
            basemodel = replace_hardswish_with_relu(basemodel)
            basemodel = replace_hardsigmoid(basemodel)
        if model_name == 'mv2':
            basemodel = torchvision.models.mobilenetv2(pretrained=False)
            basemodel = replace_hardswish_with_relu(basemodel)
            basemodel = replace_hardsigmoid(basemodel)
        elif model_name == 'resnet18':
            basemodel = torchvision.models.resnet18(pretrained=True) # turn this on for training
        elif model_name == 'resnet34':
            basemodel = torchvision.models.resnet34(pretrained=False) # turn this on for training

        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        # set basemodel
        self.base_model = basemodel
        
        if model_name == "resnet34" or model_name=="resnet18":
            num_ch = 512
        elif model_name == "mv3":
            # num_ch = 960 # for large
            num_ch = 576
        elif model_name == "mobileone":
            num_ch = 1024
        else:
            num_ch = 2048
        
        self.up1 = up(num_ch, 512,bilinear=True)
        self.up2 = up(512, 256,bilinear=True)
        self.up3 = up(256, 256,bilinear=True)
        # output classification
        self.outc = nn.Conv2d(256, n_classes, 1)
        '''
        Author: the final conv later set the bias
        initalization to log((1-math.pi)/math.pi)
        pi specifies that at the start of training,
        every anchor should be labeled as foreground
        with confidence of about pi. pi is a variable that is 0.1

        This initalization prevents a large number of background 
        anchors from generating a large destabalizing loss value in the
        in the first iteration of training
        '''
        self.outc.bias.data.fill_(-2.19)
        # output residue
        self.outr = nn.Conv2d(256, 2, 1)
        self.outr.bias.data.fill_(-2.19)
    def forward(self, x):
        batch_size = x.shape[0]
        # print("x: ",x.shape, x.dtype)
        x = self.base_model(x) 
        # print("features: ",x.shape)
        
        # Add positional info        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        outc = self.outc(x)
        outr = self.outr(x)
        return outc, outr


if __name__ == '__main__':
    model = centernet(n_classes=1,model_name='mv3')
    # print(model)
    outc, outr = model(torch.rand(2,3,512,512))
    print("outc: ",outc.shape)
    print("outr: ",outr.shape)
    # m = mobileone(variant='s0', inference_mode=False)
    # checkpoint = torch.load('/Users/mendeza/Downloads/mobileone_s0_unfused.pth.tar',map_location=torch.device('cpu'))
    # m.load_state_dict(checkpoint)
    # # m = torchvision.models.mobilenet_v3_large(pretrained=False) 
    # l=  nn.Sequential(*list(m.children())[:-2])
    # print(l(torch.rand(1,3,256,256)).shape)# 960,