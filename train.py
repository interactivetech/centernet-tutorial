from data_gen import ShapeDataset
import matplotlib.pyplot as plt
from PIL import Image
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
from tqdm import tqdm
import numpy as np
from model import centernet
from loss import centerloss4
import torch
def train(architecture='mv3',num_classes=2,learn_rate=1e-4,epochs=10,train_loader=None,val_loader=None):
        model = centernet(num_classes,model_name='mv3')
        EPOCHS = epochs
        LEARN_RATE = learn_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
        model.to('cpu')  # Move model to the device selected for training
        model.train(True)
        losses = []
        mask_losses = []
        regr_losses = []
        for epoch in tqdm(range(EPOCHS)):
                print("----EPOCH: {}----".format(epoch))
                for ind, (img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale, boxes_aug, target) in enumerate(tqdm(train_loader)):
                        # print(img.shape)
                        optimizer.zero_grad()
                        pred_hm, pred_regs = model(img)
                        loss,mask_loss,regr_loss = centerloss4(pred_hm,hm,pred_regs,reg,reg_mask,inds,wh)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                        mask_losses.append(mask_loss.item())
                        regr_losses.append(regr_loss.item())
                        if ind%10==0:
                                print("Loss: {}, Mask Loss: {}, Reg Loss: {}".format(loss.item(),mask_loss.item(),regr_loss.item()))
        return model,losses,mask_losses,regr_losses