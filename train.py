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
        model = centernet(num_classes,model_name=architecture)
        EPOCHS = epochs
        LEARN_RATE = learn_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
        model.to('cpu')  # Move model to the device selected for training
        model.train(True)
        losses = []
        mask_losses = []
        regr_losses = []
        min_confidences = []
        median_confidences = []
        max_confidences = []
        for epoch in tqdm(range(EPOCHS)):
                for ind, (img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale, boxes_aug, target) in enumerate(train_loader):
                        # print(img.shape)
                        optimizer.zero_grad()
                        pred_hm, pred_regs = model(img)
                        p = pred_hm.detach().numpy()
                        p = p[p>0]
                        if p.size > 0 :
                                # print("Min confidence: {}, Median confidence: {}, Max Confidence: {}".format(p.min(), np.median(p), p.max()))
                                min_confidences.append(p.min())
                                median_confidences.append(np.median(p))
                                max_confidences.append(p.max())
                        else:
                                min_confidences.append(0.0)
                                median_confidences.append(0.0)
                                max_confidences.append(0.0)
                        loss,mask_loss,regr_loss = centerloss4(pred_hm,hm,pred_regs,reg,reg_mask,inds,wh)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                        mask_losses.append(mask_loss.item())
                        regr_losses.append(regr_loss.item())
                if p.size > 0:
                        print("Epoch {} - Min conf: {}, Median conf: {}, Max conf: {}".format(epoch,p.min(), np.median(p), p.max()))
                else:
                        print("Epoch {} - Min conf: {}, Median conf: {}, Max conf: {}".format(epoch,0,0,0))
                print("Epoch {} - Loss: {}, Mask Loss: {}, Reg Loss: {}".format(epoch,loss.item(),mask_loss.item(),regr_loss.item()))
        return model,losses,mask_losses,regr_losses, min_confidences, median_confidences, max_confidences