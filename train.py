from data_gen import ShapeDataset
import matplotlib.pyplot as plt
from PIL import Image
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
from tqdm import tqdm
import numpy as np
from model import centernet
from loss import centerloss4
import torch

import torch
from torchvision.utils import make_grid
from PIL import Image
def train(architecture='mv3',num_classes=2,learn_rate=1e-4,epochs=10,train_loader=None,val_loader=None, writer=None):
        model = centernet(num_classes,model_name=architecture)
        EPOCHS = epochs
        LEARN_RATE = learn_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
        DEVICE = None
        if torch.cuda.is_available():
                DEVICE = 'cuda:0'
        elif not torch.cuda.is_available():
                DEVICE='cpu'
        model.to(DEVICE)  # Move model to the device selected for training
        model.train(True)
        losses = []
        mask_losses = []
        regr_losses = []
        min_confidences = []
        median_confidences = []
        max_confidences = []
        total_ind = 0
        for epoch in tqdm(range(EPOCHS)):
                for ind, (img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale, boxes_aug, target) in enumerate(train_loader):
                        # print(img.shape)
                        if DEVICE == 'cuda:0':
                                img = img.to(DEVICE).cuda(non_blocking=True)
                                hm = hm.to(DEVICE).cuda(non_blocking=True)
                                reg = reg.to(DEVICE).cuda(non_blocking=True)
                                wh = wh.to(DEVICE).cuda(non_blocking=True)
                                reg_mask = reg_mask.to(DEVICE).cuda(non_blocking=True)
                                inds = inds.to(DEVICE).cuda(non_blocking=True)
                        
                        optimizer.zero_grad()
                        pred_hm, pred_regs = model(img)
                        if DEVICE == 'cuda:0':
                                p = torch.sigmoid(pred_hm).cpu().detach().numpy()
                                # i = make_grid(torch.sigmoid(pred_hm),nrow=4).permute(1,2,0)[:,:,0].cpu().detach().numpy()
                                # im = Image.fromarray(np.dstack([i]*3))
                                # im.save('hm.png')

                        else:
                                # print("pred_hm: ",pred_hm.shape)
                                p = torch.sigmoid(pred_hm).detach().numpy()

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
                        writer.add_scalar("Loss/train", loss.item(), total_ind)
                        writer.add_scalar("Mask Loss/train", mask_loss.item(), total_ind)
                        writer.add_scalar("Reg Loss/train", regr_loss.item(), total_ind)
                        writer.add_scalar("Max Conf Score/train", max_confidences[-1], total_ind)
                        total_ind+=1

                # if p.size > 0:
                #         print("Epoch {} - Min conf: {}, Median conf: {}, Max conf: {}".format(epoch,p.min(), np.median(p), p.max()))
                # else:
                #         print("Epoch {} - Min conf: {}, Median conf: {}, Max conf: {}".format(epoch,0,0,0))
                # print("Epoch {} - Loss: {}, Mask Loss: {}, Reg Loss: {}".format(epoch,loss.item(),mask_loss.item(),regr_loss.item()))
                # Save Example
                if epoch%10==0:
                        # print(img.shape)
                        i = make_grid(img)
                        # print(i.shape)
                        i = i.permute(1,2,0).detach().numpy()
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        img_u = i*std + mean# unnormalize
                        img_u = (img_u*255.).astype(np.uint8)
                        writer.add_image('im', img_u, total_ind, dataformats='HWC')
                        # print("i: ",img_u.shape)
                        # im = Image.fromarray(img_u)
                        # im.save('im.png')

                        i = make_grid(torch.sigmoid(pred_hm),nrow=4).permute(1,2,0)
                        # print("ii: ",i.shape)
                        i0 = i[:,:,0].detach().numpy()
                        i1 = i[:,:,1].detach().numpy()
                        # print("i: ",i.shape)
                        i0 = np.dstack([i0*255]*3).astype(np.uint8)
                        i1 = np.dstack([i1*255]*3).astype(np.uint8)
                        writer.add_image('hm_0', i0, total_ind, dataformats='HWC')
                        writer.add_image('hm_1', i0, total_ind, dataformats='HWC')


                        # im0 = Image.fromarray(i0)
                        # im1 = Image.fromarray(i1)
                        # im0.save('hm_0.png')
                        # im1.save('hm_1.png')

        writer.flush()
        writer.close()
        return model,losses,mask_losses,regr_losses, min_confidences, median_confidences, max_confidences