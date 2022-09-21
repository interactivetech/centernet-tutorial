from data_gen import ShapeDataset
import matplotlib.pyplot as plt
from PIL import Image
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
from tqdm import tqdm
import numpy as np
from model import centernet
from utils import pred2box_multiclass, filter_and_nms
from loss import centerloss4
import torch
import cv2
import torch
from PIL import Image
from val import val
from efficient_centernet_model import EfficientCenternet

def train(architecture='mv3',
          num_classes=2,
          learn_rate=1e-4,
          epochs=10,
          train_loader=None,
          val_ds=None,
          val_loader=None,
          writer=None,
          multi_gpu=False,
          visualize_res=None,
          IMG_RESOLUTION=None):
        if architecture == 'emv2':
                model = EfficientCenternet(num_classes=num_classes)
        else:
                model = centernet(num_classes,model_name=architecture)
        if multi_gpu:
                model = torch.nn.DataParallel(model,device_ids=[0,1,2,3],output_device=[0])
        EPOCHS = epochs
        LEARN_RATE = learn_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
        DEVICE = None
        if torch.cuda.is_available():
                DEVICE = 'cuda:0'
        elif not torch.cuda.is_available():
                DEVICE='cpu'
        model.to(DEVICE)  # Move model to the device selected for training
        losses = []
        mask_losses = []
        regr_losses = []
        min_confidences = []
        median_confidences = []
        max_confidences = []
        total_ind = 0
        for epoch in tqdm(range(EPOCHS)):
                for ind, (img,
                          hm, 
                          reg, 
                          wh,
                          reg_mask,
                          inds,
                          in_size,
                          out_size,
                          intermediate_size,
                          scale,
                          boxes_aug,
                          target,
                          idxs) in enumerate(train_loader):
                        # print(img.shape)
                        # print(in_size)
                        # print(out_size)
                        # print("idxs: ",idxs)
                        bboxes_gt = np.vstack([np.array(i['boxes']) for i in target])
                        if DEVICE == 'cuda:0':
                                img = img.to(DEVICE).cuda(non_blocking=True)
                                hm = hm.to(DEVICE).cuda(non_blocking=True)
                                reg = reg.to(DEVICE).cuda(non_blocking=True)
                                wh = wh.to(DEVICE).cuda(non_blocking=True)
                                reg_mask = reg_mask.to(DEVICE).cuda(non_blocking=True)
                                inds = inds.to(DEVICE).cuda(non_blocking=True)
                        
                        optimizer.zero_grad()
                        # print("begin")
                        pred_hm, pred_regs = model(img)
                        # print("pred_hm: ",pred_hm.shape)


                        loss,mask_loss,regr_loss = centerloss4(pred_hm,hm,pred_regs,reg,reg_mask,inds,wh)
                        loss.backward()
                        optimizer.step()
                        pred_hm = torch.sigmoid(pred_hm)
                        p = pred_hm[pred_hm>0]
                        if len(p.size()) > 0 :
                                # print("Min confidence: {}, Median confidence: {}, Max Confidence: {}".format(p.min(), np.median(p), p.max()))
                                # print(p.min().item())
                                min_confidences.append(p.min().item())
                                median_confidences.append(p.median().item())
                                max_confidences.append(p.max().item())
                        else:
                                min_confidences.append(0.0)
                                median_confidences.append(0.0)
                                max_confidences.append(0.0)
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
                # if epoch > 0 and epoch%10==0:
                # print(img.shape)
                
                if epoch > 0 and epoch%10==0:
                        # Val
                        val(model,val_ds,val_loader, writer,epoch,visualize_res=visualize_res,IMG_RESOLUTION=IMG_RESOLUTION)
                        model.train()
                        # im0 = Image.fromarray(i0)
                        # im1 = Image.fromarray(i1)
                        # im0.save('hm_0.png')
                        # im1.save('hm_1.png')

        writer.flush()
        writer.close()
        return model,losses,mask_losses,regr_losses, min_confidences, median_confidences, max_confidences