
import torch
from data_gen import ShapeDataset
import matplotlib.pyplot as plt
from PIL import Image
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
from tqdm import tqdm
import numpy as np
from model import centernet
from loss import centerloss4
from train import train
from utils import pred2box_multiclass, filter_and_nms
import cv2
from val import val
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ds = COCODetectionDataset(img_dir='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial',
                ann_json='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial/coco_shapes.json',
                IMG_RESOLUTION=512,
                transform=train_transform_norm)
    val_ds = COCODetectionDataset(img_dir='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial',
                    ann_json='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial/coco_shapes.json',
                    IMG_RESOLUTION=512,
                    transform=validation_transform_norm)
    BATCH_SIZE = 8
    train_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True,
                                            collate_fn = coco_detection_collate_fn)
    val_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            collate_fn = coco_detection_collate_fn)
    
    LR = 1e-3
    # LR = 2.5e-4*BATCH_SIZE
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment='mv2')

    model, losses, mask_losses, regr_losses, min_confidences, median_confidences, max_confidences = train('mv2',
                                                                                                            ds.num_classes,
                                                                                                            learn_rate=LR,
                                                                                                            epochs=300,
                                                                                                            train_loader=train_loader,
                                                                                                            val_loader=val_loader,
                                                                                                            writer=writer)
    torch.save(model.state_dict(),'centernet_{}.pth'.format(300))
    plt.plot(range(len(losses)),losses )
    plt.plot(range(len(losses)),mask_losses)
    plt.plot(range(len(losses)),regr_losses)
    plt.title("centernet (mobilenetv3 backbone) training")
    plt.legend(['loss','mask loss','regression loss'])
    # plt.show()
    plt.savefig("loss.png")
    plt.clf()
        


    plt.plot(range(len(min_confidences)),min_confidences )
    plt.plot(range(len(median_confidences)),median_confidences)
    plt.plot(range(len(max_confidences)),max_confidences)
    plt.title("confidence scores for centernet (mv3 backbone) during training")
    plt.legend(['min_confidences','median_confidences','max_confidences'])
    # plt.show()
    plt.savefig("conf.png")
    plt.clf()
    model.eval()
    model.cpu()

    # eval
    val(model,val_ds,val_loader)

    for img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale,boxes_aug, target in val_loader:
            break

    pred_hm, pred_regs = model(img)# (4,1,128,128), (4,2,128,128)
    pred_hm = torch.sigmoid(pred_hm)
    # bboxes,scores,classes = pred2box_multiclass(pred_hm[0].cpu().data.numpy(),
    #                                                         pred_regs[0].cpu().detach().numpy(),128,1,thresh=0.0)
    bboxes,scores,classes = pred2box_multiclass(pred_hm[0].data.numpy(),pred_regs[0].data.numpy(),128,1,thresh=0.25)
    bboxes,scores,classes =  filter_and_nms(bboxes,scores,classes,nms_threshold=0.45,n_top_scores=20)
    print(bboxes)

    for i in range(hm.shape[1]):
        hm_gt = hm[0].data.numpy()[i]
        hm_pred = pred_hm[0].data.numpy()[i]
        hm_pred = np.dstack([hm_pred*255]*3).astype(np.uint8)
        for b,c in zip(bboxes,classes):
            if c == i:
                x,y,x2,y2 = [int(k) for k in b]
                # print(x,y)
                cv2.rectangle(hm_pred,(x,y),(x2,y2),(255,0,0),1)
            
        plt.imshow(hm_gt,cmap='gray')
        plt.title("GT centerpoints of Class {}".format(i))
        plt.imshow(hm_gt,cmap='gray')
        plt.savefig("gt_hm.png")
        plt.clf()

        plt.title("prediction centerpoints for Class {} from model".format(i))
        plt.imshow(hm_pred)
        plt.savefig("hm_preds.png")