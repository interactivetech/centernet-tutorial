import torch
from data_gen import ShapeDataset
import matplotlib.pyplot as plt
from PIL import Image
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
from tqdm import tqdm
import numpy as np
from model import centernet
from efficient_centernet_model import EfficientCenternet
from loss import centerloss4
from train import train
from utils import pred2box_multiclass, filter_and_nms
import cv2
import datetime
from val import val
import matplotlib.pyplot as plt
import random
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()
def main(args):
    # IMG_RESOLUTION=384 
    IMG_RESOLUTION=256
    visualize_res=128
    MODEL_SCALE=IMG_RESOLUTION//visualize_res
    torch.cuda.set_device ( args.local_rank ) # use set_device and cuda to specify the desired GPU
    torch.distributed.init_process_group(
    'nccl',
        init_method='env://',
        world_size=4,
        timeout=datetime.timedelta(seconds=18000),
        rank=args.local_rank,
    )
    # ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/',
    #                      '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/coco_shapes.json',
    #                      MODEL_SCALE=MODEL_SCALE,
    #                      transform=train_transform_norm,
    #                      IMG_RESOLUTION=IMG_RESOLUTION)
    # val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/',
    #                      '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/coco_shapes.json',
    #                      MODEL_SCALE=MODEL_SCALE,
    #                      transform=validation_transform_norm,
    #                      IMG_RESOLUTION=IMG_RESOLUTION)
    # Specs Fruit Dataset
    # ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/fruit_specs_dataset/images',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/fruit_specs_dataset/annotations/coco-specs-fruit.json',
    # transform=train_transform_norm,
    # MODEL_SCALE=1,
    # IMG_RESOLUTION=IMG_RESOLUTION)
    # val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/fruit_specs_dataset/images',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/fruit_specs_dataset/annotations/coco-specs-fruit.json',
    # transform=validation_transform_norm,
    # MODEL_SCALE=1,
    # IMG_RESOLUTION=IMG_RESOLUTION)

    # COCO FRUIT DATASET
    # ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/train2017',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/fruit_train_2017.json',
    # transform=train_transform_norm,
    # MODEL_SCALE=1,
    # IMG_RESOLUTION=IMG_RESOLUTION)
    # val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/val2017',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/fruit_val_2017.json',
    # transform=validation_transform_norm,
    # MODEL_SCALE=1,
    # IMG_RESOLUTION=IMG_RESOLUTION)

    # # MINI COCO DATASET
    ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/train2017',
    '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_minitrain2017.json',
    transform=train_transform_norm,
    MODEL_SCALE=MODEL_SCALE,
    IMG_RESOLUTION=IMG_RESOLUTION)
    val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/val2017',
    '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_val2017.json',
    transform=validation_transform_norm,
    MODEL_SCALE=MODEL_SCALE,
    IMG_RESOLUTION=IMG_RESOLUTION)

    # # Food COCO DATASET
    # ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/train2017',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_food_train2017.json',
    # transform=train_transform_norm,
    # MODEL_SCALE=MODEL_SCALE,
    # IMG_RESOLUTION=IMG_RESOLUTION)
    # val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/val2017',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_food_val2017.json',
    # transform=validation_transform_norm,
    # MODEL_SCALE=MODEL_SCALE,
    # IMG_RESOLUTION=IMG_RESOLUTION)

    # Banana COCO DATASET
    # ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/train2017',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_banana_train2017.json',
    # transform=train_transform_norm,
    # MODEL_SCALE=1,
    # IMG_RESOLUTION=IMG_RESOLUTION)
    # val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/train2017',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_banana_train2017.json',
    # transform=validation_transform_norm,
    # MODEL_SCALE=1,
    # IMG_RESOLUTION=IMG_RESOLUTION)
    # val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/val2017',
    # '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/coco_dataset/annotations/instances_banana_val2017.json',
    # transform=validation_transform_norm,
    # MODEL_SCALE=1,
    # IMG_RESOLUTION=IMG_RESOLUTION)

    # ds = COCODetectionDataset(img_dir='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial',
    #             ann_json='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial/coco_shapes.json',
    #             IMG_RESOLUTION=512,
    #             transform=train_transform_norm)
    # val_ds = COCODetectionDataset(img_dir='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial',
    #                 ann_json='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial/coco_shapes.json',
    #                 IMG_RESOLUTION=512,
    #                 transform=validation_transform_norm)
    # BATCH_SIZE = 72
    # BATCH_SIZE = 72
    BATCH_SIZE = 52

    # Distributed Sampler has own shuffling mechanism, only use shuffle in sampler ranther than dataloader
    # https://github.com/NVIDIA/tacotron2/issues/168#issuecomment-474578149
    samp = torch.utils.data.distributed.DistributedSampler(ds,shuffle=True)
    t_sampler = torch.utils.data.distributed.DistributedSampler(val_ds,shuffle=False)
    # generator = torch.Generator()
    # generator.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(ds,
                                            batch_size=BATCH_SIZE,
                                            num_workers=8,
                                            pin_memory=True,
                                            sampler=samp,
                                            collate_fn = coco_detection_collate_fn,
                                            generator=None)
    print("Len of Train Loader: ",len(train_loader))
    val_loader = torch.utils.data.DataLoader(val_ds,
                                            batch_size=1,
                                            num_workers=0,
                                            sampler=None,
                                            pin_memory=True,
                                            collate_fn = coco_detection_collate_fn,
                                            generator=None)
    print("Len of Val Loader: ",len(val_loader))
    # val_loader = torch.utils.data.DataLoader(val_ds,
    #                                         batch_size=1,
    #                                         num_workers=0,
    #                                         pin_memory=True,
    #                                         collate_fn = coco_detection_collate_fn,
    #                                         generator=generator)
    
    LR = 1e-3
    # LR = 5e-4
    # LR = 0.05
    # LR=5e-4
    # LR = 0.02
    # LR = 2.5e-4*BATCH_SIZE
    from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(comment='mv2')
    print("args.local_rank: ",args.local_rank)
    writer=None
    if args.local_rank == 0:
        # writer = SummaryWriter(comment='_shape_mv2')
        writer = SummaryWriter(comment='_efd')

    multi_gpu=True
    # visualize_res=IMG_RESOLUTION
    EPOCHS = 401
    # model, losses, mask_losses, regr_losses, min_confidences, median_confidences, max_confidences = train('mv2',
    #                                                                                                         ds.num_classes,
    #                                                                                                         learn_rate=LR,
    #                                                                                                         epochs=300,
    #                                                                                                         train_loader=train_loader,
    #                                                                                                         val_ds=val_ds,
    #                                                                                                         val_loader=val_loader,
    #                                                                                                         writer=writer,
    #                                                                                                         multi_gpu=multi_gpu,
    #                                                                                                         visualize_res=visualize_res,
    #                                                                                                         IMG_RESOLUTION=IMG_RESOLUTION)
    model, losses, mask_losses, regr_losses, min_confidences, median_confidences, max_confidences = train('efd',
                                                                                                            ds.num_classes,
                                                                                                            learn_rate=LR,
                                                                                                            epochs=EPOCHS,
                                                                                                            train_loader=train_loader,
                                                                                                            val_ds=val_ds,
                                                                                                            val_loader=val_loader,
                                                                                                            writer=writer,
                                                                                                            multi_gpu=multi_gpu,
                                                                                                            visualize_res=visualize_res,
                                                                                                            IMG_RESOLUTION=IMG_RESOLUTION,
                                                                                                            local_rank=args.local_rank)
    if multi_gpu:
        torch.save(model.module.state_dict(),'ddp_efficient_centernet_{}_coco_pre.pth'.format(EPOCHS))
    else:
        torch.save(model.state_dict(),'efficient_centernet_{}.pth'.format(EPOCHS))
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
    # model.cpu()

    # eval
    # val(model,val_ds,val_loader,writer,epoch)

    for img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale,boxes_aug, target, idxs in val_loader:
            break

    pred_hm, pred_regs = model(img)# (4,1,128,128), (4,2,128,128)
    pred_hm = torch.sigmoid(pred_hm)
    # bboxes,scores,classes = pred2box_multiclass(pred_hm[0].cpu().data.numpy(),
    #                                                         pred_regs[0].cpu().detach().numpy(),128,1,thresh=0.0)
    if torch.cuda.is_available():
        bboxes,scores,classes = pred2box_multiclass(pred_hm[0].cpu().data.numpy(),pred_regs[0].cpu().data.numpy(),visualize_res,1,thresh=0.25)
    else:
        bboxes,scores,classes = pred2box_multiclass(pred_hm[0].data.numpy(),pred_regs[0].data.numpy(),visualize_res,1,thresh=0.25)
    bboxes,scores,classes =  filter_and_nms(bboxes,scores,classes,nms_threshold=0.45,n_top_scores=100)
    print(bboxes)

    for i in range(hm.shape[1]):
        if torch.cuda.is_available():
            hm_gt = hm[0].cpu().data.numpy()[i]
            hm_pred = pred_hm[0].cpu().data.numpy()[i]
        else:
            hm_gt = hm[0].data.numpy()[i]
            hm_pred = pred_hm[0].data.numpy()[i]
        hm_pred = np.dstack([hm_pred*255]*3).astype(np.uint8)
        for b,c in zip(bboxes,classes):
            if c == 0:
                x,y,x2,y2 = [int(k) for k in b]
                # print(x,y)
                cv2.rectangle(hm_pred,(x,y),(x2,y2),(255,0,0),1)
            if c == 1:
                x,y,x2,y2 = [int(k) for k in b]
                # print(x,y)
                cv2.rectangle(hm_pred,(x,y),(x2,y2),(0,255,0),1)
            
        plt.imshow(hm_gt,cmap='gray')
        plt.title("GT centerpoints of Class {}".format(i))
        plt.imshow(hm_gt,cmap='gray')
        plt.savefig("gt_hm.png")
        plt.clf()

        plt.title("prediction centerpoints for Class {} from model".format(i))
        plt.imshow(hm_pred)
        plt.savefig("hm_preds.png")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)