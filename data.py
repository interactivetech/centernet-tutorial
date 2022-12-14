import torch
import torchvision
from torchvision import transforms
import torch.onnx as onnx
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import matplotlib.pyplot as plt
import contextlib
import io
import numpy as np
import cv2
from utils import make_hm_regr,make_hm_regr2, pred2box, pred2box_multiclass, make_hm_regr_multiclass, make_hm_regr_multiclass2,_box_xyxy_to_xywh
import albumentations as albu

def train_transform_norm(annotations,INPUT_SIZE,with_bboxes=True):
    image = annotations['image']
    # print("image.shape: ",image.shape)
    size = (INPUT_SIZE[0], INPUT_SIZE[1]) # height, width
    scale = min(size[0] / image.shape[0], size[1] / image.shape[1])
    intermediate_size = int(image.shape[0] * scale), int(image.shape[1] * scale)
    augmentation = albu.Compose(
        [
            albu.RandomSizedBBoxSafeCrop(*intermediate_size),
            # albu.Resize(height = intermediate_size[0],width = intermediate_size[1]),
            # albu.Resize(*intermediate_size),
            albu.HorizontalFlip(p=0.5),
            albu.HueSaturationValue(p=0.5),
            albu.RGBShift(p=0.5),
            albu.RandomBrightnessContrast(p=0.5),
            albu.MotionBlur(p=0.5),
            albu.PadIfNeeded(*size,border_mode=cv2.BORDER_CONSTANT,mask_value=0.0),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ],
        albu.BboxParams(format='coco', min_area=0.0,
                        min_visibility=0.0, label_fields=['labels'])
    )
    
    augmented = augmentation(**annotations)

    augmented['scale'] = scale    # augmented['image'] = augmented['image'].astype(
    #     np.float32).transpose(2, 0, 1)
    augmented['scale'] = scale

    augmented['in_size'] = image.shape[:2]
    augmented['out_size'] = size
    augmented['intermediate_size'] = intermediate_size
    return augmented

def validation_transform_norm(annotations, INPUT_SIZE,with_bboxes=True):
    bbox_params = None
    if with_bboxes:
        bbox_params = albu.BboxParams(format='coco', min_area=0.0,
                        min_visibility=0.0, label_fields=['labels'])
    
    image = annotations['image']#H,W,C
    # print("image.shape: ",image.shape)

    size = (INPUT_SIZE[0], INPUT_SIZE[1])# h,w
    # print(image.shape)
    scale = min(size[0] / image.shape[0], size[1] / image.shape[1])
    # intermediate_size = [int(dim * scale) for dim in image.shape[:2]]
    intermediate_size = int(image.shape[0] * scale), int(image.shape[1] * scale)

    
    augmentation = albu.Compose(
        [
            albu.Resize(*intermediate_size),
            albu.PadIfNeeded(*size,border_mode=cv2.BORDER_CONSTANT,mask_value=0.0),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ],
        bbox_params
    )
    try:
        augmented = augmentation(**annotations)
    except Exception as e:
        print(annotations)
    augmented['scale'] = scale

    augmented['in_size'] = image.shape[:2]
    augmented['out_size'] = size
    augmented['intermediate_size'] = intermediate_size
    return augmented

class COCODetectionDataset(torch.utils.data.Dataset):

    def __init__(self,
                 img_dir,
                 ann_json,
                 IMG_RESOLUTION=None,
                 MODEL_SCALE=None,
                 transform=None):
        # self.img_id = img_id
        # self.labels = labels
        self.IMG_RESOLUTION = IMG_RESOLUTION
        min_keypoints_per_image = 10
        self.transform = None
        if transform:
            self.transform = transform
        

        with contextlib.redirect_stdout(io.StringIO()):     # redict pycocotools print()
            coco = COCO(ann_json)
        
        # filter no labeled examples
        # ids = []
        # for ds_idx, img_id in enumerate(coco.getImgIds()):
        #     ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        #     anno = coco.loadAnns(ann_ids)
        #     if self._has_valid_annotation(anno):
        #         ids.append(ds_idx+1)

        # dataset = torch.utils.data.Subset(dataset, ids)

        cat_ids = sorted(coco.getCatIds())
        label_map = {v: i for i, v in enumerate(cat_ids)}
        print("label_map:", label_map )
        inverse_label_map = {v: k for k, v in label_map.items()}
        img_ids = sorted(coco.getImgIds())
        self.ids = img_ids

        # print(self.ids)
        imgs = coco.loadImgs(self.ids)                       # each img has keys filename, height, width, id
        target = [coco.imgToAnns[idx] for idx in img_ids]   # each ann has keys bbox, category_id, id
        
        img_names = [x["file_name"] for x in imgs]
        targets = []
        # boxes = 
        '''
        (09/03/2022) Andrew: issue with albumentations that cant handle labels with [0,0,0,0]
        if a bbox annotation exists, then do not add the bbox or the label
        '''
        for img_anns, img in zip(target, imgs):
            boxes = []
            labels = []
            for ann in img_anns:
                if ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
                    boxes.append(ann['bbox'])
                    labels.append(label_map[ann['category_id']])
            t = {
                'boxes': boxes,
                "labels": labels,
                "image_width":img["width"],
                "image_height":img["height"],
                "image_id":img['id'],
                }
            targets.append(t)
            # {
            # "boxes": [ann["bbox"] for ann in img_anns if ann["bbox"][2] > 0 or ann["bbox"][3] > 0 ],
            # "labels": [label_map[ann["category_id"]] for ann in img_anns],
            # "image_width": img["width"],
            # "image_height": img["height"],
            # "image_id": img["id"]
            # } 
        self.coco = coco
        self.img_dir = img_dir
        self.img_names = img_names
        self.MODEL_SCALE = MODEL_SCALE
        self.targets = targets
        self.transform = transform
        self.num_classes = len(cat_ids)
        self.label_map = label_map
        self.inverse_label_map = inverse_label_map
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # print("idx: ",idx)
        # print(self.img_names[idx])
        # print( self.targets[idx])
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = np.array(Image.open(img_path).convert("RGB"))

        target = self.targets[idx]
        boxes = np.array(target['boxes']).reshape(-1,4)
        # print("boxes: ",boxes)
        labels = np.array(target['labels'])
        annotations = {'image': img,
                'bboxes':boxes,
                'labels': labels }
        try:
            anns = self.transform(annotations,(self.IMG_RESOLUTION,self.IMG_RESOLUTION))
        except Exception as e:
            print(annotations)

        
        img  = anns['image'].transpose(2,0,1)
        boxes_aug = np.asarray(anns['bboxes'])
        # print(boxes_aug)

        labels = np.asarray(anns['labels'])
        in_size = anns['in_size']
        out_size = anns['out_size']
        intermediate_size = anns['intermediate_size']
        scale =  anns['scale']

        if self.num_classes >1:
            # print("boxes_aug: ",boxes_aug)
            # print("Model Scale: ",self.MODEL_SCALE)
            hm, reg,wh,reg_mask,inds  = make_hm_regr_multiclass(boxes_aug,
                                                                labels,
                                                                N_CLASSES=self.num_classes,
                                                                input_size=self.IMG_RESOLUTION,
                                                                MODEL_SCALE=self.MODEL_SCALE,
                                                                IN_SCALE=1,
                                                                MAX_N_OBJECTS=128)

            hm = np.ascontiguousarray(hm)
            reg = np.ascontiguousarray(reg)
            wh = np.ascontiguousarray(wh)
            reg_mask = np.ascontiguousarray(reg_mask)
            inds = np.ascontiguousarray(inds)
            # h_hm_regr_multiclass2(boxesm,reg, cat_wh,cat_reg_mask,inds = make,
            #                                                             labels,
            #                                                             N_CLASSES=2,
            #                                                             input_size=512,
            #                                                             MODEL_SCALE=4,
            #                                                             IN_SCALE=1,
            #                                                             MAX_N_OBJECTS=128)
        elif self.num_classes ==1:
            # boxes = np.array(target['boxes']).reshape(-1,4)
            # labels = np.array(target['labels'])
            hm, reg,wh,reg_mask,inds  = make_hm_regr_multiclass(boxes_aug,
                                                                labels,
                                                                N_CLASSES=self.num_classes,
                                                                input_size=self.IMG_RESOLUTION,
                                                                MODEL_SCALE=self.MODEL_SCALE,
                                                                IN_SCALE=1,
                                                                MAX_N_OBJECTS=128)
            hm = np.ascontiguousarray(hm)
            reg = np.ascontiguousarray(reg)
            wh = np.ascontiguousarray(wh)
            reg_mask = np.ascontiguousarray(reg_mask)
            inds = np.ascontiguousarray(inds)
            # hm,reg,wh,reg_mask,inds = make_hm_regr2(boxes,input_size=512,IN_SCALE=1,MODEL_SCALE=4,gauss='msra',MAX_N_OBJECTS=128)
            # hm = np.expand_dims(hm,0)
        # print("target: ",target)
        # print("img: ",img.shape)
        # print("target: ",target)
        # print( inds, in_size, out_size, intermediate_size, scale)
        return img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale, boxes_aug, target, idx
        # return img, hm,reg, cat_wh,cat_reg_mask,inds 
def coco_detection_collate_fn(batch):
    img = torch.Tensor(np.stack([x[0] for x in batch], axis=0))
    hm = torch.Tensor(np.stack([x[1] for x in batch], axis=0))
    reg = torch.Tensor(np.stack([x[2] for x in batch], axis=0))
    wh = torch.Tensor(np.stack([x[3] for x in batch], axis=0))
    reg_mask = torch.Tensor(np.stack([x[4] for x in batch], axis=0))
    inds = torch.Tensor(np.stack([x[5] for x in batch], axis=0)).type(torch.int64)
    in_size = torch.Tensor(np.stack([x[6] for x in batch], axis=0))
    out_size = torch.Tensor(np.stack([x[7] for x in batch], axis=0))
    intermediate_size = torch.Tensor(np.stack([x[8] for x in batch], axis=0))
    scale = torch.Tensor(np.stack([x[9] for x in batch], axis=0))
    # print(np.vstack([x[10] for x in batch]).shape)
    boxes_aug = torch.Tensor(np.vstack([x[10] for x in batch]))
    targets = [x[11] for x in batch]
    idxs = [x[12] for x in batch]

    return img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale, boxes_aug, targets, idxs
