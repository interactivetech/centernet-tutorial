import torch
import torch.nn as nn
import numpy as np
import math
import cv2
from torch import Tensor
import torchvision
def _box_xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) which will be converted.
    Returns:
        boxes (Tensor[N, 4]): boxes in (x, y, w, h) format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1  # x2 - x1
    h = y2 - y1  # y2 - y1
    boxes = torch.stack((x1, y1, w, h), dim=-1)
    return boxes

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

def make_hm_regr_multiclass(bboxes,classes,N_CLASSES,input_size=512,MODEL_SCALE=4,IN_SCALE=1,MAX_N_OBJECTS=128):
    reg_mask = np.zeros((MAX_N_OBJECTS),dtype=np.uint8)
    wh = np.zeros((MAX_N_OBJECTS,2),dtype=np.float32)
    inds = np.zeros((MAX_N_OBJECTS),dtype=np.int64)
    feature_scale =input_size//MODEL_SCALE
    hm = np.zeros((N_CLASSES,feature_scale,feature_scale))
    reg = np.zeros((2,feature_scale,feature_scale))
    try:
      # print("bboxes: ",bboxes.shape)
      if bboxes is not None:
        centers = np.array([bboxes[:,0]+bboxes[:,2]//2,bboxes[:,1]+bboxes[:,3]//2,bboxes[:,2],bboxes[:,3]]).T
        for ind,(c,l )in enumerate(zip(centers,classes)):
            h, w = c[3]/MODEL_SCALE, c[2]/MODEL_SCALE
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            # print("radius:", radius)
            draw_umich_gaussian(hm[l], [int(c[0])//MODEL_SCALE,int(c[1])//MODEL_SCALE], 
                                    radius)
            # draw_msra_gaussian(hm[l], [int(c[0])//MODEL_SCALE,int(c[1])//MODEL_SCALE], 
            #                         radius)
            reg_mask[ind] = 1
            wh[ind] = c[2:]/input_size
            draw_dense_reg(reg,hm[l],c[:2]//MODEL_SCALE,wh[ind],radius)


            inds[ind] = (int(c[1])//MODEL_SCALE)*feature_scale + (int(c[0])//MODEL_SCALE)
    except Exception as e:
      print(e)
      print("bboxes: ",bboxes.shape)

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)

  dim = value.shape[0]
  reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value

  if is_offset and dim == 2:
    delta = np.arange(diameter*2+1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]

  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap

def make_hm_regr(bboxes, input_size=512,MODEL_SCALE=4,IN_SCALE=1,MAX_N_OBJECTS=128):
    reg_mask = np.zeros((MAX_N_OBJECTS),dtype=np.uint8)
    wh = np.zeros((MAX_N_OBJECTS,2),dtype=np.float32)
    inds = np.zeros((MAX_N_OBJECTS),dtype=np.int64)
    feature_scale =input_size//MODEL_SCALE
    hm = np.zeros((feature_scale,feature_scale))
    reg = np.zeros((2,feature_scale,feature_scale))
    centers = np.array([bboxes[:,0]+bboxes[:,2]//2,bboxes[:,1]+bboxes[:,3]//2,bboxes[:,2],bboxes[:,3]]).T
    for ind,c in enumerate(centers):
        h, w = c[3]/MODEL_SCALE, c[2]/MODEL_SCALE
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        # print("radius:", radius)
        draw_umich_gaussian(hm, [int(c[0])//MODEL_SCALE,int(c[1])//MODEL_SCALE], 
                                radius)
        reg_mask[ind] = 1
        wh[ind] = c[2:]/input_size
        draw_dense_reg(reg,hm,c[:2]//MODEL_SCALE,wh[ind],radius)

        inds[ind] = (int(c[1])//MODEL_SCALE)*feature_scale + (int(c[0])//MODEL_SCALE)
    return hm,reg, wh,reg_mask,inds

def make_hm_regr2(bboxes,input_size=512,IN_SCALE=1,MODEL_SCALE=4,gauss='msra',MAX_N_OBJECTS=128):
    # make output heatmap for single class
    hm = np.zeros([input_size//MODEL_SCALE, input_size//MODEL_SCALE])
    # make regr heatmap 
    regr = np.zeros([2, input_size//MODEL_SCALE, input_size//MODEL_SCALE])
    
    reg_mask = np.zeros((MAX_N_OBJECTS),dtype=np.uint8)
    wh = np.zeros((MAX_N_OBJECTS,2),dtype=np.float32)
    inds = np.zeros((MAX_N_OBJECTS),dtype=np.int64)
    feature_scale =input_size//MODEL_SCALE

    if len(bboxes) == 0:
        return hm, 

def make_hm_regr_multiclass(bboxes,classes,N_CLASSES,input_size=512,MODEL_SCALE=4,IN_SCALE=1,MAX_N_OBJECTS=128):
    reg_mask = np.zeros((MAX_N_OBJECTS),dtype=np.uint8)
    wh = np.zeros((MAX_N_OBJECTS,2),dtype=np.float32)
    inds = np.zeros((MAX_N_OBJECTS),dtype=np.int64)
    feature_scale =input_size//MODEL_SCALE
    hm = np.zeros((N_CLASSES,feature_scale,feature_scale))
    reg = np.zeros((2,feature_scale,feature_scale))
    try:
      # print("bboxes: ",bboxes.shape)
      if bboxes is not None:
        centers = np.array([bboxes[:,0]+bboxes[:,2]//2,bboxes[:,1]+bboxes[:,3]//2,bboxes[:,2],bboxes[:,3]]).T
        for ind,(c,l )in enumerate(zip(centers,classes)):
            h, w = c[3]/MODEL_SCALE, c[2]/MODEL_SCALE
            # print("h, w: ",h,w)
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            # print("radius: ",radius)
            radius = max(0, int(radius))
            # print("radius: ",radius)
            # print("radius:", radius)
            draw_umich_gaussian(hm[l], [int(c[0])//MODEL_SCALE,int(c[1])//MODEL_SCALE], 
                                    radius)
            # draw_msra_gaussian(hm[l], [int(c[0])//MODEL_SCALE,int(c[1])//MODEL_SCALE], 
            #                         radius)
            reg_mask[ind] = 1
            wh[ind] = c[2:]/input_size
            draw_dense_reg(reg,hm[l],c[:2]//MODEL_SCALE,wh[ind],radius)


            inds[ind] = (int(c[1])//MODEL_SCALE)*feature_scale + (int(c[0])//MODEL_SCALE)
    except Exception as e:
      print(e)
      print("bboxes: ",bboxes.shape)

    return hm,reg, wh,reg_mask,inds
def make_hm_regr_multiclass2(bboxes,classes,N_CLASSES,input_size=512,MODEL_SCALE=4,IN_SCALE=1,MAX_N_OBJECTS=128):
    cat_reg_mask = np.zeros((MAX_N_OBJECTS,N_CLASSES*2),dtype=np.uint8)
    wh = np.zeros((MAX_N_OBJECTS,2),dtype=np.float32)

    cat_wh = np.zeros((MAX_N_OBJECTS,N_CLASSES*2),dtype=np.float32)
    inds = np.zeros((MAX_N_OBJECTS),dtype=np.int64)
    feature_scale =input_size//MODEL_SCALE
    hm = np.zeros((N_CLASSES,feature_scale,feature_scale))
    reg = np.zeros((2,feature_scale,feature_scale))
    centers = np.array([bboxes[:,0]+bboxes[:,2]//2,bboxes[:,1]+bboxes[:,3]//2,bboxes[:,2],bboxes[:,3]]).T
    for ind,(c,l )in enumerate(zip(centers,classes)):
        h, w = c[3]/MODEL_SCALE, c[2]/MODEL_SCALE
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        # print("radius:", radius)
        draw_umich_gaussian(hm[l], [int(c[0])//MODEL_SCALE,int(c[1])//MODEL_SCALE], 
                                radius)
        cat_reg_mask[ind,l*2:l*2+2] = 1
        wh[ind] = c[2:]/input_size
        cat_wh[ind,l*2:l*2+2] = wh[ind]
        draw_dense_reg(reg,hm[l],c[:2]//MODEL_SCALE,wh[ind],radius)

        inds[ind] = (int(c[1])//MODEL_SCALE)*feature_scale + (int(c[0])//MODEL_SCALE)
    return hm,reg, cat_wh,cat_reg_mask,inds
def pred2box(hm, regr,input_size,MODEL_SCALE ,thresh=0.99):
    # make binding box from heatmaps
    # thresh: threshold for logits.
        
    # get center
    pred = hm > thresh
    pred_center = np.where(hm>thresh)
    # get regressions
    pred_r = regr[:,pred].T

    # wrap as boxes
    # [xmin, ymin, width, height]
    # size as original image.
    boxes = []
    scores = hm[pred]
    for i, b in enumerate(pred_r):
        arr = np.array([pred_center[1][i]*MODEL_SCALE-b[0]*input_size//2, pred_center[0][i]*MODEL_SCALE-b[1]*input_size//2, 
                      int(b[0]*input_size), int(b[1]*input_size)])
        arr = np.clip(arr, 0, input_size)
        # filter 
        #if arr[0]<0 or arr[1]<0 or arr[0]>input_size or arr[1]>input_size:
            #pass
        boxes.append(arr)
    return np.asarray(boxes), np.asarray(scores), np.asarray([1]*len(scores))
def pred2box_multiclass(hm, regr,input_size,MODEL_SCALE ,thresh=0.99):
    # make binding box from heatmaps
    # thresh: threshold for logits.
    all_boxes = []
    all_classes = []
    all_scores = []
    for cl in range(hm.shape[0]):    
        # get center
        pred = hm[cl] > thresh
        pred_center = np.where(hm[cl]>thresh)
        # get regressions
        pred_r = regr[:,pred].T

        # wrap as boxes
        # [xmin, ymin, width, height]
        # size as original image.
        boxes = []
        scores = hm[cl,pred]
        for i, b in enumerate(pred_r):
            arr = np.array([pred_center[1][i]*MODEL_SCALE-b[0]*input_size//2, pred_center[0][i]*MODEL_SCALE-b[1]*input_size//2, 
                        int(b[0]*input_size), int(b[1]*input_size)])
            arr = np.clip(arr, 0, input_size)
            # filter 
            #if arr[0]<0 or arr[1]<0 or arr[0]>input_size or arr[1]>input_size:
                #pass
            boxes.append(arr)
        all_boxes+=boxes
        all_scores+=scores.tolist()
        all_classes+=[cl]*len(scores.tolist())
    return np.asarray(all_boxes), np.asarray(all_scores), np.asarray(all_classes)

def filter_and_nms(bboxes, scores,classes,nms_threshold=0.6,n_top_scores=None ):
    '''
    clean boxes and filter via nms
    '''
    try:
        # inds = [ind for ind,i in enumerate(bboxes) if i[0]>0 and i[1]>0 and i[2]>0 and i[3]>0]
        # bboxes = bboxes[inds]
        # scores = scores[inds]
        # classes = classes[inds]

        # print("bboxes: ",bboxes[0])
        # print("scores: ",scores.shape)    
        bboxes_xyxy = torch.Tensor(np.array([bboxes[:,0],bboxes[:,1],bboxes[:,0]+bboxes[:,2],bboxes[:,1]+bboxes[:,3]])).T
        # print("bboxes_xyxy: ",bboxes_xyxy.shape)
        # print("bboxes_xyxy: ",bboxes_xyxy[0])
        scores = torch.Tensor(scores)
        classes = torch.Tensor(classes)
        indicies = torchvision.ops.nms(bboxes_xyxy,scores,nms_threshold)
        bboxes_xyxy = bboxes_xyxy[indicies].numpy()
        scores = scores[indicies].numpy()
        classes = classes[indicies].numpy()
        # N_TOP = 100
        N_TOP = n_top_scores

        bboxes_xyxy = bboxes_xyxy[:N_TOP]
        scores = scores[:N_TOP]
        classes = classes[:N_TOP]
    except Exception as e:
        print(e)
        bboxes_xyxy = []
        scores = []
        classes = []
    
    return bboxes_xyxy, scores, classes

def per_class_coco_ap(coco,coco_eval):
    results_per_category=[]
    precisions = coco_eval.eval['precision']
    cats =  [i['id'] for i in coco.cats.values()]
    # print(cats)
    for idx, catId in enumerate(cats):
        # print(catId)
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        nm = coco.loadCats([catId])
        # print(nm[0]['name'])
        # print("precisions: ",precisions.shape)
        precision = precisions[:, :, idx, 0, -1]
        # print("precision:", precision.shape)
        # for ind, row in enumerate(precision):
            # print(ind,row)
        precision = precision[precision > -1]
        if precision.size:
            ap = np.mean(precision)
        else:
            ap = float('nan')
        results_per_category.append(
            (f'{nm[0]["name"]}', f'{float(ap):0.3f}'))
    return results_per_category