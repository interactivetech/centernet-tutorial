import cv2
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
from utils import make_hm_regr_multiclass

def iou(bbox1, bbox2):
    # IoU calculate
    # print("bbox1: ",bbox1)
    w = bbox1[2] - bbox1[0]
    h = bbox1[3] - bbox1[1]
    area1 = w*h
    # print("bbox2: ",bbox2)
    w = bbox2[2] - bbox2[0]
    h = bbox2[3] - bbox2[1]
    area2 = w*h
    xx1 = np.maximum(bbox1[ 1], bbox2[1])
    yy1 = np.maximum(bbox1[ 0], bbox2[0])
    xx2 = np.minimum(bbox1[ 3], bbox2[3])
    yy2 = np.minimum(bbox1[ 2], bbox2[2])
    # print(xx1,xx2,yy1,yy2)
    w = np.maximum(0.0, xx2 - xx1)
    # print("w ",w,"xx2 - xx1: ",xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    # print("inter: ",inter)
    # print("area1: ",area1)
    # print("area2: ",area2)
    iou = inter / (area1 + area2 - inter)
    return iou

def get_rand_num():
    return np.random.randint(10,256)
def get_rand_color():
    return (get_rand_num(),get_rand_num(),get_rand_num())

def gen_circle_obj(min_r,max_r,im_h,im_w,PAD):
    r = np.random.randint(min_r,max_r)
    # r = 50
    cx,cy = np.random.randint(r,im_w-r), np.random.randint(r,im_h-r)
    bbox = np.array([cx-r-PAD,cy-r-PAD,cx+r+PAD,cy+r+PAD])
    bbox[0] = np.clip(bbox[0],0,im_w)
    bbox[1] = np.clip(bbox[1],0,im_h)
    bbox[2] = np.clip(bbox[2],0,im_w)
    bbox[3] = np.clip(bbox[3],0,im_h)
    return cx,cy,r,bbox

def gen_rectangle_obj(min_wh,max_wh,im_h,im_w,PAD):
    w,h = np.random.randint(min_wh,max_wh), np.random.randint(min_wh,max_wh)
    cx,cy = np.random.randint(w,im_w-w), np.random.randint(h,im_h-h)

    # w,h = 50,50
    bbox = np.array([cx-w//2,cy-h//2,cx+w//2,cy+h//2])
    bbox_sh = np.array([cx-w//2-PAD,cy-h//2-PAD,cx+w//2+PAD,cy+h//2+PAD])

    
    bbox_sh[0] = np.clip(bbox_sh[0],0,im_w)
    bbox_sh[1] = np.clip(bbox_sh[1],0,im_h)
    bbox_sh[2] = np.clip(bbox_sh[2],0,im_w)
    bbox_sh[3] = np.clip(bbox_sh[3],0,im_h)
    return cx,cy,w,h,bbox,bbox_sh
def render_circle(img,cx,cy,r):
    cv2.circle(img,(cx,cy),r,get_rand_color(),-1)# bbox annotation
    # cv2.circle(img,(cx,cy),r,(255,0,0),-1)# bbox annotation

def render_rectangle(img,bbox):
    cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]), get_rand_color(), -1)
    # cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (0,0,255), -1)

def visualize_circle(img,bbox,color=(0,255,0),thickness=2):
    cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,thickness)

def visualize_rectangle(img,bbox,color=(0,255,0),thickness=2):
    cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color,thickness)

def get_predictions(annotations):
    detections = []
    
    # Generate Detections
    for a in annotations:
        # x,y,w,h = a['bbox']
        # print(a['image_id'],a['bbox'],a['category_id'])
        # print()
        # b = np.array([x,y,x+w,y+h])
        detection = {
            "image_id": a['image_id'],
            "category_id": a['category_id'],
            "bbox": a['bbox'],
            "score": 1.0
        }

        detections.append(detection)
    return detections
def xyxy2xywh(bbox):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    return np.array([bbox[0],bbox[1],w,h])
class ShapeDataset():
    def __init__(self,n_images = 100,n_circles=20,n_rectangles=20):
        self.N_IMAGES=n_images
        self.N_CIRCLES=n_circles
        self.N_RECTANGLES=n_rectangles
    def gen_dataset(self):
        images = []
        annotations = []
        ann_id_counter = 1 # annotation id needs to start with 1
        categories = [{'id': 0, 'name': 'circle'},{'id': 1, 'name': 'rectangle'}]
        # categories = [{'id': 0, 'name': 'circle'}]

        for image_id in range(1,self.N_IMAGES+1):
            img = np.zeros((512,512,3), np.uint8)
            bboxes = []
            anns = []

            # random_bg = get_rand_color()
            # img[:,:] = random_bg
            im_h,im_w,c = img.shape
            for _ in range(self.N_CIRCLES):
                cx,cy,r,bbox = gen_circle_obj(min_r=19,max_r=30,im_h=im_h,im_w=im_h,PAD=4)
                # cx,cy,r,bbox = gen_circle_obj(min_r=49,max_r=50,im_h=im_h,im_w=im_h,PAD=4)

                # print("{image_id}, sum( [iou(i,bbox) for i in bboxes]): ",not sum( [iou(i,bbox)>0.5 for i in bboxes])> 0)
                if not any( [iou(i,bbox)> 0 for i in bboxes]):
                    render_circle(img,cx,cy,r)
                    # visualize_circle(img,bbox)
                    c_bbox = xyxy2xywh(bbox)

                    bboxes.append(bbox)
                    area = c_bbox[2]*c_bbox[3]
                    # print("c_bbox: ",c_bbox)
                    # print("Area: ",area )
                    ann = {
                        'segmentation':[],
                        'area':int(area),
                        'iscrowd':0,
                        'image_id':image_id,
                        'bbox': [i for i in c_bbox.tolist()],
                        'category_id': 0,
                        'id':ann_id_counter
                    }
                    anns.append(ann)
                    ann_id_counter+=1
                # plt.imshow(img)
                # plt.scatter(cx, cy, color='red', s=10)
            print("Num Circle: {}".format(len(anns)))
            for _ in range(self.N_RECTANGLES):
                cx,cy,w,h,bbox,bbox_sh = gen_rectangle_obj(min_wh=29,max_wh=30,im_h=im_h,im_w=im_h,PAD=2)
                # cx,cy,w,h,bbox,bbox_sh = gen_rectangle_obj(min_wh=49,max_wh=50,im_h=im_h,im_w=im_h,PAD=2)


                if not any( [iou(i,bbox_sh)> 0 for i in bboxes]):
                    render_rectangle(img,bbox)
                    # visualize_rectangle(img,bbox_sh)
                    r_bbox = xyxy2xywh(bbox_sh)

                    bboxes.append(bbox_sh)
                    area = r_bbox[2]*r_bbox[3]
                    ann = {
                        'segmentation':[],
                        'area':int(area),
                        'iscrowd':int(0),
                        'image_id':image_id,
                        'bbox': [i for i in r_bbox.tolist()],
                        'category_id': int(1),
                        'id':ann_id_counter
                    }
                    anns.append(ann)
                    ann_id_counter+=1
            print("Num Circle+Rect: {}".format(len(anns)))

            annotations+=anns
            im =  {
            # 'file_name': 'COCO_val2014_000000001268.jpg',
                'file_name': "dummy_{}.png".format(image_id),
                'height': im_h,
                'width': im_w,
                'id': image_id
            }
            cv2.imwrite(im['file_name'],img)
            images.append(im)
            # plt.imshow(img)
            # plt.show()
            print("Num Anns: {}".format(ann_id_counter))
            print(annotations)
            print(images)
            coco_json = {
                'images':images,
                'annotations':annotations,
                'categories': categories
            }
            NAME = 'coco_shapes.json'
            json.dump(coco_json, open(NAME, 'w'))