import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
from utils import pred2box_multiclass, filter_and_nms, per_class_coco_ap
from tqdm import tqdm
import json
from model import centernet
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
# from val import val
def rescale_boxes( boxes, out_size, intermediate_size, scale):
    """ Removes padding shift and rescales bounding boxes to original
    input image size """
    outh, outw = out_size
    rh, rw = intermediate_size

    if rh < outh:
        boxes[:, 1] -= ((outh - rh) / 2)
        boxes[:, 1] = np.clip(boxes[:, 1], a_min=0, a_max=outh-1)
    if rw < outw:
        boxes[:, 0] -= ((outw - rw) / 2)
        boxes[:, 0] = np.clip(boxes[:, 0], a_min=0, a_max=outw-1)

    boxes /= scale
    
    return boxes

def val(model,val_ds,val_loader):
    
    detections = []
    for img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale,boxes_aug, targets  in tqdm(val_loader):
            image_ids = [i['image_id'] for i in targets]
            bboxes_gt = np.vstack([np.array(i['boxes']) for i in targets])
            # for b in bboxes_gt:
            #     print("gt: ",b)
            # print("bboxes_gt: ",[i.shape for i in bboxes_gt])
            # print("bboxes_gt: ",bboxes_gt.shape)
            with torch.no_grad():
                pred_hm, pred_regs = model(img)# (4,1,128,128), (4,2,128,128)
                pred_hm = torch.sigmoid(pred_hm)

                for ind in range(pred_hm.shape[0]):
                        try:
                            image_id = image_ids[ind]
                            # TODO(ANDREW): Make prediction done all in torch!
                            # bboxes,scores,classes = pred2box_multiclass(pred_hm[ind].cpu().detach().numpy(),
                            #                                         pred_regs[ind].cpu().detach().numpy(),512,4,thresh=0.0)
                            bboxes,scores,classes = pred2box_multiclass(pred_hm[ind].cpu().detach().numpy(),
                                                                    pred_regs[ind].cpu().detach().numpy(),512,4,thresh=0.25)

                            # print(out_size[ind].numpy().tolist(),intermediate_size[ind].numpy().tolist(),scale[ind].numpy())
                            # print("bboxes.shape: ",bboxes.shape)

                            # bboxes = np.array([bboxes[:,0],bboxes[:,1],bboxes[:,0]+bboxes[:,2],bboxes[:,1]+bboxes[:,3]]).T
                            # print("bboxes-pred.shape: ",bboxes.shape)
                            # print("bboxes-pred: ",bboxes[0])

                            # expects xywh
                            # bboxes = rescale_boxes(bboxes,out_size[ind].numpy().tolist(),intermediate_size[ind].numpy().tolist(),scale[ind].numpy() )
                            # print("bboxes_gt: ",bboxes_gt)
                            # print("bboxes: ",bboxes)

                            # print("bboxes-rescale: ",bboxes[0])
                            # bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
                            # bboxes[:,3] =  bboxes[:,3] - bboxes[:,1]
                            # print("bboxes: ",bboxes.shape)
                            bboxes,scores,classes =  filter_and_nms(bboxes,scores,classes,nms_threshold=0.6,n_top_scores=100)
                            # print("nms - bboxes: ",len(bboxes))

                            # print("bboxes: ", bboxes.shape)
                            # ---
                            # for cl in range(pred_hm.shape[1]):
                            #     h = np.dstack([pred_hm[0][cl].cpu().detach().numpy()*255]*3).astype(np.uint8)
                            #     for b,s,l in zip(bboxes,scores,classes):
                            #         # print(b,s,l)
                            #         # if b[2]>0 and b[3]>0:
                            #         if l.item() == cl:
                            #             cv2.rectangle(h,
                            #                     (int(b[0]), int(b[1])),
                            #                     (int(b[2]), int(b[3])),
                            #                     (0, 220, 0), 1)
                            #     plt.imshow(h)
                            #     plt.title("Epoch: {} Ex: {}, Class: {} preds".format(epoch,0,cl))
                            #     plt.savefig("pred_bboxes_mult_300_{}.png".format(cl))

                            # ---
                            for ind,(b,s,c) in enumerate(zip(bboxes,scores,classes)):
                                x,y,x2,y2 = b
                                # print("classes: ",c)
                                # print(" x,y,x2,y2: ", x,y,x2,y2)
                                w = x2-x
                                h = y2-y
                                # print(" x,y,w,h: ", x,y,w,h)

                                # area = abs(w*h)
                                # print("pred: ", [int(x),int(y),int(w),int(h)])
                                # print(image_id, b, s, c,area)

                                detection = {
                                        'image_id':image_id,
                                        'bbox': [int(x),int(y),int(w),int(h)],
                                        'category_id': int(c),
                                        'score': float(s),
                                    }
                                detections.append(detection)
                        except Exception as e:
                            print(e)
                            continue
    json.dump(detections, open('res.json', 'w'))
    coco_dets = val_ds.coco.loadRes('res.json')
    coco_eval = COCOeval(val_ds.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    model.eval()
    results_per_category = per_class_coco_ap(val_ds.coco,coco_eval)
    print(results_per_category)

if __name__ == '__main__':
    val_ds = COCODetectionDataset(img_dir='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial',
                    ann_json='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial/coco_shapes.json',
                    IMG_RESOLUTION=512,
                    transform=validation_transform_norm)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            collate_fn = coco_detection_collate_fn)
    
    ckpt = torch.load("centernet_300.pth",map_location='cpu')
    model = centernet(2,model_name='mv2')
    model.load_state_dict(ckpt)
    model.eval()
    # print(model)
    val(model,val_ds=val_ds,val_loader=val_loader)

