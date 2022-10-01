import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
from utils import pred2box_multiclass, filter_and_nms, per_class_coco_ap
from tqdm import tqdm
import json
from model import centernet
from data import COCODetectionDataset, coco_detection_collate_fn, train_transform_norm, validation_transform_norm
# from val import val
from torchvision.utils import make_grid
import cv2
from compute_map import compute_map, index_pred_and_gt_by_class
from time import time
def visualize_and_report_perf(img,pred_hm,pred_regs,gt_hm,target,writer,total_ind,visualize_res):
                i = make_grid(img)
                # print(i.shape)
                if torch.cuda.is_available():
                        i = i.permute(1,2,0).cpu().detach().numpy()
                else:
                        i = i.permute(1,2,0).detach().numpy()
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_u = i*std + mean# unnormalize
                img_u = (img_u*255.).astype(np.uint8)
                if writer is not None:
                    writer.add_image('im', img_u, total_ind, dataformats='HWC')
                # print("i: ",img_u.shape)
                # im = Image.fromarray(img_u)
                # im.save('im.png')
                if torch.cuda.is_available():
                    # print("pred_hm: ", pred_hm.shape)
                    i = pred_hm.cpu().detach()[0].sum(0)#BNHW->HW
                    ii= gt_hm.cpu().detach()[0].sum(0)#BNHW->HW
                    # print("i: ", i.shape)
                else:
                    i = pred_hm.detach()[0].sum(0)
                    ii= gt_hm.detach()[0].sum(0)
                # i = make_grid(pred_hm,nrow=4).permute(1,2,0)
                # ii = make_grid(gt_hm,nrow=4).permute(1,2,0)

                # print("ii: ",i.shape)
                # if torch.cuda.is_available():
                #         i0 = i[:,:].detach().numpy()
                #         i1 = i[:,:].detach().numpy()
                #         ii0 = ii[:,:].detach().numpy()
                #         ii1 = ii[:,:].detach().numpy()
                # else:
                #         i0 = i[:,:].detach().numpy()
                #         i1 = i[:,:].detach().numpy()
                #         ii0 = ii[:,:].detach().numpy()
                #         ii1 = ii[:,:].detach().numpy()
                # print("i: ",i.shape)
                # print("ii: ",ii.shape)
                i = np.dstack([i*255]*3).astype(np.uint8)
                ii = np.dstack([ii*255]*3).astype(np.uint8)
                
                # for i in range(hm.shape[1]):

                # hm_gt = hm[0].cpu().data.numpy()[i]
                # for i in range(pred_hm.shape[1]):
                for q in range(pred_hm.shape[0]):
                    bboxes,scores,classes = pred2box_multiclass(pred_hm[q].cpu().detach().numpy(),
                                                                pred_regs[q].cpu().detach().numpy(),visualize_res,1,thresh=0.25)
                    boxes,scores,classes =  filter_and_nms(bboxes,scores,classes,nms_threshold=0.6,n_top_scores=100)
                    if torch.cuda.is_available():
                            # print(pred_hm.shape)
                            hm_pred = pred_hm[q].cpu().data.numpy()
                            hm_pred2 = hm_pred.sum(0)
                            h,w = hm_pred2.shape
                            # print("hm_pred2: ", hm_pred2.shape)
                            hm_pred = np.dstack([hm_pred2*255]*3).astype(np.uint8)

                    else:
                            hm_pred = pred_hm[q].data.numpy()
                            hm_pred2 = hm_pred.sum(0)
                            h,w = hm_pred2.shape
                        #     print("hm_pred2: ", hm_pred2.shape)
                            hm_pred = np.dstack([hm_pred2*255]*3).astype(np.uint8)
                    for b,c in zip(boxes,classes):
                            x,y,x2,y2 = [int(k) for k in b]
                            # print('pred: ',x,y,x2,y2)
                            if c == 0:
                                    # x,y,x2,y2 = [int(k) for k in b]
                                    # print(x,y)
                                    cv2.rectangle(hm_pred,(x,y),(x2,y2),(255,0,0,125),1)
                            if c == 1:
                                    # x,y,x2,y2 = [int(k) for k in b]
                                    # print(x,y)
                                    cv2.rectangle(hm_pred,(x,y),(x2,y2),(0,255,0,125),1)
                    '''# add if you want to see scaled ground truth for heatmap
                    # challenge: spec dataset is 1440x1920, versus synth dataset is 512x512
                    # easy to scale boxes for synth, but need custom scaled bbox to get scaled gt boxes
                    for b in target[q]['boxes']:
                            # b0 = b/2.0
                            # print(b)
                            # this is for original heatmap resolution (Native//4)
                        #     b = [int(k)//4 for k in b]# to allow original bbox be visualized on heatmap
                            # new efficient center net native res is 
                            b = [int(k)//2 for k in b]# to allow original bbox be visualized on heatmap

                            x,y,w,h = b
                            # print(x,y,w,h )
                            # print("gt: ",x,y,w,h)
                            # print("gt2: ",x,y,x+w,y+h)
                            cv2.rectangle(hm_pred,(x,y),(x+w,y+h),(0,0,255,125),1)
                    '''
                    print("hm_pred: ",hm_pred.shape)
                    if writer is not None:
                        writer.add_image('hm_pred_{}'.format(q), hm_pred, total_ind, dataformats='HWC')

                if writer is not None:
                    writer.add_image('pred_hm', i, total_ind, dataformats='HWC')
                    writer.add_image('gt_hm', ii, total_ind, dataformats='HWC')
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

def val(model,val_ds,val_loader,writer,epoch,visualize_res=None,IMG_RESOLUTION=None,device=None):
#     n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
    model.eval()
    # DEVICE = torch.device("cpu")
    detections = []
    gt = {}
    pred = {}
    for img, hm, reg, wh,reg_mask,inds, in_size, out_size, intermediate_size, scale,boxes_aug, targets, idxs  in tqdm(val_loader):
            # print("idxs: ",idxs)
            
            for t in targets:
                gt[t['image_id']]= {
                        'boxes':[[int(i[0]),int(i[1]),int(i[0])+int(i[2]),int(i[1])+int(i[3])] for i in  t['boxes']],
                        'classes':t['labels']
                        }

        #     print(gt)
            if device is not None:
                img = img.to(device)
                hm = hm.to(device)
                reg = reg.to(device)
                wh = wh.to(device)
                reg_mask = reg_mask.to(device)
                inds = inds.to(device)
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
                                                                    pred_regs[ind].cpu().detach().numpy(),IMG_RESOLUTION,IMG_RESOLUTION//visualize_res,thresh=0.25)

                        #     print(in_size[ind].numpy().tolist(),
                        #           out_size[ind].numpy().tolist(),
                        #           intermediate_size[ind].numpy().tolist(),
                        #           scale[ind].numpy())
                            # print("bboxes.shape: ",bboxes.shape)

                            # bboxes = np.array([bboxes[:,0],bboxes[:,1],bboxes[:,0]+bboxes[:,2],bboxes[:,1]+bboxes[:,3]]).T
                            # print("bboxes-pred.shape: ",bboxes.shape)
                            # print("bboxes-pred: ",bboxes[0])

                            # expects xywh
                            bboxes = rescale_boxes(bboxes,out_size[ind].numpy().tolist(),intermediate_size[ind].numpy().tolist(),scale[ind].numpy() )
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
                            pred[image_id] = {
                                'boxes': [],
                                'scores': [],
                                'classes': []
                            }
                            for indy,(b,s,c) in enumerate(zip(bboxes,scores,classes)):
                                x,y,x2,y2 = b
                                # print([int(x),int(y),int(x2),int(y2)])
                                # print(int(c))
                                pred[image_id]['boxes'].append([int(x),int(y),int(x2),int(y2)])
                                pred[image_id]['scores'].append(float(s))
                                pred[image_id]['classes'].append(int(c))
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
                        break
    

    h,w =in_size[ind].numpy()
#     print("h,w: ",h,w)
    test_pred = np.zeros((int(h),int(w),3)).astype(np.uint8)
    for b,c in zip(bboxes,classes):
        x,y,x2,y2 = [int(k) for k in b]
        # print('pred: ',x,y,x2,y2)
        if c == 0:
                # x,y,x2,y2 = [int(k) for k in b]
                # print(x,y)
                cv2.rectangle(test_pred,(x,y),(x2,y2),(255,0,0),3)
        if c == 1:
                # x,y,x2,y2 = [int(k) for k in b]
                # print(x,y)
                cv2.rectangle(test_pred,(x,y),(x2,y2),(0,255,0),3)
        if c == 2:
                # x,y,x2,y2 = [int(k) for k in b]
                # print(x,y)
                cv2.rectangle(test_pred,(x,y),(x2,y2),(255,255,0),3)
        if c == 3:
                # x,y,x2,y2 = [int(k) for k in b]
                # print(x,y)
                cv2.rectangle(test_pred,(x,y),(x2,y2),(0,255,255),3)
    for b in targets[ind]['boxes']:
            # b0 = b/2.0
            # print(b)
            b = [int(k) for k in b]
            x,y,w,h = b
        #     print("gt: ",x,y,x+w,y+h )
            # print("gt: ",x,y,w,h)
            # print("gt2: ",x,y,x+w,y+h)
            cv2.rectangle(test_pred,(x,y),(x+w,y+h),(0,0,255),3)
    if writer is not None:
        writer.add_image('test_pred', test_pred, epoch, dataformats='HWC')
        visualize_and_report_perf(img,
                                pred_hm=pred_hm,
                                pred_regs=pred_regs,
                                gt_hm=hm,
                                target = targets,
                                writer=writer,
                                total_ind=epoch,
                                visualize_res=visualize_res)
    print(len(detections))
    if len(detections) > 0:
        json.dump(detections, open('res.json', 'w'))
        coco_dets = val_ds.coco.loadRes('res.json')
        coco_eval = COCOeval(val_ds.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        model.eval()
        results_per_category = per_class_coco_ap(val_ds.coco,coco_eval)
        print(results_per_category)
        # [('circle', '0.006'), ('rectangle', '0.001')]
        cls_names = []
        for i in results_per_category:
            cls_name = i[0]
            cls_names.append(i[0])
            mAP_val = float(i[1])
            print(cls_name,mAP_val)
            if writer is not None:
                writer.add_scalar("{} coco mAP/val".format(cls_name), mAP_val, epoch)
        print("Computing mAP (Non COCO)...")
        t0 = time()
        maps = []
        maps_50 = []
        for c_ind,c in enumerate(tqdm(cls_names)):
                p,g = index_pred_and_gt_by_class(pred,gt,c_ind)
                mAP = compute_map(g,p,c_ind)
                maps.append(mAP[c_ind])
                maps_50.append(mAP[str(c_ind)+'_50'])
                print("{}-mAP: {}".format(c,mAP[c_ind]))
                print("{}-mAP_50: {}".format(c,mAP[str(c_ind)+'_50']))
                if writer is not None:
                        writer.add_scalar("{} mAP/val".format(c), mAP[c_ind], epoch)
                        writer.add_scalar("{} mAP/val".format(c+'_50'), mAP[str(c_ind)+'_50'], epoch)
        print("mAP calculation completed! Time: {}".format(time() - t0))
        mean_mAP = np.array(maps).mean()
        mean_mAP_50 = np.array(maps_50).mean()
        print("mean mAP: {}".format(mean_mAP))
        print("mean mAP 50: {}".format(mean_mAP_50))
        if writer is not None:
                writer.add_scalar("mAP/val".format(c), mean_mAP, epoch)
                writer.add_scalar("mAP_50/val".format(c), mean_mAP_50, epoch)
        # torch.set_num_threads(n_threads)
    # writer.add_scalar("Loss/train", loss.item(), total_ind)
    #                     writer.add_scalar("Mask Loss/train", mask_loss.item(), total_ind)
    #                     writer.add_scalar("Reg Loss/train", regr_loss.item(), total_ind)

if __name__ == '__main__':
    # val_ds = COCODetectionDataset(img_dir='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial',
    #                 ann_json='/Users/mendeza/Documents/projects/cent-tutorial/centernet-tutorial/tutorial/coco_shapes.json',
    #                 IMG_RESOLUTION=512,
    #                 transform=validation_transform_norm)
    # val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/',
    #                      '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/shapes_dataset/coco_shapes.json',
    #                      transform=validation_transform_norm)
    val_ds = COCODetectionDataset('/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/fruit_specs_dataset/images',
        '/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/fruit_specs_dataset/annotations/coco-specs-fruit.json',
        transform=validation_transform_norm,
        IMG_RESOLUTION=512)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            collate_fn = coco_detection_collate_fn)
    
    ckpt = torch.load("centernet_300.pth",map_location='cpu')
    model = centernet(val_ds.num_classes,model_name='mv2')
    model.load_state_dict(ckpt)
    model.eval()
    # print(model)
    val(model,val_ds=val_ds,val_loader=val_loader,writer=None,epoch=0)

