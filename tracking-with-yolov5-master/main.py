import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import tqdm
import glob
import argparse
import numpy as np

import torch
import torchvision

from sort import Sort
import time
from utils import non_max_suppression

# python detect.py --name yolo_tracker --weights yolov5s.pt --source C:/workspace/python/etc/yolov5_tracker/data/MOT17-09-raw.mp4 --save-txt --save-conf --nosave --iou-thres 1 --img 640

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--video_path', default="sample_data/MOT17-10-raw.mp4", type=str, help="video path")
    parser.add_argument('-d', '--det_dir', default="sample_data/labels",
                        type=str, help="yolov5 detect result directory")
    parser.add_argument('--max_id', default=10000, type=int, help="max id")
    parser.add_argument('--conf_thres', default=0.4, type=float)
    parser.add_argument('--iou_thres', default=0.4, type=float)
    parser.add_argument('--no_smoothing', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--show', default=True, action='store_true')

    return parser.parse_args()


def get_frame_idx(s):
    return int(os.path.split(s)[-1].split('.')[0].split('_')[-1])


def yolo_det_to_np(label_file, width, height):
    data = np.loadtxt(label_file)

    # cls_id x y w h conf => x y w h conf cls_id
    data = np.concatenate((data[:, 1:5], data[:, 5].reshape(
        len(data), 1), data[:, 0].reshape(len(data), 1)), axis=1)
    
    # x y w h => x1 y1 x2 y2
    data[:, [0,1]] -= data[:, [2,3]] / 2
    data[:, [2,3]] += data[:, [0,1]]
    
    # float pos => int pos
    data[:, [0, 2]] *= width
    data[:, [1, 3]] *= height
    
    return data


def main(args):
    
    # Generate Random Color Table
    colors = np.random.randint(0, 255, size=(args.max_id, 3), dtype=int).tolist()
    
    # Open Video File
    cap = cv2.VideoCapture(args.video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    assert cap.isOpened()
    print("Video has been successfully opened.")


    # Save Configuration
    if not args.no_save:
        result_dir = os.path.join("results", os.path.split(args.video_path)[-1].split('.')[0])
        if os.path.isdir(result_dir):
            result_dir = result_dir + "_" + str(len(os.listdir("results")))
        os.makedirs(result_dir)
        
        text_file = os.path.join(result_dir, "trackings.txt")
        video_file = os.path.join(result_dir, "trackings.mp4")
        
        ft = open(text_file, 'w')
        vw = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))


    # Load Yolo Detections
    label_files = glob.glob(os.path.join(args.det_dir, '*.txt'))
    # parse detect data
    unique_classes = set()
    #print(len(label_files))
    #label_datas = [np.array([]) for _ in range(len(label_files))]
    label_datas = [np.array([]) for _ in range(10000)]
    for label_file in label_files:
        frame_idx = get_frame_idx(label_file)
        print(frame_idx)
        #if frame_idx == 278:
        #    continue
        try:
            label_data = yolo_det_to_np(label_file, width=W, height=H)
        except:
            continue
        label_datas[frame_idx-1] = label_data
        unique_classes |= set(label_data[:,-1].astype(int))
    print(f"{len(label_datas)} detections has been successfully loaded.")
    
    # Gen SORT Tracker
    tracker = {class_id: Sort(max_age=100) for class_id in unique_classes}
    
    # Apply Custom Tracking
    if not args.no_smoothing:
        print("Applying Smoothing Method")

    stack = []
    stack.append(1)
    temp = 1
    k1 = 1
    t = time.time()
    tt = 1
    s = 0
    v = 0
    deltaV = 0
    deltaT = 0
    sv = 0
    av_speed = [0]
    index = []
    index.append(1)
    speed_out = 0
    frames = [-1]
    res_av_speed = []


    for frame_idx, data in tqdm.tqdm(enumerate(label_datas)):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # draw bboxes here
        draw = frame.copy()
        
        # conf thresh filter
        if len(data) == 0:
            continue

        data = data[data[:,4] >= args.conf_thres]

        class_ids = np.unique(data[:,5]).astype(int)

        # tracking target bboxes
        for class_id in class_ids:
            
            tracking_targets = np.empty((0, 5))
            
            class_mask = np.where(data[:,5] == class_id)

            data_part = data[class_mask].copy()
            
            bboxes    = data_part[:, :4]
            scores    = data_part[:, 4]
            
            # get iou matrix
            iou_mat   = torchvision.ops.box_iou(torch.tensor(bboxes),
                                                torch.tensor(bboxes)).numpy()
            
            # get nms filtered bbox
            nms_idxes = torchvision.ops.nms(torch.tensor(bboxes),
                                            torch.tensor(scores),
                                            torch.tensor(args.iou_thres)).numpy()
            
            if not args.no_smoothing:
                # smoothing bboxes
                for nms_idx in nms_idxes:
                    iou_mask = iou_mat[nms_idx] > 0.8
                    
                    tracking_target = np.concatenate((bboxes[iou_mask, :].mean(axis=0), 
                                                      np.array([class_id])))
                    
                    tracking_targets = np.append(tracking_targets, 
                                                 np.expand_dims(tracking_target, 0), axis=0)
            else:
                tracking_target = np.concatenate((bboxes[nms_idxes, :], np.full((len(nms_idxes), 1), class_id)), axis=1)
                tracking_targets = np.append(tracking_targets, tracking_target, axis=0)
                
            trcks = tracker[class_id].update(tracking_targets)
            vt = time.time()
            for trck in trcks:
                bbox = trck[:4].astype(int)
                obj_id = trck[-1].astype(int)
                frm = frames.pop()
                if frm == frame_idx:
                    frames.append(frm)
                    continue
                frames.append(frm)
                frames.append(frame_idx)
                temp = stack.pop()
                stack.append(temp)
                if obj_id > temp:
                    k1 = obj_id
                    stack.append(obj_id)
                    index.append(obj_id)
                    t = time.time()
                    tt = frame_idx
                #v = (obj_id - k1) * 0.42 / ((time.time()-t)*3)
                print(frame_idx)
                print(tt)
                if (frame_idx+1) % 10 == 0:
                    if len(index) > 1 and frame_idx-tt >= 10:
                        p1 = index.pop()
                        p2 = index.pop()
                        v = (p1 - p2) * 0.42 / ((frame_idx - tt) * 0.05)
                        if speed_out > v:
                            av_speed = [v]
                        index.append(p2)
                        index.append(p1)
                    if not np.isinf(v):
                        av_speed.append(v)
                    else:
                        av_speed.append(speed_out)
                    if len(av_speed) > 20:
                        av_speed.pop(0)
                    if speed_out > v:
                        speed_out = v
                elif (frame_idx + 1) % 5 == 0:
                    speed_out = sum(av_speed) / len(av_speed)
                    if speed_out > v:
                        speed_out = v

                if not np.isinf(v) and not np.isnan(v):
                    deltaV += v
                    deltaT += 1
               # print(k1)
               # print(obj_id)
                #print(frame_idx)
               # print(tt)
                #print(time.time() - t)
                res_av_speed.append(speed_out)
                s = (len(index)-1) * 0.42
                #sv += speed_out * 0.05
                print(s)
                #print(sv)
                print(v)
                print(speed_out)
                print(sum(res_av_speed) / len(res_av_speed))
                print(len(index))
               # print(deltaV/deltaT)
               # print(v*(time.time()-vt))
               # print(time.time()-testtime)
                #if not np.isnan(v*(time.time()-vt)*3) and not np.isinf(v*(time.time()-vt)*3):
                #    s += v*((time.time()-vt)*3)
               # vt = time.time()
                draw = cv2.putText(draw, str(obj_id) + " S=" + str(float('{:.2f}'.format(s))) + " V=" + str(float('{:.2f}'.format(speed_out))), (bbox[0], bbox[1]-4), 0, 1, (0, 255, 0), 2)
                draw = cv2.rectangle(draw, bbox[:2], bbox[2:4], colors[obj_id], 2)
        
            # save txt
            if not args.no_save:
                fm = np.concatenate((np.full((len(trcks), 1), frame_idx+1), 
                                     trcks,
                                     np.full((len(trcks), 1), class_id)), 
                                    axis=1).astype(int).tolist()
                string = []
                for f in fm:
                    string.append(' '.join(map(str, f))+'\n')
                string = ''.join(string)
                
                ft.write(string)
        
        # save video
        if not args.no_save:
            vw.write(draw)
            
        # show
        if args.show:
            cv2.imshow('d', draw)
            if cv2.waitKey(1) == 27:
                break
        #print(trcks)
        #print(time.time())
        print("TEST")

    cap.release()
    cv2.destroyAllWindows()
        
    if not args.no_save:
        ft.close()
        vw.release()

if __name__ == "__main__":

    args = get_args()

    main(args)
