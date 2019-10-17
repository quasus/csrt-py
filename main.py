import itertools
import os
import sys
from lib.utils import get_ground_truthes, plot_precision, plot_success

import cv2
import numpy as np
from lib.utils import get_img_list,get_ground_truthes,APCE,PSR
from csrt import CSRDCF, DEFAULT_PARAMS, DEFAULT_SCALE_PARAMS


class PyTracker:
    def __init__(self,img_dir,tracker_type=None, tracker_params=DEFAULT_PARAMS,
                 dataset_config=None, max_frames=None, IOUmin=0.33):
        self.img_dir=img_dir
        self.tracker_type=tracker_type
        self.frame_list = get_img_list(img_dir)
        self.frame_list.sort()
        if max_frames:
            self.frame_list = self.frame_list[:max_frames]
        dataname=img_dir.split('/')[-2]

        self.gts=get_ground_truthes(img_dir[:-4])
        frame_list_len = len(self.frame_list)
        gts_len = len(self.gts)
        if gts_len > frame_list_len:
            self.gts = self.gts[:frame_list_len]
        elif gts_len < frame_list_len:
            self.gts += [None] * (frame_list_len - gts_len)

        self.tracker=CSRDCF(config=tracker_params)
        self.init_gt=self.gts[0]

        self.IOUmin = IOUmin

    def updating_histograms(self):
        init_frame = cv2.imread(self.frame_list[0])
        init_gt = np.array(self.init_gt)
        self.tracker.init(init_frame,init_gt)
        for _ in range(100):
            self.tracker.update_histograms(init_frame,
                                           [100,
                                            100,
                                            self.tracker.target_sz[0],
                                            self.tracker.target_sz[1]])

    def tracking(self,verbose=True,video_path=None):
        poses = []
        init_frame = cv2.imread(self.frame_list[0])
        init_gt = np.array(self.init_gt)
        x1, y1, w, h =init_gt
        init_gt=tuple(init_gt)
        self.tracker.init(init_frame,init_gt)
        writer=None
        if verbose is True and video_path is not None:
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (init_frame.shape[1], init_frame.shape[0]))

        fpss = []
        ious = []

        for i, (frame_name, gt) in enumerate(zip(self.frame_list[1:], self.gts[1:])):
            timer = cv2.getTickCount()
            current_frame=cv2.imread(frame_name)
            height,width=current_frame.shape[:2]
            bbox=self.tracker.update(current_frame,vis=verbose)

            if gt is not None:
                iou = IOU(bbox, gt)
                if iou < self.IOUmin:
                    break
                ious.append(iou)
                if i == 100:
                    print("Mean IOU (100 frames): {}".format(sum(ious)/len(ious)))

            x1,y1,w,h=bbox
            if verbose:
                if len(current_frame.shape)==2:
                    current_frame=cv2.cvtColor(current_frame,cv2.COLOR_GRAY2BGR)
                score = self.tracker.score
                apce = APCE(score)
                psr = PSR(score)
                F_max = np.max(score)
                size=self.tracker.template_size
                score = cv2.resize(score, size)
                score -= score.min()
                score =score/ score.max()
                score = (score * 255).astype(np.uint8)
                # score = 255 - score
                score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
                center = (int(x1+w/2),int(y1+h/2))
                x0,y0=center
                x0=np.clip(x0,0,width-1)
                y0=np.clip(y0,0,height-1)
                center=(x0,y0)
                xmin = int(center[0]) - size[0] // 2
                xmax = int(center[0]) + size[0] // 2 + size[0] % 2
                ymin = int(center[1]) - size[1] // 2
                ymax = int(center[1]) + size[1] // 2 + size[1] % 2
                left = abs(xmin) if xmin < 0 else 0
                xmin = 0 if xmin < 0 else xmin
                right = width - xmax
                xmax = width if right < 0 else xmax
                right = size[0] + right if right < 0 else size[0]
                top = abs(ymin) if ymin < 0 else 0
                ymin = 0 if ymin < 0 else ymin
                down = height - ymax
                ymax = height if down < 0 else ymax
                down = size[1] + down if down < 0 else size[1]
                score = score[top:down, left:right]
                crop_img = current_frame[ymin:ymax, xmin:xmax]
                score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
                current_frame[ymin:ymax, xmin:xmax] = score_map
                show_frame=cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0),1)
                if gt is not None:
                    x0, y0, w0, h0 = gt
                show_frame=cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0),1)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                fpss.append(fps)
                cv2.putText(show_frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                """
                cv2.putText(show_frame, 'APCE:' + str(apce)[:5], (0, 250), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (0, 0, 255), 5)
                cv2.putText(show_frame, 'PSR:' + str(psr)[:5], (0, 300), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (255, 0, 0), 5)
                cv2.putText(show_frame, 'Fmax:' + str(F_max)[:5], (0, 350), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (255, 0, 0), 5)
                """

                cv2.imshow('demo', show_frame)
                if writer is not None:
                    writer.write(show_frame)
                cv2.waitKey(1)


        poses.append(np.array([int(x1), int(y1), int(w), int(h)]))
        print("FPS: " + str(sum(fpss)/len(fpss)))
        print("Lost after {} frames".format(len(ious)))
        return np.array(poses)


def intersect_intervals(*intervals):
    a = max(interval[0] for interval in intervals)
    b = min(interval[1] for interval in intervals)
    return (a, b) if a < b else None


def intersect_boxes(*boxes):
    starts_lengths_by_coord = tuple(zip(*(((x, w), (y, h)) for (x, y, w, h) in boxes)))
    sides_by_coord = tuple([(start, start + length) for (start, length) in starts_lengths]
                           for starts_lengths in starts_lengths_by_coord)
    intersections_by_coord = tuple(intersect_intervals(*sides) for sides in sides_by_coord)
    if all(intersections_by_coord):
        (x0, x1), (y0, y1) = intersections_by_coord
        return (x0, y0, x1 - x0, y1 - y0)
    else:
        return None


def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    i = intersect_boxes(box1, box2)
    if i:
        x, y, w, h = intersect_boxes(box1, box2)
        S1 = w1 * h1
        S2 = w2 * h2
        Sint = w * h
        return Sint/(S1 + S2 - Sint)
    else:
        return 0


if __name__ == '__main__':

    #if len(sys.argv) > 3:
    #    csrdcf.MAX_ITER = 2
    #    csrdcf.TARGET_SIZE = 64

    #print("MI = " + str(csrdcf.MAX_ITER))
    #print("TS = " + str(csrdcf.TARGET_SIZE))

    scale_params = DEFAULT_SCALE_PARAMS

    #scale_params['number_of_scales_filter'] = 5 # default: 33
    #scale_params['number_of_interp_scales'] = 5 # default: 33
    #scale_params['scale_step_filter']  = 1.14  # 2^(1/5)

    #scale_params['number_of_scales_filter'] = 20 # default: 33
    #scale_params['number_of_interp_scales'] = 20 # default: 33
    #scale_params['scale_step_filter']  = 1.03  # 2^(1/20)

    params = DEFAULT_PARAMS
    params['scale_params'] = scale_params

    #params['admm_iterations'] = 2 # default: 4
    #params['template_size'] = 64 # default: 200
    #params['top_channels'] = 7 # default: None; less is riskier

    data_path=sys.argv[1]
    gts = get_ground_truthes(data_path)
    img_dir = os.path.join(data_path, 'img')
    tracker = PyTracker(img_dir,
                        tracker_params=params,
                        max_frames=None,
                        IOUmin=0.33
                        )
    poses=tracker.tracking(verbose=True)
    #tracker.updating_histograms()
    #plot_success(gts,poses,os.path.join('../results/CF',data_name+'_success.jpg'))
    #plot_precision(gts,poses,os.path.join('../results/CF',data_name+'_precision.jpg'))
