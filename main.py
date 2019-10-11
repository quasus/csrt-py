import os
import sys
from lib.utils import get_ground_truthes, plot_precision, plot_success

import cv2
import numpy as np
from lib.utils import get_img_list,get_ground_truthes,APCE,PSR
from csrt import CSRDCF, DEFAULT_PARAMS, DEFAULT_SCALE_PARAMS


class PyTracker:
    def __init__(self,img_dir,tracker_type=None, tracker_params=DEFAULT_PARAMS,
                 dataset_config=None, max_frames=None):
        self.img_dir=img_dir
        self.tracker_type=tracker_type
        self.frame_list = get_img_list(img_dir)
        self.frame_list.sort()
        if max_frames:
            self.frame_list = self.frame_list[:max_frames]
        dataname=img_dir.split('/')[-2]
        self.gts=get_ground_truthes(img_dir[:-4])
        self.tracker=CSRDCF(config=tracker_params)
        self.init_gt=self.gts[0]

    def tracking(self,verbose=True,video_path=None):
        poses = []
        init_frame = cv2.imread(self.frame_list[0])
        #print(init_frame.shape)
        init_gt = np.array(self.init_gt)
        x1, y1, w, h =init_gt
        init_gt=tuple(init_gt)
        self.tracker.init(init_frame,init_gt)
        writer=None
        if verbose is True and video_path is not None:
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (init_frame.shape[1], init_frame.shape[0]))

        fpss = []

        for frame_name in self.frame_list[1:]:
            timer = cv2.getTickCount()
            current_frame=cv2.imread(frame_name)
            height,width=current_frame.shape[:2]
            bbox=self.tracker.update(current_frame,vis=verbose)
            x1,y1,w,h=bbox
            if verbose:
                if len(current_frame.shape)==2:
                    current_frame=cv2.cvtColor(current_frame,cv2.COLOR_GRAY2BGR)
                score = self.tracker.score
                apce = APCE(score)
                psr = PSR(score)
                F_max = np.max(score)
                size=self.tracker.crop_size
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
        return np.array(poses)


if __name__ == '__main__':

    #if len(sys.argv) > 3:
    #    csrdcf.MAX_ITER = 2
    #    csrdcf.TARGET_SIZE = 64

    #print("MI = " + str(csrdcf.MAX_ITER))
    #print("TS = " + str(csrdcf.TARGET_SIZE))

    scale_params = DEFAULT_SCALE_PARAMS

    #scale_params['number_of_scales_filter'] = 5 # default: 33
    #scale_params['number_of_interp_scales'] = 5 # default: 33
    #scale_params['scale_step_filter']  = 1.02  # this is the default; probably should be changed

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
                        max_frames=100)
    poses=tracker.tracking(verbose=True)
    #plot_success(gts,poses,os.path.join('../results/CF',data_name+'_success.jpg'))
    #plot_precision(gts,poses,os.path.join('../results/CF',data_name+'_precision.jpg'))
