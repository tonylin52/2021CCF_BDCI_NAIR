import os.path as osp
import pickle
import numpy as np

from .base1 import BaseDataset1
from ..registry import DATASETS


@DATASETS.register()
class PoseDataset(BaseDataset1):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        keypoint_file (str): Path to the data file.
        label_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose.
            For a video with n frames, it is a valid training sample only if
            n * valid_ratio frames have human pose. None means not applicable
            (only applicable to Kinetics Pose). Default: None.
        box_thr (str | None): The threshold for human proposals. Only boxes
            with confidence score larger than `box_thr` is kept. None means
            not applicable (only applicable to Kinetics Pose [ours]). Allowed
            choices are '0.5', '0.6', '0.7', '0.8', '0.9'. Default: None.
        class_prob (dict | None): The per class sampling probability. If not
            None, it will override the class_prob calculated in
            BaseDataset.__init__(). Default: None.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 keypoint_file,
                 label_file,
                 conf_thres,
                 pipeline,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 **kwargs):
        modality = 'Pose'

        super().__init__(
            keypoint_file, label_file,conf_thres, pipeline, start_index=0, modality=modality, **kwargs)

        # box_thr, which should be a string
        self.box_thr = box_thr
        if self.box_thr is not None:
            assert box_thr in ['0.5', '0.6', '0.7', '0.8', '0.9']

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None:
            assert isinstance(self.valid_ratio, float)
            if self.box_thr is None:
                self.video_infos = self.video_infos = [
                    x for x in self.video_infos
                    if x['valid_frames'] / x['total_frames'] >= valid_ratio
                ]
            else:
                key = f'valid@{self.box_thr}'
                self.video_infos = [
                    x for x in self.video_infos
                    if x[key] / x['total_frames'] >= valid_ratio
                ]
                if self.box_thr != '0.5':
                    box_thr = float(self.box_thr)
                    for item in self.video_infos:
                        inds = [
                            i for i, score in enumerate(item['box_score'])
                            if score >= box_thr
                        ]
                        item['anno_inds'] = np.array(inds)

        if class_prob is not None:
            self.class_prob = class_prob

        # logger = get_root_logger()
        # logger.info(f'{len(self)} videos remain after valid thresholding')

    def load_annotations(self):
        """Load annotation file to get video information."""
        
        return self.load_npy_annotations()

    def load_npy_annotations(self):
        # data = mmcv.load(self.ann_file)
        # fr0 = open(self.ann_file, "rb+") 
        # data = pickle.load(fr0)
        
        toral_frame_list = []
        data = []
        if not self.test_mode:
            label = np.load(self.label_file)
        traindata = np.load(self.keypoint_file)
        
        #traindata = np.transpose(traindata,(0,3,1,2))
        #traindata = np.expand_dims(traindata,axis=-1)
        if traindata.ndim!=5:
            print("######################",traindata.shape)
            traindata = np.expand_dims(traindata,axis=-1)
        print("!!!!!!!!!",traindata.shape)
        
        for i in range(traindata.shape[0]):
            
            w,h = 400,400
            res = {}
                        
            if not self.test_mode:
                res["label"] = int(label[i])
            else:
                res["label"] = 0
            
            res["img_shape"] = (h,w)
            res["original_shape"] = (h,w)
            res["keypoint"] = np.transpose(traindata[i][:2,], (3,1,2,0))
            sum_ = np.sum(res["keypoint"],axis=(0,2,3))
            id_ = np.where(sum_!=0)[0]
            res["keypoint"] = res["keypoint"][:,id_,...]
            res["keypoint_score"] = np.transpose(traindata[i][2,], (2, 0, 1))
            res["keypoint_score"] = res["keypoint_score"][:,id_,:]
            
            max_ = np.max(res["keypoint_score"], axis=(0, 2))
            id_1 = np.where(max_ > self.conf_thres)[0]
            res["keypoint_score"] = res["keypoint_score"][:, id_1, :]
            res["keypoint"] = res["keypoint"][:, id_1, ...]
                    
            res["keypoint"][:,:,:,0] = w/2 * (1+res["keypoint"][:,:,:,0])
            res["keypoint"][:,:,:,1] = h/2 * (1+res["keypoint"][:,:,:,1])

            res["total_frames"] = res["keypoint"].shape[1]
            # print(res["keypoint"].shape,res["keypoint_score"].shape)
            
            
            # video_i = traindata[i,...]
            # keypoint = np.zeros([1,0,25,2],dtype=np.float32)
            # keypoint_score = np.zeros([1, 0, 25],dtype=np.float32)
            # for j in range(video_i.shape[1]):
            #     frame_i = video_i[:,j, ...]
            #     sum_ = np.sum(frame_i)
            #     if sum_==0.0:
            #         break
            #     # print(sum_)
            #     # im = 255 * np.ones([800, 800], dtype=np.uint8)
            #     keypoint_i = np.zeros([1, 1, 25, 2],dtype=np.float32)
            #     keypoint_score_i = np.zeros([1, 1, 25],dtype=np.float32)

            #     for m in range(frame_i.shape[1]):
            #         point_i = frame_i[:,m,:]
            #         # print(point_i.shape[-1])
            #         keypoint_i[0,0,m,0] = (point_i[0, 0] + 1)*100
            #         keypoint_i[0, 0, m, 1] = (point_i[1, 0] + 1) * 100
            #         keypoint_score_i[0,0,m] = point_i[2, 0]

 

            #     keypoint = np.append(keypoint, keypoint_i, axis=1)
            #     keypoint_score = np.append(keypoint_score, keypoint_score_i, axis=1)

            # frame_dir = str(i) + "_rgb"
            # image_shape = (200, 200)
            # original_shape = (200, 200)
            # total_frame = keypoint.shape[1]
    
            # toral_frame_list.append(total_frame)
            # res = {}
            # res["total_frames"] = total_frame
            # res["img_shape"] = image_shape
            # res["original_shape"] = original_shape
            # res["keypoint"] = keypoint
            # res["keypoint_score"] = keypoint_score
            # res["frame_dir"] = frame_dir
            # print(res["keypoint"].shape)
            

            data.append(res)

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
        return data
