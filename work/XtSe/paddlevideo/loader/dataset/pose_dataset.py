import os.path as osp
import pickle
import numpy as np

from .base1 import BaseDataset1
from ..registry import DATASETS

COCO_idx = []

@DATASETS.register()
class PoseDataset(BaseDataset1):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
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
                 pipeline,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 **kwargs):
        modality = 'Pose'

        super().__init__(
            keypoint_file, label_file, pipeline, start_index=0, modality=modality, **kwargs)

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
            data=  np.load(self.keypoint_file)
            if not self.test_mode:
                label = np.load(self.label_file) 
            
            final = []
            print("Convert Start")
            for i in range(data.shape[0]):
                tmp = {}
                tmp['label'] = label[i] if not self.test_mode else 0
                w, h = 200, 200
                tmp["img_shape"] = (h, w)
                tmp["original_shape"] = (h, w)
                
                tmp["keypoint"] = np.transpose(data[i][:2,], (3,1,2,0))
                sum_ = np.sum(tmp['keypoint'], axis=(0,2,3))
                id_  = np.where(sum_!=0)[0]
                tmp['keypoint'] = tmp['keypoint'][:,id_,...]
                tmp["keypoint"][:,:,:,0] = w/2 * (1+tmp["keypoint"][:,:,:,0])
                tmp["keypoint"][:,:,:,1] = h/2 * (1+tmp["keypoint"][:,:,:,1])
                tmp["keypoint_score"] = np.transpose(data[i][2,], (2, 0, 1))
                tmp["keypoint_score"] = tmp["keypoint_score"][:,id_,:]
                # tmp["total_frames"] = data[i].shape[1]
                tmp["total_frames"] = tmp["keypoint"].shape[1]
                
                final.append(tmp)
            print("Convert Success")
            return final
