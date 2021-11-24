import os.path as osp
import pickle
# import mmcv
import numpy as np
import os
import cv2

# from ..utils import get_root_logger
from .base1 import BaseDataset1
from ..registry import DATASETS


@DATASETS.register()
class MnistDataset(BaseDataset1):
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
                 ann_file,
                 pipeline,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 **kwargs):
        modality = 'Pose'

        super().__init__(
            ann_file, pipeline, start_index=0, modality=modality, **kwargs)

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
        # assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        # data = mmcv.load(self.ann_file)
        # fr0 = open(self.ann_file, "rb+") 
        data = []
        
        for path_i in os.listdir(self.ann_file):
            self.path_ = os.path.join(self.ann_file,path_i)
            label = int(path_i)
            for path_j in os.listdir(self.path_):
                img_path = os.path.join(self.path_,path_j)
                img = cv2.imread(img_path)
                results = dict()
                results['label'] = label
                img = np.expand_dims(img,axis = 0)
                for i in range(3):
                    img = np.concatenate([img,img],axis=0)
                results['num_clips'] = 1
                results['clip_len'] = 2**3
                results['imgs'] = img
                results['input_shape'] = img.shape
                data.append(results)
                

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
        return data
