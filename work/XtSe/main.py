# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import argparse
from paddlevideo.utils import get_config
from paddlevideo.tasks import train_model, train_model_multigrid, test_model, train_dali
from paddlevideo.utils import get_dist_info


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo train script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        # default='configs/recognition/stgcn/stgcn_fsd.yaml',
                        default='configs/recognition/stgcn/stgcn_fsd.yaml',
                        help='config file path')
    parser.add_argument('-f',
                        '--folder',
                        type=str,
                        default="/aiot_nfs/jzz_data/paddle_jzz/data/5folder_Resample_origin",
                        help="root path of data")
    parser.add_argument('-m',
                        '--mtype',
                        type=str,
                        default="XtSe",
                        help="type of model")
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default="K5_Origin",
                        help="name of save model")
    parser.add_argument('-l',
                    '--lr',
                    type=float,
                    default=0.1,
                    help="learning rate")
    parser.add_argument('-e',
                    '--epoch',
                    type=int,
                    default=180,
                    help="total epochs")
    parser.add_argument('-k',
                        '--kfolder',
                        type=int,
                        default=0)
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('--test',
                        action='store_true',
                        help='whether to test a model')
    parser.add_argument('--train_dali',
                        action='store_true',
                        help='whether to use dali to speed up training')
    parser.add_argument('--multigrid',
                        action='store_true',
                        help='whether to use multigrid training')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        help='weights for finetuning or testing')
    parser.add_argument('--fleet',
                        action='store_true',
                        help='whether to use fleet run distributed training')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether to open amp training.')

    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override)
    cfg.MODEL.backbone["mtype"] = args.mtype
    cfg["epochs"] = args.epoch
    cfg.OPTIMIZER.learning_rate["T_max"] = args.epoch
    cfg.OPTIMIZER.learning_rate["learning_rate"] = args.lr

    _, world_size = get_dist_info()
    parallel = world_size != 1
    if parallel:
        paddle.distributed.init_parallel_env()

    if args.test:
        test_model(cfg, weights=args.weights, out_name=args.name, k=args.kfolder, parallel=parallel)
    elif args.train_dali:
        train_dali(cfg, weights=args.weights, parallel=parallel)
    elif args.multigrid:
        train_model_multigrid(cfg, world_size, validate=args.validate)
    else:
        train_model(cfg,
                    data_path=args.folder,
                    num_folder=args.kfolder,
                    model_name=args.name,
                    weights=args.weights,
                    parallel=parallel,
                    validate=args.validate,
                    use_fleet=args.fleet,
                    amp=args.amp)


if __name__ == '__main__':
    main()
