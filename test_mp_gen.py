import argparse
import math
import time

import numpy as np
import torch
import torch.distributed as dist

from layers import ModelParallelResnetGenerator
from networks import ResnetGenerator
from conf import get_conf

def get_args():
    parser = argparse.ArgumentParser("test model parallel resnet-generator")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument('--rank0-ip', type=str, default='localhost')
    parser.add_argument('--rank0-port', type=str, default='6666')
    parser.add_argument('--async', type=bool, default=False)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--random-seed', type=int, default=1234)
    parser.add_argument("--ngf", type=int, default=64, help="residual block in&out channels")

    return parser.parse_args()


def main():
    """"""
    args = get_args()
    torch.manual_seed(args.random_seed)
    # init cuda
    torch.rand(100).cuda().sum() # pylint: disable=no-member

    dist.init_process_group(backend="nccl",
                            init_method="tcp://{}:{}".format(
                                args.rank0_ip, args.rank0_port),
                            world_size=args.world_size,
                            rank=args.rank)

    bs = args.batch_size
    s = args.img_size
    mp_plan = get_conf(args.rank, args.img_size)
    mp_gen = ModelParallelResnetGenerator(mp_plan, 3, 3, args.ngf, 4, args.img_size).cuda()
    orig_gen = ResnetGenerator(3, 3, args.ngf, 4, args.img_size)

    rand_img = torch.rand((bs, 3, s, s)) # pylint: disable=no-member
    can_output = orig_gen(rand_img)
    output_img = mp_gen(rand_img.cuda())
    print(torch.allclose(can_output, output_img)) # pylint: disable=no-member

if __name__ == "__main__":
    main()
