import argparse
import math
import time

import numpy as np
import torch
import torch.distributed as dist

from layers import ModelParallelResnetGenerator, AttributeParallelConv2d
from networks import ResnetGenerator
from conf import get_conf
import pickle

def get_args():
    parser = argparse.ArgumentParser("test resnet-generator")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--random-seed', type=int, default=1234)
    parser.add_argument("--ngf", type=int, default=64, help="residual block in&out channels")
    parser.add_argument('--repeat', type=int, default=200)
    parser.add_argument('--warm-up', type=int, default=100)

    return parser.parse_args()

conv_event_lst = []

def print_conv_avg_time():
    ts = []
    for s, e in conv_event_lst:
        e.synchronize()
        elp_ms = s.elapsed_time(e)
        ts += [elp_ms]
    print('avg conv takes (ms)', np.mean(ts[-200:]))


def record_conv_layer(module):
    childs = list(module.children())

    def _create_start_record(mod, input):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        conv_event_lst.append([start_event, end_event])
        start_event.record()

    def _after_fwd(mod, input, output):
        conv_event_lst[-1][1].record()
    
    if isinstance(module, torch.nn.Conv2d):
        if module.kernel_size[0] > 1:
            module.register_forward_pre_hook(_create_start_record)
            module.register_forward_hook(_after_fwd)

    if len(childs) == 0:
        return
    else:
        for c in childs:
            record_conv_layer(c)


def main():
    """"""
    # pylint: disable=no-member
    args = get_args()
    
    # init cuda
    torch.rand(100).cuda().sum() # pylint: disable=no-member

    bs = args.batch_size
    s = args.img_size
    orig_gen = ResnetGenerator(3, 3, args.ngf, 4, args.img_size, random_seed=args.random_seed)
    orig_gen = orig_gen.cuda()
    record_conv_layer(orig_gen)

    print('repeat', args.repeat)
    print('warmup', args.warm_up)
    print('img-size', args.img_size)
    print('batch-size', bs)
    print('ngf', args.ngf)

    torch.manual_seed(args.random_seed)

    ts = []
    for i in range(args.repeat):
        rand_img = torch.rand((bs, 3, s, s)).cuda()

        torch.cuda.synchronize()
        _start = time.time()
        output_img = orig_gen(rand_img)
        torch.cuda.synchronize()
        _end = time.time()

        if i < args.warm_up:
            ts.append(_end - _start)
    # check intermediate outputs are close

    print( "average fwd time", np.mean(ts)*1e3, 'ms')
    print_conv_avg_time()

if __name__ == "__main__":
    main()
