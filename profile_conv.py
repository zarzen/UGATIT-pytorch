from networks import ResnetBlock, ResnetAdaILNBlock
import argparse
import torch
import torch.nn as nn
import time
import numpy as np
import sys

# def record_gpu():


def get_args():
    parser = argparse.ArgumentParser("profiler args")
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--dim", type=int, default=256, help="residual block in&out channels")
    parser.add_argument("--img-sizes", type=str, default="32-64-128", help="profile on different sizes of inputs")
    parser.add_argument("--block-type", type=str, default="resnet", help="resnet/ailn")
    parser.add_argument("--sep-fwd", type=bool, default=False)

    return parser.parse_args()

def main():
    """"""
    args = get_args()
    exp_sizes = [int(s) for s in args.img_sizes.split('-')]
    bs = 1

    blocks = []
    for _ in range(4):
        if args.block_type == "resnet":
            blocks += [ResnetBlock(args.dim, use_bias=False)]
            down_or_up = nn.Sequential(*blocks)
            down_or_up = down_or_up.cuda()
        elif args.block_type == "ailn":
            blocks += [ResnetAdaILNBlock(args.dim, use_bias=False).cuda()]
        else:
            sys.exit("wrong block type")

    

    for s in exp_sizes:
        
        ts = []
        fwd_ts = []
        for _ in range(args.repeat):
            one_batch = torch.rand((bs, args.dim, s, s)) # pylint: disable=no-member
            one_batch = one_batch.cuda()
            if args.block_type == 'ailn':
                # pylint: disable=no-member
                gamma = torch.rand((bs, args.dim)).cuda()
                beta = torch.rand((bs, args.dim)).cuda()
            else:
                gamma = None
                beta = None

            torch.cuda.synchronize()
            _start_time = time.time()

            if args.block_type == 'ailn':
                outputs = one_batch
                for b in blocks:
                    outputs = b(outputs, gamma, beta)
            else:
                outputs = down_or_up(one_batch)
    
            fake_loss = outputs.sum()

            if args.sep_fwd:
                torch.cuda.synchronize()
                fwd_ts.append(time.time() - _start_time)

            fake_loss.backward()
            torch.cuda.synchronize()
            ts.append(time.time() - _start_time)
        if not args.sep_fwd:
            print("img-size {}, average fwd bwd time {} ms".format(s, np.mean(ts[-100:]) * 1e3))
        else:
            print("img-size {}, average fwd {} ms; fwd&bwd time {} ms".format(s, np.mean(fwd_ts[-100:])*1e3, np.mean(ts[-100:]) * 1e3))


if __name__ == "__main__":
    main()
