from networks import ResnetGenerator
import argparse
import torch
import torch.nn as nn
import time
import numpy as np
import sys

# def record_gpu():


def get_args():
    parser = argparse.ArgumentParser("profiler args")
    parser.add_argument("--repeat", type=int, default=150)
    parser.add_argument("--ngf", type=int, default=64, help="residual block in&out channels")
    parser.add_argument('--res_n', type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256, help="profile on different sizes of inputs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sep-fwd", type=bool, default=False)

    return parser.parse_args()

def main():
    """"""
    args = get_args()

    bs = args.batch_size

    s = args.img_size
    gen = ResnetGenerator(3, 3, args.ngf, args.res_n, s)
    opt = torch.optim.Adam(gen.parameters())
    loss_fn = torch.nn.MSELoss().cuda()
    gen = gen.cuda()

    ts = []
    fwd_ts = []
    for _ in range(args.repeat):
        # pylint: disable=no-member
        one_batch = torch.rand((bs, 3, s, s)).cuda()
        rand_img = torch.rand((bs, 3, s, s)).cuda()
        rand_img2 = torch.rand((bs, 3, s, s)).cuda()

        opt.zero_grad()
        torch.cuda.synchronize()
        _start_time = time.time()

        fake_img, _, _ = gen(one_batch)

        loss1 = loss_fn(fake_img, rand_img)
        loss2 = loss_fn(fake_img, rand_img2)
        loss = loss1 + loss2

        fwd_ts += [time.time() - _start_time]
        if args.sep_fwd:
            torch.cuda.synchronize()
            fwd_ts.append(time.time() - _start_time)

        loss.backward()

        opt.step()
        torch.cuda.synchronize()
        ts.append(time.time() - _start_time)
        

    print("img-size {}, average fwd {:.4f} ms; fwd&bwd time {:.4f} ms".format(s, np.mean(fwd_ts[-100:])*1e3, np.mean(ts[-100:]) * 1e3))


if __name__ == "__main__":
    main()
