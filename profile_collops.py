import torch.distributed as dist
import torch
import argparse
import math
import time
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size-order', type=int, default=23)
    parser.add_argument('--op', type=str, default='allreduce',
                        help="allreduce/allgather")
    parser.add_argument('--repeat', type=int, default=100)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--rank0-ip', type=str, default='localhost')
    parser.add_argument('--rank0-port', type=str, default='6666')
    parser.add_argument('--asyn', type=int, default=0)

    return parser.parse_args()


def main():
    """"""
    args = get_args()
    if args.asyn > 0:
        args.asyn = True
    else:
        args.asyn = False
    # init cuda
    torch.rand(100).cuda().sum()

    dist.init_process_group(backend="nccl",
                            init_method="tcp://{}:{}".format(
                                args.rank0_ip, args.rank0_port),
                            world_size=args.world_size,
                            rank=args.rank)

    for o in range(args.size_order):
        t_deltas = []
        func_ret = []
        s = int(math.pow(2, o))
        for _ in range(args.repeat):
            t = torch.rand((s,)).cuda()
            t_list = []
            if args.op == 'allgather':
                for _ in range(args.world_size):
                    t_list.append(torch.zeros_like(t).cuda())

            t1 = time.time()
            if args.op == 'allreduce':
                h = dist.all_reduce(t, async_op=args.asyn)
            elif args.op == 'allgather':
                h = dist.all_gather(t_list, t, async_op=args.asyn)
            else:
                return -1
            
            func_ret += [time.time() - t1]
            if args.asyn:
                while not h.is_completed():
                    pass
            torch.cuda.synchronize()
            t2 = time.time()
            t_deltas += [t2 - t1]

        if args.rank == 0:
            print('async', args.asyn, "size {}; op {} takes {:.3f} ms; func ret {:.3f} ms".format(s, args.op,
                                                                        np.mean(t_deltas) * 1e3, np.mean(func_ret) * 1e3))


if __name__ == "__main__":
    main()
