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


def check_weights_allclose(mp_model, orig_model):
    # pylint: disable=no-member
    # for downblocks
    for i in range(len(mp_model.DownBlock)):
        if isinstance(mp_model.DownBlock[i], AttributeParallelConv2d):
            w1 = mp_model.DownBlock[i].conv.weight
            w2 = orig_model.DownBlock[i].weight
            print('allclose', torch.allclose(w1, w2))

    for i in range(len(mp_model.FC)):
        if isinstance(mp_model.FC[i], torch.nn.Linear):
            w1 = mp_model.FC[i].weight 
            w2 = orig_model.FC[i].weight
            print('allclose', torch.allclose(w1, w2))
    
    print('allclose beta', torch.allclose(mp_model.beta.weight, orig_model.beta.weight))
    print('allclose gamma', torch.allclose(mp_model.gamma.weight, orig_model.gamma.weight))
    print('allclose gap_fc',  torch.allclose(mp_model.gap_fc.weight, orig_model.gap_fc.weight))
    print('allclose gmp_fc', torch.allclose(mp_model.gmp_fc.weight, orig_model.gmp_fc.weight))

    # up blocks
    print('upblock1_1', 
            torch.allclose(mp_model.UpBlock1_1.conv1.conv.weight, orig_model.UpBlock1_1.conv1.weight),
            torch.allclose(mp_model.UpBlock1_1.conv2.conv.weight, orig_model.UpBlock1_1.conv2.weight))
    print('upblock1_2', 
            torch.allclose(mp_model.UpBlock1_2.conv1.conv.weight, orig_model.UpBlock1_2.conv1.weight),
            torch.allclose(mp_model.UpBlock1_2.conv2.conv.weight, orig_model.UpBlock1_2.conv2.weight))
    print('upblock1_3', 
            torch.allclose(mp_model.UpBlock1_3.conv1.conv.weight, orig_model.UpBlock1_3.conv1.weight),
            torch.allclose(mp_model.UpBlock1_3.conv2.conv.weight, orig_model.UpBlock1_3.conv2.weight))
    print('upblock1_4', 
            torch.allclose(mp_model.UpBlock1_4.conv1.conv.weight, orig_model.UpBlock1_4.conv1.weight),
            torch.allclose(mp_model.UpBlock1_4.conv2.conv.weight, orig_model.UpBlock1_4.conv2.weight))
    
    for i in range(len(mp_model.UpBlock2)):
        if isinstance(mp_model.UpBlock2[i], AttributeParallelConv2d):
            w1 = mp_model.UpBlock2[i].conv.weight
            w2 = orig_model.UpBlock2[i].weight 
            print('Upblock2', i, torch.allclose(w1,w2))


def save_fwd(module, input, output):
    rank = dist.get_rank()
    with open("./rank-"+str(rank)+"-conv1-output.pkl", 'wb') as ofile:
        pickle.dump(output, ofile)

def get_hook(saveto):

    def _hook(module, input, output):
        saveto.append([input[0].clone().cpu(), output.clone().cpu()])

    return _hook

def save_mp_conv_outputs(module, saveto):
    childs = list(module.children())
    if isinstance(module, AttributeParallelConv2d):
        module.register_forward_hook(get_hook(saveto))
    
    if len(childs) == 0:
        return
    else:
        for c in childs:
            save_mp_conv_outputs(c, saveto)

def save_orig_conv_outputs(module, saveto):
    childs = list(module.children())
    if isinstance(module, torch.nn.Conv2d):
        module.register_forward_hook(get_hook(saveto))
    
    if len(childs) == 0:
        return
    else:
        for c in childs:
            save_orig_conv_outputs(c, saveto)

def main():
    """"""
    # pylint: disable=no-member
    args = get_args()
    
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
    mp_gen = ModelParallelResnetGenerator(mp_plan, 3, 3, args.ngf, 4, args.img_size, random_seed=args.random_seed)
    orig_gen = ResnetGenerator(3, 3, args.ngf, 4, args.img_size, random_seed=args.random_seed)

    check_weights_allclose(mp_gen, orig_gen)
    mp_gen = mp_gen.cuda()
    orig_gen = orig_gen.cuda()
    orig_conv_outputs = []
    mp_conv_outputs = []
    save_orig_conv_outputs(orig_gen, orig_conv_outputs)
    save_mp_conv_outputs(mp_gen, mp_conv_outputs)

    torch.manual_seed(args.random_seed)
    rand_img = torch.rand((bs, 3, s, s)) 
    can_output = orig_gen(rand_img.cuda())
    output_img = mp_gen(rand_img.cuda())
    # check intermediate outputs are close
    del orig_conv_outputs[11] # remove the conv1x1 layer 
    for i in range(len(orig_conv_outputs)):
        input_match = torch.allclose(orig_conv_outputs[i][0], mp_conv_outputs[i][0], atol=1e-05)
        output_match = torch.allclose(orig_conv_outputs[i][1], mp_conv_outputs[i][1], atol=1e-05)
        print('intermediate results are close', input_match, output_match)

    print('final outputs 0', torch.allclose(can_output[0].cpu(), output_img[0].cpu(), atol=1e-5))
    print('final outputs 1',torch.allclose(can_output[1].cpu(), output_img[1].cpu(), atol=1e-5))
    print('final outputs 2',torch.allclose(can_output[2].cpu(), output_img[2].cpu(), atol=1e-5))


if __name__ == "__main__":
    main()
