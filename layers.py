import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parameter import Parameter
import math
import numpy as np

cuda_events = []
conv_events = []

def average_event_time(n_batches):
    rank = dist.get_rank()
    ts = []
    for s, e in cuda_events:
        e.synchronize()
        elapsed_time_ms = s.elapsed_time(e)
        ts += [elapsed_time_ms]
    
    print('at rank', rank, 'avg allgather time: ', np.mean(ts[-200:]), 'ms',
            'std:', np.std(ts[-200:]), 'total allgater time per batch', sum(ts)/n_batches, 'ms')
    
    conv_ts = []
    for s, e in conv_events:
        e.synchronize()
        elapsed_time_ms = s.elapsed_time(e)
        conv_ts += [elapsed_time_ms]
    print('at rank', rank, 'conv cost avg (ms)', np.mean(conv_ts[-200:]))


class AttributeParallelConv2d(nn.Module):
    """parallel in width height of the output"""

    def __init__(self, key, mp_plan, in_channel, out_channel, kernel_size=7, stride=1, padding=0, bias=False):
        super(AttributeParallelConv2d, self).__init__()

        self.name = key
        self.parallel_inputs = mp_plan[key]['parallel_inputs']
        self.input_chunk_info = mp_plan[key]['input_chunk']
        self.gather_outputs = mp_plan[key]['gather_outputs']

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def _split_input(self, x, idxs):
        return x[:, :, idxs[0]:idxs[1], idxs[2]:idxs[3]]

    def _gather_output(self, x):
        """ TODO: might affect the gradients """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # pylint: disable=no-member
        tensor_list = [torch.empty_like(x) for _ in range(world_size)]

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        dist.all_gather(tensor_list, x)
        # torch.cuda.synchronize()
        end_event.record()
        cuda_events.append([start_event, end_event])
        tensor_list[rank] = x
        return self._assemble_outputs(tensor_list)

    def _assemble_outputs(self, gathered):
        shape = gathered[0].shape

        outputs = torch.stack(gathered, dim=2)  # pylint: disable=no-member
        world_size = dist.get_world_size()  # FIXME: suppose only work for 4 now
        s = int(math.sqrt(world_size))
        outputs = outputs.reshape(shape[0], -1, s, s, shape[-2], shape[-1])
        outputs = outputs.transpose(-2, -3).reshape(shape[0], shape[1], -1, s * shape[-1])
        return outputs

    def forward(self, input):
        """"""
        if not self.parallel_inputs:
            input = self._split_input(input, self.input_chunk_info)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        x = self.conv(input)
        end_event.record()
        conv_events.append([start_event, end_event])

        if self.gather_outputs:
            x = self._gather_output(x)
        return x


class MPResnetBlock(nn.Module):
    def __init__(self, module_key, mp_plan, dim, use_bias):
        super(MPResnetBlock, self).__init__()

        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                    AttributeParallelConv2d(module_key+'-conv-1', mp_plan, dim, dim, kernel_size=3, stride=1,
                                padding=0, bias=use_bias),
                    nn.InstanceNorm2d(dim),
                    nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       AttributeParallelConv2d(module_key+'-conv-2', mp_plan, dim, dim, kernel_size=3, stride=1,
                                 padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MPResnetAdaILNBlock(nn.Module):
    def __init__(self, key, mp_plan, dim, use_bias):
        super(MPResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)

        self.conv1 = AttributeParallelConv2d(key+'-conv-1', mp_plan, dim, dim, kernel_size=3,
                               stride=1, padding=0, bias=use_bias)
        self.norm1 = MPadaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = AttributeParallelConv2d(key+'-conv-2', mp_plan, dim, dim, kernel_size=3,
                               stride=1, padding=0, bias=use_bias)
        self.norm2 = MPadaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class MPadaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(MPadaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        # pylint: disable=no-member
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(
            input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(
            input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
            1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + \
            beta.unsqueeze(2).unsqueeze(3)

        return out


class MPILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(MPILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        # pylint: disable=no-member
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(
            input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(
            input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
            1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * \
            self.gamma.expand(
                input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class ModelParallelResnetGenerator(nn.Module):
    def __init__(self, mp_plan, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False, model_parallel=False, random_seed=123):
        assert(n_blocks >= 0)
        super(ModelParallelResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light
        self.cuda_device = None
        self.model_parallel = model_parallel
        torch.manual_seed(random_seed)

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      AttributeParallelConv2d("down-conv-1", mp_plan, input_nc, ngf, kernel_size=7,
                                stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          AttributeParallelConv2d("down-conv-" + str(i+2), mp_plan, ngf * mult, ngf * mult * 2,
                                    kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [MPResnetBlock("down-resblock-"+ str(i+1), mp_plan, ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1),
                    MPResnetAdaILNBlock('adaRes-'+str(i+1), mp_plan, ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         AttributeParallelConv2d('up-conv-'+str(i+1), mp_plan, ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=1, padding=0, bias=False),
                         MPILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     AttributeParallelConv2d('up-conv-last', mp_plan, ngf, output_nc, kernel_size=7,
                               stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input):
        if self.model_parallel and input.device != self.cuda_device:
            input = input.to(self.cuda_device)
        x = self.DownBlock(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        # pylint: disable=no-member
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap
