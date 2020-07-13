

def get_templ(img_size):
    model_parallel_plan_templ = {
        'rank-templ': {
            'down-conv-1': {
                'parallel_inputs': False,
                'input_chunk': None,
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size,
                    'padding': 3,
                    'kernel_size': 7,
                    'stride': 1
                }
            },
            'down-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (64-1)*2+3, 0, (64-1)*2 + 3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 2
                }
            },
            'down-conv-3': {
                'parallel_inputs': False,
                'input_chunk': [0, (32-1)*2+3, 0, (32-1)*2+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//2,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 2
                }
            },
            'down-resblock-1-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'down-resblock-1-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'down-resblock-2-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'down-resblock-2-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'down-resblock-3-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'down-resblock-3-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'down-resblock-4-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'down-resblock-4-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'adaRes-1-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'adaRes-1-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'adaRes-2-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'adaRes-2-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'adaRes-3-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'adaRes-3-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'adaRes-4-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'adaRes-4-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (32 - 1)+3, 0, (32-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//4,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'up-conv-1': {
                'parallel_inputs': False,
                'input_chunk': [0, (64-1)+3, 0, (64-1)+3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size//2,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'up-conv-2': {
                'parallel_inputs': False,
                'input_chunk': [0, (128 -1) + 3, 0, (128 - 1) + 3],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size,
                    'padding': 1,
                    'kernel_size': 3,
                    'stride': 1
                }
            },
            'up-conv-last': {
                'parallel_inputs': False,
                'input_chunk': [0, (128-1) + 7, 0, (128 -1)+7],
                'gather_outputs': True,
                'conv_info': {
                    'input_size': img_size,
                    'padding': 3,
                    'kernel_size': 7,
                    'stride': 1
                }
            }
        }
        
    }
    return model_parallel_plan_templ


def compute_data_block(input_size, rank, kernel_size, padding, stride):
    """ because current impl always use padding before convolution
    so the padding is included in the input_size
    """
    input_size += 2 * padding
    output_size = ((input_size - kernel_size) / stride) + 1
    # assume world size 4
    if rank == 0:
        return [0, int((output_size // 2 - 1) * stride + kernel_size), 
                0, int((output_size // 2 - 1) * stride + kernel_size)]
    elif rank == 1:
        return [0, int((output_size // 2 - 1) * stride + kernel_size), 
                int(input_size - ((output_size // 2 - 1) * stride + kernel_size)),
                int(input_size)]
    elif rank == 2:
        return [int(input_size - ((output_size // 2 - 1) * stride + kernel_size)),
                int(input_size),
                0, int((output_size // 2 - 1) * stride + kernel_size)]
    elif rank == 3:
        return [int(input_size - ((output_size // 2 - 1) * stride + kernel_size)),
                int(input_size),
                int(input_size - ((output_size // 2 - 1) * stride + kernel_size)),
                int(input_size)]
    else:
        return None


def get_conf(rank, img_size):
    temp_conf = get_templ(img_size)['rank-templ']
    for key in temp_conf:
        idxs = compute_data_block(rank=rank, **temp_conf[key]['conv_info'])
        temp_conf[key]['input_chunk'] = idxs
    return temp_conf