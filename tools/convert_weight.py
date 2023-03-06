import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author="OpenSelfSup")
    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
            has_backbone = True
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)


def convert_wangqc_pretrian():
    """
    Convert facial recognition ckpt trained by WangQC to mmcv style
    head -> head.fc1
    bn -> head.bn
    """
    from collections import OrderedDict
    origin = torch.load('pretrain_recognition_wang_origin.ckpt', map_location='cpu')['net_state_dict']
    new_dict = OrderedDict()
    for k, v in origin.items():
        if k.startswith('head.'):
            new_dict[f'head.fc1.{k[5:]}'] = v
        elif k.startswith('bn.'):
            new_dict[f'head.bn.{k[3:]}'] = v
        else:
            new_dict[f'backbone.{k}'] = v
    print(new_dict.keys())
    torch.save(new_dict, 'pretrain_recognition_wang.ckpt')


if __name__ == '__main__':
    main()
    # convert_wangqc_pretrian()
