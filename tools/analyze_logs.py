# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)

    for log_dict in log_dicts:
        plt.figure(0)
        plt.plot(range(len(log_dict['mIoU'])), log_dict['mIoU'])
        plt.plot(range(len(log_dict['mAcc'])), log_dict['mAcc'])
        plt.legend(['mIoU', 'mAcc'])
        plt.xlabel('test iter')
        plt.ylabel('mIoU/mAcc')
        plt.savefig('./mIoU_mAcc.png')

        plt.figure(1)
        lengend = list()
        for k, v in log_dict.items():
            if k.split('.')[0] == 'IoU':
                plt.plot(range(len(log_dict[k])), v)
                lengend.append(k.split('.')[1])
        plt.legend(lengend)
        plt.xlabel('test iter')
        plt.ylabel('mIoU')
        plt.savefig('./mIoU.png')

        plt.figure(2)
        lengend = list()
        for k, v in log_dict.items():
            if k.split('.')[0] == 'Acc':
                plt.plot(range(len(log_dict[k])), v)
                lengend.append(k.split('.')[1])
        plt.legend(lengend)
        plt.xlabel('test iter')
        plt.ylabel('Acc')
        plt.savefig('./mAcc.png')

        plt.figure(3)
        plt.plot(range(100, 100+50*200, 50), log_dict['decode.acc_seg'][0:200])
        plt.plot(range(100, 100+50*200, 50), log_dict['mix.decode.acc_seg'][0:200])
        plt.legend(['acc_seg', 'mixture_acc_seg'])
        plt.xlabel('iter')
        plt.ylabel('Acc')
        plt.savefig('./acc_seg.png')

        plt.figure(4)
        plt.plot(range(100, 100+50*200, 50), log_dict['decode.loss_seg'][0:200])
        plt.plot(range(100, 100+50*200, 50), log_dict['mix.decode.loss_seg'][0:200])
        plt.legend(['loss_seg', 'mixture_loss_seg'])
        plt.xlabel('iter')
        plt.ylabel('Loss')
        plt.savefig('./loss_seg.png')
        print('save figure to ./mIoU_mAcc.png, ./mIoU.png, ./mAcc.png, ./acc_seg.png, ./loss_seg.png')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='the metric that you want to plot')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default='./result')
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                for k, v in log.items():
                    if k in log_dict:
                        log_dict[k].append(v)
                    else:
                        log_dict[k] = list()
    return log_dicts


def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)
    plot_curve(log_dicts, args)


if __name__ == '__main__':
    main()
