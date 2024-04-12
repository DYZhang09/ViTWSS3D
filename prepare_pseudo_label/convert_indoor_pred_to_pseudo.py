import numpy as np
import argparse
import ipdb
import os
import os.path as osp


def merge_sunrgbd_gt_pseudo_txt(gt_data_root, pseudo_txt_root, load_gt_interval, use_v1=False):
    gt_trainval_root = osp.join(gt_data_root, 'sunrgbd_trainval')
    trainval_label_root = osp.join(gt_trainval_root, 'label' if not use_v1 else 'label_v1')

    # need to copy val files to pseudo_txt_root
    val_idx_file = osp.join(gt_trainval_root, 'val_data_idx.txt')
    val_idx_list = [int(line.rstrip()) for line in open(val_idx_file, 'r')]
    val_file_name = [f'{idx:06d}.txt' for idx in val_idx_list]

    for val_file in val_file_name:
        src_path = osp.join(trainval_label_root, val_file)
        dst_path = osp.join(pseudo_txt_root, val_file)
        os.system(f'cp {src_path} {dst_path}')
    print(f'Copy {len(val_file_name)} val files')

    # need to copy train files that used to train the annotator 
    train_idx_file = osp.join(gt_trainval_root, 'train_data_idx.txt')
    train_idx_list = [int(line.rstrip()) for line in open(train_idx_file, 'r')]
    train_cp_idx_list = train_idx_list[::load_gt_interval]
    train_file_name = [f'{idx:06d}.txt' for idx in train_cp_idx_list]

    for train_file in train_file_name:
        src_path = osp.join(trainval_label_root, train_file)
        dst_path = osp.join(pseudo_txt_root, train_file)
        os.system(f'cp {src_path} {dst_path}')

    print(f'Total {len(train_idx_list)} train files, '
          f'copy {len(train_file_name)}({len(train_file_name) * 100 / len(train_idx_list)}%).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_gt_interval', type=int, required=True)
    parser.add_argument('--gt_data_root', type=str, required=True)
    parser.add_argument('--pseudo_txt_root', type=str, required=True)
    parser.add_argument('--use_v1', type=bool, default=False)
    args = parser.parse_args()

    merge_sunrgbd_gt_pseudo_txt(
        args.gt_data_root,
        args.pseudo_txt_root,
        args.load_gt_interval,
        args.use_v1
    )
