import os  
import numpy as np
from shutil import copyfile

cls_list = ['Car', 'Pedestrian', 'Cyclist']

label_ratio = 20

save_path = f'../data/kitti/training/label_pseudo_{label_ratio}/'
pseudo_root = f'../results/kitti-3class/kitti_results_{label_ratio}/'

source_root = '../data/kitti/training/label_2/'
imageset_train_file_path = '../data/kitti/ImageSets/train.txt'


def _read_imageset_file(path, generate_label_data=False):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line for line in lines]

def takeSecond(elem):
    return float(elem.split(' ')[3])

def modify_gt(file_name):
    file_path_gt = source_root + file_name + '.txt'
    with open(file_path_gt, 'r') as f:
        lines_gt = f.readlines()

    new_lines_gt = []
    for idx in range(len(lines_gt)):
        new_lines_gt.append(lines_gt[idx])

    save_gt = []

    for idx in range(len(lines_gt)):

        temp =  new_lines_gt[idx].split(' ')

        if temp[0] == cls_list[0]:
            temp[8:11] = ['1.56', '1.6', '3.9']
        elif temp[0] == cls_list[1]:
            temp[8:11] = ['1.73', '0.8', '0.6']
        elif temp[0] == cls_list[2]:
            temp[8:11] = ['1.73', '0.8', '1.76']

        temp[-1] = temp[-1]
        temp = ' '.join(temp)
        save_gt.append(temp)

    file_path = save_path + file_name + '.txt'
    with open(file_path, 'w') as f:
        f.writelines(save_gt)
    print(save_gt)

    return cnt_equal, cnt_not_equal






if __name__ == '__main__':

    path_list_full = _read_imageset_file(imageset_train_file_path) # load training list
    path_list_pseudo = os.listdir(pseudo_root)  # load prediction list

    path_name_full = []
    path_name_pseudo = []
    for i in path_list_full:
        path_name_full.append(i)

    for i in path_list_pseudo:
        path_name_pseudo.append((i.split(".")[0]))  

    print('len_gt', len(path_name_full), 'len_pseudo', len(path_name_pseudo))

    cnt_direct_copy = 0
    cnt_equal = 0
    cnt_not_equal = 0

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    index = 0
    for index in range(len(path_name_full)):
        file_name = path_name_full[index]
        file_name_change = file_name.split('\n')[0]

        if index%label_ratio==0:
            #print(file_name_change)
            #cnt_equal, cnt_not_equal = merge_pseudo_label(file_name_change,cnt_equal, cnt_not_equal)
            source_file = source_root + path_list_full[path_name_full.index(file_name)].split('\n')[0] + '.txt'
            destination_file = save_path + path_list_full[path_name_full.index(file_name)].split('\n')[0] + '.txt'
            # print(source_file)
            copyfile(source_file, destination_file)
            cnt_direct_copy = cnt_direct_copy + 1
        else:
            modify_gt(file_name_change)
            #cnt_equal, cnt_not_equal = merge_pseudo_label(file_name_change, cnt_equal, cnt_not_equal)
            print("test")
    print('cnt_direct_copy', cnt_direct_copy, 'cnt_equal', cnt_equal, 'cnt_not_equal', cnt_not_equal, 'all', cnt_direct_copy + cnt_equal + cnt_not_equal)

