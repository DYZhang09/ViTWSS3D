import os  
import numpy as np
from shutil import copyfile

cls_list = ['Car', 'Pedestrian', 'Cyclist']

label_ratio = 10

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

def merge_pseudo_label(file_name, cnt_equal, cnt_not_equal):
    file_path_gt = source_root + file_name + '.txt'
    with open(file_path_gt, 'r') as f:
        lines_gt = f.readlines()

    new_lines_gt = []
    for idx in range(len(lines_gt)):
        if lines_gt[idx].split(' ')[0] in cls_list:
            new_lines_gt.append(lines_gt[idx])
    #lines_gt = new_lines_gt
    #print(lines_gt)

    file_path_pseudo = pseudo_root + file_name + '.txt'
    with open(file_path_pseudo, 'r') as f:
        lines_pseudo = f.readlines()

    save_gt = []
    flag_using_origianl = False

    for idx in range(len(lines_pseudo)):
        if lines_pseudo[idx].split(' ')[0] == new_lines_gt[idx].split(' ')[0]:
            #new_lines_gt[idx] = lines_pseudo[idx]
            temp =  new_lines_gt[idx].split(' ')
            pseudo_temp =  lines_pseudo[idx].split(' ')
            temp[3:] = pseudo_temp[3:-1]
            temp[-1] = temp[-1] + '\n'
            temp = ' '.join(temp)
            save_gt.append(temp)

        else:
            #a = lines_gt.sort(key=takeSecond)
            ''''directly using gt version'''
            # file_path = '/data/dkliang/projects/synchronous/baidu_a100_3090/YOLOS3d/data/kitti/training/label_pseudo/' + file_name + '.txt'
            # with open(file_path, 'w') as f:
            #     f.writelines(lines_gt)
            # print('not equal, using original gt', file_path)

            flag_using_origianl = True

            pseudo_temp = []
            for i in range(len(lines_pseudo)):
                temp = lines_pseudo[i].split(' ')
                temp[1] = '0.00'
                temp[2] = '0'
                temp = ' '.join(temp)
                pseudo_temp.append(temp)
            file_path = save_path + file_name + '.txt'
            with open(file_path, 'w') as f:
                f.writelines(pseudo_temp)
            print('not equal, set as zero for truncated and occluded', file_path,pseudo_temp)
            cnt_not_equal = cnt_not_equal + 1
            break


    if not flag_using_origianl:
        file_path = save_path + file_name + '.txt'
        with open(file_path, 'w') as f:
            f.writelines(save_gt)
        cnt_equal = cnt_equal + 1
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

    for file_name in path_name_full:
        file_name_change = file_name.split('\n')[0]

        if file_name_change in path_name_pseudo:
            #print(file_name_change)
            cnt_equal, cnt_not_equal = merge_pseudo_label(file_name_change,cnt_equal, cnt_not_equal)

        else:
            source_file = source_root  + path_list_full[path_name_full.index(file_name)].split('\n')[0] + '.txt'
            destination_file = save_path + path_list_full[path_name_full.index(file_name)].split('\n')[0] + '.txt'
            #print(source_file)
            copyfile(source_file, destination_file)
            cnt_direct_copy = cnt_direct_copy + 1

    print('cnt_direct_copy', cnt_direct_copy, 'cnt_equal', cnt_equal, 'cnt_not_equal', cnt_not_equal, 'all', cnt_direct_copy + cnt_equal + cnt_not_equal)

