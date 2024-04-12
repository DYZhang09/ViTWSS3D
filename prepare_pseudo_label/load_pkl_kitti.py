import mmcv



ann_file = '/data/dkliang/projects/synchronous/baidu_a100_3090/YOLOS3d/data/kitti/kitti_infos_train.pkl'
results_file = '/data/dkliang/projects/synchronous/baidu_a100_3090/YOLOS3d/data/kitti/kitti_infos_train_20_20.pkl'

def load_annotations( ann_file):
    """Load annotations from ann_file.

    Args:
        ann_file (str): Path of the annotation file.

    Returns:
        list[dict]: List of annotations sorted by timestamps.
    """
    data = mmcv.load(ann_file)

    return data


def load_results( results_file):
    """Load annotations from ann_file.

    Args:
        ann_file (str): Path of the annotation file.

    Returns:
        list[dict]: List of annotations sorted by timestamps.
    """
    data_infos = mmcv.load(results_file)
    # data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    # load_interval = 1
    # data_infos = data_infos[::load_interval]
    # metadata = data['metadata']
    # version = metadata['version']
    return data_infos




data_infos = load_annotations(ann_file)
results_infos = load_results(results_file)

for idx in range(len(results_infos)):
    data_info = data_infos[idx]['annos']
    pseudo_info = results_infos[idx]['annos']
    print(pseudo_info)

a = 1