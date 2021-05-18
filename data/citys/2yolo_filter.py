import json
import os
from pathlib import Path
import re
from tqdm import tqdm
import shutil


def convert_annotation(image_id, paths):
    global label_map

    def find_box(points):  # 该函数用来找出xmin, xmax, ymin ,ymax 即bbox包围框
        _x, _y = [float(pot[0]) for pot in points], [float(pot[1]) for pot in points]
        return min(_x), max(_x), min(_y), max(_y)

    def convert(size, bbox):  # 转为中心坐标
        # size: (原图宽， 原图长)
        center_x, center_y = (bbox[0] + bbox[1]) / 2.0 - 1, (bbox[2] + bbox[3]) / 2.0 - 1
        center_w, center_h = bbox[1] - bbox[0], bbox[3] - bbox[2]
        return center_x / size[0], center_y / size[1], center_w / size[0], center_h / size[1]

    final_label_path, final_output_path = paths
    label_json_url = final_label_path / f'{image_id}_gtFine_polygons.json'
    # 输出到 ：final_output_path / f'{image_id}_leftImg8bit.txt'

    load_dict = json.load(open(label_json_url, 'r'))  # 图像的实例
    output_cache = []
    for obj in load_dict['objects']:  # load_dict['objects'] -> 目标的几何框体
        obj_label = obj['label']  # 目标的类型
        if obj_label not in list(label_map.keys()):
            continue

        if obj_label not in label_map.keys():  # 记录目标类型转为int值
            label_map[obj_label] = len(label_map.keys())  # 标签从0开始

        x, y, w, h = convert((load_dict['imgWidth'], load_dict['imgHeight']), find_box(obj['polygon']))  # 归一化为中心点

        # yolo 标准格式：img.jpg -> img.txt
        # 内容的类别 归一化后的中心点x坐标 归一化后的中心点y坐标 归一化后的目标框宽度w 归一化后的目标况高度h
        output_cache.append(f'{label_map[obj_label]} {x} {y} {w} {h}\n')

    with open(final_output_path / f'{image_id}_leftImg8bit.txt', 'w') as label_f:  # 写出标签文件
        label_f.writelines(output_cache)


def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)


if __name__ == '__main__':
    root_dir = Path(__file__).parent
    image_dir = root_dir / 'leftImg8bit'
    label_dir = root_dir / 'gtFine'
    image_output_root_dir = root_dir /'images'
    label_output_root_dir = root_dir / 'labels'

    label_map = { "car": 0,
                  "traffic sign": 1,
                  "rider": 2,
                  "person": 3,
                  "bicycle": 4,
                  "traffic light": 5,
                  "bus": 6,
                  "motorcycle": 7,
                  "truck": 8,
                  "train": 9,
   }  # 存放所有的类别标签  eg. {'car': 0, 'person': 1}
    for _t_ in tqdm(os.listdir(image_dir)):  # _t_ as ['train', 'test' 'val;]
        type_files = []  # 存放一类的所有文件，如训练集所有文件
        mkdir(image_output_root_dir / _t_), mkdir(label_output_root_dir / _t_)
        for cities_name in os.listdir(image_dir / _t_):
            _final_img_path = image_dir / _t_ / cities_name  # root_dir / leftImg8bit / test / berlin
            _final_label_path = label_dir / _t_ / cities_name  # root_dir / gtfine / test / berlin

            # berlin_000000_000019_leftImg8bit.png -> berlin_000000_000019_gtFine_polygons.json
            image_ids = list(map(lambda s: re.sub(r'_leftImg8bit\.png', '', s), os.listdir(_final_img_path)))
            # print(names[:0])  -> berlin_000000_000019

            for img_id in image_ids:
                convert_annotation(img_id, [_final_label_path, label_output_root_dir / _t_])  # 转化标签
                img_file = f'{img_id}_leftImg8bit.png'
                shutil.copy(_final_img_path / img_file, f'images/{_t_}/{img_file}')  # 复制移动图片
                type_files.append(f'images/{_t_}/{img_id}_leftImg8bit.png\n')

        with open(root_dir / f'yolo_{_t_}.txt', 'w') as f:  # 记录训练样本等的具体内容
            f.writelines(type_files)

    with open(label_output_root_dir / 'classes.txt', 'w') as f:  # 写出类别对应
        for k, v in label_map.items():
            f.write(f'{k}\n')
    print([k for k in label_map.keys()], len([k for k in label_map.keys()]))

