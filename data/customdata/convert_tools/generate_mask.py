import cv2
import numpy as np
import os  
import argparse
from PIL import Image  # PIL比opencv保存稳定

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", help="由labelme转换的voc格式数据中npy文件所在路径")
    parser.add_argument("--output-dir", help="输出mask文件夹")
    args = parser.parse_args()
    input_dir = args.input_dir  # "convert_tools/example/segvoc/SegmentationClass"
    output_dir = args.output_dir  # "convert_tools/example/masklabels"

    npys = sorted(os.listdir(input_dir))
    for filename in npys:
        if filename.endswith(".npy"):
            inputpath = os.path.join(input_dir, filename)
            mask = np.load(inputpath)
            assert mask.max() <= 255  # 最大255（最多0到254类，255为忽略类）
            mask[mask<0] = 255  # 标注为负数的忽略，变为255
            filename = filename.replace(".npy", ".png")
            outpath = os.path.join(output_dir, filename)
            mask = Image.fromarray(mask.astype(np.uint8))
            mask.save(outpath)
    print(f"Successfully convert to { output_dir } !")