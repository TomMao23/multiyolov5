# Train Custom Dataset
检测数据集没有特别要求，同yolov5本身格式即可，分割数据集以BDD100k这种不存在id到trianid转换，ignore类别标为255的为例实现了CustomData分割类  
## 数据和模型配置
参考cityscapes实验  
数据放在yolo的data文件夹下某个子文件夹中，如citys和customdata  
- 检测数据按照yolov5组织格式，如customdata的detdata文件夹下images和labels两个文件夹分别放图像和yolo格式标签，两个文件夹下各有train和val两个子文件夹用于存放训练和验证数据、标签。  
- 分割图像、标签放在segimages和seglabels两个文件夹。这两个文件夹下各有trian，val两个子文件夹，子文件夹里面放图像（jpg、png）或标签(必须png，像素值就是类别，忽略类值为255，参考bdd100k的格式)。  
- 仿照（复制修改）data/cityscapes_det.yaml的配置数据，train和val写图片数据的train和val文件夹的路径（yolo自己会对应找labels），nc为检测类别数，segtrain和segval写分割数据segimages和seglabels父文件夹的路径，names为各检测类别名字。  
- 仿照（复制修改）models/yolov5s_city_seg.yaml的配置模型，nc为检测类别数，n_segcls为分割类别数。depth_multiple，width_multiple控制深度和宽度，是yolov5s，m，l，x唯一的区别。  
- 对应主文件夹下README的训练和测试部分命令，修改为你的数据集、数据集配置文件、模型配置文件路径即可训练你的数据  
## 调用示例
回到主目录下
```bash
$ python train_custom.py --data 你的数据.yaml --cfg 你的模型.yaml --batch-size 18 --epochs 100 --weights ./yolov5s.pt --workers 8 --label-smoothing 0.1 --img-size 832 --device 0
```
注意：训自己的数据一般就不要关autoanchors了，--img-size既是检测的训练尺寸，也是分割的crop-size和base-size(分割crop前resize的大小)，会不均匀的随机按长边resize到base-size的一定范围（默认0.75到1.5倍）再crop出(img-size, img-size)一块用于训练分割（要更改请参考train_custom.py和SegmentationDataset.py。train.py与train_custom.py略不同，针对cityscapes和bdd100k写死了base-size=1024没有暴露出来，crop是矩形而非方形，长边由img-size指定，建议832或1024）   
## 训练自己数据前请认真考虑base-size和crop-size的设置是否合理(建议用IDE先打断点开debug调试Segmentation.py可视化crop效果)，这将极大影响训练效果
## labelme标注数据转换
当前customdata文件夹里的示例数据就是convert_tools/example的转换结果,转换后  
转换工具在convert_tools中  
转换例子在example中  
- 标注后把图片和同名json文件放在一个文件夹中，如example的det和seg  
- 先使用两个labelme脚本（来自labelme自己的检测、分割example，因同名重命名过）转换为voc格式。仿照example定义好自己的detlabels.txt和seglabels.txt类别    
  ```bash
  $ cd convert_tools
  $ python labelme2detvoc.py example/det example/detvoc --labels example/detlabels.txt
  $ python labelme2segvoc.py example/seg example/segvoc --labels example/seglabels.txt
  ```
  即可在example中看到detvoc和segvoc这两个voc格式的检测和分割数据  
  tips：若转换失败要删除输出文件夹重新运行脚本  
- 使用convert2Yolo工具把检测转为yolo格式
  ```bash
  $ python convert2Yolo/example.py --datasets VOC --img_path ./example/detvoc/JPEGImages --label ./example/detvoc/Annotations --convert_output_path ./example/yolodetlabels --img_type ".jpg" --manifest_path ./example --cls_list_file ./example/names.txt 
  ```
  即可在./example/yolodetlabels找到转换完成的yolo标签，这些就是要放入detdata的labels/train或val内的检测标签，把对应的图像也放在detdata的images/train或val里。根据以上和主目录的REAME说明修改训练和测试部分命令，修改为你的数据集(customdata/detdata)、数据集配置文件、模型配置文件路径即完成了检测部分的数据准备
- 使用generate_mask.py把voc格式的分割npy文件转为png格式的mask  
  ```bash
   $ python generate_mask.py --input-dir ./example/segvoc/SegmentationClass --output-dir ./example/masklabels

  ```
  即可在./example/masklabels找到图片同名的.png标签（如example中1为crack，0为background，数值很小所以看起来几乎是黑的）把这些mask标签放在seglabels/train或val中，对应图像文件放在segimages/train或val中。根据以上和主目录的REAME说明修改训练和测试部分命令，修改为你的数据集(customdata)、数据集配置文件、模型配置文件路径即完成了分割部分的数据准备   
  对于此例子，可用以下调用
```bash
$ python train_custom.py --data custom.yaml --cfg yolov5s_custom_seg.yaml --weights ./yolov5s.pt --workers 4 --label-smoothing 0.1 --img-size 832 --batch-size 4 --epochs 1000 --rect  
```
这个例子效果并不好，特别是检测，仅用于演示labelme格式生成可训练数据