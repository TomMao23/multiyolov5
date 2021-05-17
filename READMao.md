!cd yolov5/data/citys 先下载cityscapes数据集  
!python 2yolo_filter.py  
!mkdir detdata  
!mv labels detdata  
!mv images detdata  
!cd ../../
python train.py --noautoanchor --data cityscapes_det.yaml --cfg yolov5s_city_seg.yaml --batch-size 12 --epochs 80 --weights weights/yolov5s.pt --workers 4

