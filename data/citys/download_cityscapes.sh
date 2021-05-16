#!/bin/bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=1652573@tongji.edu.cn&password=58545256mzw&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
