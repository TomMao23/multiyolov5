#!/bin/bash
echo "Now begin to search json file..."
for file in ./*
do
    if [ "${file##*.}"x = "json"x ]
    then
    filename=`basename $file`
    temp_filename=`basename $file  .json`
    suf=_json
    new_filename=${temp_filename}${suf}
#    echo $new_filename
    cmd="labelme_json_to_dataset ${filename} -o ${new_filename}"
    eval $cmd
    fi
#    printf "no!\n "
done

