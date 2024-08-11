#!/bin/bash
NAME="mag_131415"

TAR_FILE="./tar/${NAME}.tar.gz"

tar --skip-old-files \
    -xzvf "$TAR_FILE" \
    -C ./try/data_tar/hmi/magnet_pt > ./log/${NAME}.log 2>&1


# tar -xzvf ../tar/mag_131415.tar.gz -C ../data_tar/hmi/magnet_pt