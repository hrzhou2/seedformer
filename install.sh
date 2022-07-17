#!/bin/bash

cd ./codes/pointnet2_ops_lib
python setup.py install

cd ../Chamfer3D
python setup.py install
