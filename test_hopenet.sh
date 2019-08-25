#!/bin/bash

python -W ignore test_hopenet.py --gpu 0 --num_bins 66 --Model_num 0 --data_dir "/usr/stud/sharmaki/Projects/Advertima/datasets/AFLW2000-3D/AFLW2000/" --filename_list '/usr/stud/sharmaki/Projects/Advertima/deep-head-pose-master/deep-head-pose-master/dataset_filename/' --snapshot 'Debug_result/output/snapshots/_epoch_1.pkl' --batch_size '16' --save_viz 'False' --dataset 'AFLW2000' --output_dir 'Debug_result'
