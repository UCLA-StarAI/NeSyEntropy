#!/bin/sh -x
################################################################ 3
# semantic_weight: 0.01
#python -u -W ignore train.py --id 10_3_1 --gpu_id 0 --batch_size 128 --semantic_weight 0.01 --entreg_weight 0.01 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/10_3_1.txt
#python -u -W ignore train.py --id 10_3_2 --gpu_id 0 --batch_size 128 --semantic_weight 0.01 --entreg_weight 0.05  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/10_3_2.txt
#python -u -W ignore train.py --id 10_3_3 --gpu_id 0 --batch_size 128 --semantic_weight 0.01 --entreg_weight 0.1  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/10_3_3.txt
#python -u -W ignore train.py --id 10_3_4 --gpu_id 0 --batch_size 128 --semantic_weight 0.01 --entreg_weight 0.5  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/10_3_4.txt
#
#
## semantic_weight: 0.05
#python -u -W ignore train.py --id 11_3_1 --gpu_id 0 --batch_size 128 --semantic_weight 0.05 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/11_3_1.txt
#python -u -W ignore train.py --id 11_3_2 --gpu_id 0 --batch_size 128 --semantic_weight 0.05 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/11_3_2.txt
#python -u -W ignore train.py --id 11_3_3 --gpu_id 0 --batch_size 128 --semantic_weight 0.05 --entreg_weight 0.1  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/11_3_3.txt
#python -u -W ignore train.py --id 11_3_4 --gpu_id 0 --batch_size 128 --semantic_weight 0.05 --entreg_weight 0.5  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/11_3_4.txt
#
#
### semantic_weight: 0.1
#python -u -W ignore train.py --id 12_3_1 --gpu_id 0 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/12_3_1.txt
python -u -W ignore train.py --id 12_3_1_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 2345 > logs_sle/12_3_1_1.txt
python -u -W ignore train.py --id 12_3_1_2 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 3456 > logs_sle/12_3_1_2.txt
#python -u -W ignore train.py --id 12_3_2 --gpu_id 0 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/12_3_2.txt
#python -u -W ignore train.py --id 12_3_3 --gpu_id 0 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.1 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/12_3_3.txt
#python -u -W ignore train.py --id 12_3_4 --gpu_id 0 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/12_3_4.txt
#
#
## semantic_weight: 0.01
#python -u -W ignore train.py --id 13_3_1 --gpu_id 0 --batch_size 256 --semantic_weight 0.01 --entreg_weight 0.01 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/13_3_1.txt
#python -u -W ignore train.py --id 13_3_2 --gpu_id 0 --batch_size 256 --semantic_weight 0.01 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/13_3_2.txt
#python -u -W ignore train.py --id 13_3_3 --gpu_id 0 --batch_size 256 --semantic_weight 0.01 --entreg_weight 0.1 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/13_3_3.txt
#python -u -W ignore train.py --id 13_3_4 --gpu_id 0 --batch_size 256 --semantic_weight 0.01 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/13_3_4.txt
#
#
## semantic_weight: 0.05
#python -u -W ignore train.py --id 14_3_1 --gpu_id 0 --batch_size 256 --semantic_weight 0.05 --entreg_weight 0.01 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/14_3_1.txt
#python -u -W ignore train.py --id 14_3_2 --gpu_id 0 --batch_size 256 --semantic_weight 0.05 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/14_3_2.txt
#python -u -W ignore train.py --id 14_3_3 --gpu_id 0 --batch_size 256 --semantic_weight 0.05 --entreg_weight 0.1 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/14_3_3.txt
#python -u -W ignore train.py --id 14_3_4 --gpu_id 0 --batch_size 256 --semantic_weight 0.05 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/14_3_4.txt
#
#
## semantic_weight: 0.1
#python -u -W ignore train.py --id 15_3_1 --gpu_id 0 --batch_size 256 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/15_3_1.txt
#python -u -W ignore train.py --id 15_3_2 --gpu_id 0 --batch_size 256 --semantic_weight 0.1 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/15_3_2.txt
#python -u -W ignore train.py --id 15_3_3 --gpu_id 0 --batch_size 256 --semantic_weight 0.1 --entreg_weight 0.1 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/15_3_3.txt
#python -u -W ignore train.py --id 15_3_4 --gpu_id 0 --batch_size 256 --semantic_weight 0.1 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/15_3_4.txt
#
## semantic_weight: 0.5
#python -u -W ignore train.py --id 16_3_1 --gpu_id 0 --batch_size 128 --semantic_weight 0.5 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/16_3_1.txt
#python -u -W ignore train.py --id 16_3_2 --gpu_id 0 --batch_size 128 --semantic_weight 0.5 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/16_3_2.txt
#python -u -W ignore train.py --id 16_3_3 --gpu_id 0 --batch_size 128 --semantic_weight 0.5 --entreg_weight 0.1 --save_dir './saved_models/sle'  --num_samples 3  --seed 1234 > logs_sle/16_3_3.txt
#python -u -W ignore train.py --id 16_3_4 --gpu_id 0 --batch_size 128 --semantic_weight 0.5 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/16_3_4.txt
#
#python -u -W ignore train.py --id 17_3_1 --gpu_id 0 --batch_size 256 --semantic_weight 0.5 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/17_3_1.txt
#python -u -W ignore train.py --id 17_3_2 --gpu_id 0 --batch_size 256 --semantic_weight 0.5 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/17_3_2.txt
#python -u -W ignore train.py --id 17_3_3 --gpu_id 0 --batch_size 256 --semantic_weight 0.5 --entreg_weight 0.1 --save_dir './saved_models/sle'  --num_samples 3  --seed 1234 > logs_sle/17_3_3.txt
#python -u -W ignore train.py --id 17_3_4 --gpu_id 0 --batch_size 256 --semantic_weight 0.5 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/17_3_4.txt
#
### semantic_weight: 0.5
#python -u -W ignore train.py --id 18_3_1 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entreg_weight 0.01 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/18_3_1.txt
#python -u -W ignore train.py --id 18_3_2 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/18_3_2.txt
#python -u -W ignore train.py --id 18_3_3 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entreg_weight 0.1  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/18_3_3.txt
#python -u -W ignore train.py --id 18_3_4 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entreg_weight 0.5  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/18_3_4.txt
#
#python -u -W ignore train.py --id 19_3_1 --gpu_id 0 --batch_size 256 --semantic_weight 1.0 --entreg_weight 0.01 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/19_3_1.txt
#python -u -W ignore train.py --id 19_3_2 --gpu_id 0 --batch_size 256 --semantic_weight 1.0 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/19_3_2.txt
#python -u -W ignore train.py --id 19_3_3 --gpu_id 0 --batch_size 256 --semantic_weight 1.0 --entreg_weight 0.1  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/19_3_3.txt
#python -u -W ignore train.py --id 19_3_4 --gpu_id 0 --batch_size 256 --semantic_weight 1.0 --entreg_weight 0.5  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/19_3_4.txt
