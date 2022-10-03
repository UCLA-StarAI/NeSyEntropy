#!/bin/sh -x
python -u -W ignore train.py --id 12_3_1 --gpu_id 0 --batch_size 128 --semantic_weight 0.1 --entropy_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/12_3_1.txt
python -u -W ignore train.py --id 12_3_1_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entropy_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 2345 > logs_sle/12_3_1_1.txt
python -u -W ignore train.py --id 12_3_1_2 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entropy_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 3456 > logs_sle/12_3_1_2.txt

python -u -W ignore train.py --id 15_5_2 --gpu_id 0 --batch_size 256 --semantic_weight 0.1 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 5 --seed 1234 > logs_sle/15_5_2.txt
python -u -W ignore train.py --id 15_5_2_1 --gpu_id 1 --batch_size 256 --semantic_weight 0.1 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 5 --seed 2345 > logs_sle/15_5_2_1.txt
python -u -W ignore train.py --id 15_5_2_2 --gpu_id 1 --batch_size 256 --semantic_weight 0.1 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 5 --seed 3456 > logs_sle/15_5_2_2.txt

python -u -W ignore train.py --id 18_10_2 --gpu_id 1 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 10 --seed 1234 > logs_sle/18_10_2.txt
python -u -W ignore train.py --id 18_10_2_1 --gpu_id 1 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 10 --seed 2345 > logs_sle/18_10_2_1.txt
python -u -W ignore train.py --id 18_10_2_2 --gpu_id 1 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 10 --seed 3456 > logs_sle/18_10_2_2.txt

python -u -W ignore train.py --id 16_15_2 --gpu_id 1 --batch_size 128 --semantic_weight 0.5 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 15 --seed 1234 > logs_sle/16_15_2.txt
python -u -W ignore train.py --id 16_15_2_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.5 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 15 --seed 2345 > logs_sle/16_15_2_1.txt
python -u -W ignore train.py --id 16_15_2_2 --gpu_id 1 --batch_size 128 --semantic_weight 0.5 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 15 --seed 3456 > logs_sle/16_15_2_2.txt

python -u -W ignore train.py --id 18_25_2 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 25  --seed 1234 > logs_sle/18_25_2.txt
python -u -W ignore train.py --id 18_25_2_1 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 25  --seed 2345 > logs_sle/18_25_2_1.txt
python -u -W ignore train.py --id 18_25_2_2 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.05 --save_dir './saved_models/sle'  --num_samples 25  --seed 3456 > logs_sle/18_25_2_2.txt

python -u -W ignore train.py --id 18_50_3 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.1 --save_dir './saved_models/sle'  --num_samples 50   --seed 1234 > logs_sle/18_50_3.txt
python -u -W ignore train.py --id 18_50_3_1 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.1 --save_dir './saved_models/sle'  --num_samples 50   --seed 2345 > logs_sle/18_50_3_1.txt
python -u -W ignore train.py --id 18_50_3_2 --gpu_id 0 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.1 --save_dir './saved_models/sle'  --num_samples 50   --seed 3456 > logs_sle/18_50_3_2.txt

python -u -W ignore train.py --id 18_75_1 --gpu_id 1 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.01  --save_dir './saved_models/sle'  --num_samples 75 --seed 1234 > logs_sle/18_75_1.txt
python -u -W ignore train.py --id 18_75_1_1 --gpu_id 1 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.01  --save_dir './saved_models/sle'  --num_samples 75 --seed 2345 > logs_sle/18_75_1_1.txt
python -u -W ignore train.py --id 18_75_1_2 --gpu_id 1 --batch_size 128 --semantic_weight 1.0 --entropy_weight 0.01  --save_dir './saved_models/sle'  --num_samples 75 --seed 3456 > logs_sle/18_75_1_2.txt
