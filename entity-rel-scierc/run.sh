python -u -W ignore train.py --id 12_3_1 --gpu_id 0 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 1234 > logs_sle/12_3_1.txt
python -u -W ignore train.py --id 12_3_1_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 2345 > logs_sle/12_3_1_1.txt
python -u -W ignore train.py --id 12_3_1_2 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 3 --seed 3456 > logs_sle/12_3_1_2.txt

python -u -W ignore train.py --id 12_5_2 --gpu_id 0 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 5 --seed 1234 > logs_sle/12_5_2.txt
python -u -W ignore train.py --id 12_5_2_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 5 --seed 2345 > logs_sle/12_5_2_1.txt
python -u -W ignore train.py --id 12_5_2_2 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.05 --save_dir './saved_models/sle'  --num_samples 5 --seed 3456 > logs_sle/12_5_2_2.txt

python -u -W ignore train.py --id 14_15_4 --gpu_id 1 --batch_size 256 --semantic_weight 0.05 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 15 --seed 1234 > logs_sle/14_15_4.txt
python -u -W ignore train.py --id 14_15_4_1 --gpu_id 1 --batch_size 256 --semantic_weight 0.05 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 15 --seed 2345 > logs_sle/14_15_4_1.txt
python -u -W ignore train.py --id 14_15_4_2 --gpu_id 1 --batch_size 256 --semantic_weight 0.05 --entreg_weight 0.5 --save_dir './saved_models/sle'  --num_samples 15 --seed 3456 > logs_sle/14_15_4_2.txt

python -u -W ignore train.py --id 16_50_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.5 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 50 --seed 1234 > logs_sle/16_50_1.txt
python -u -W ignore train.py --id 16_50_1_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.5 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 50 --seed 2345 > logs_sle/16_50_1_1.txt
python -u -W ignore train.py --id 16_50_1_2 --gpu_id 1 --batch_size 128 --semantic_weight 0.5 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 50 --seed 3456 > logs_sle/16_50_1_2.txt

python -u -W ignore train.py --id 12_75_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 75 --seed 1234 > logs_sle/12_75_1.txt
python -u -W ignore train.py --id 12_75_1_1 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 75 --seed 2345 > logs_sle/12_75_1_1.txt
python -u -W ignore train.py --id 12_75_1_2 --gpu_id 1 --batch_size 128 --semantic_weight 0.1 --entreg_weight 0.01  --save_dir './saved_models/sle'  --num_samples 75 --seed 3456 > logs_sle/12_75_1_2.txt
