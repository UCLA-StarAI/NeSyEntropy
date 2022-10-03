from eval_script import evaluate
import os

# python3 eval.py > eval_out.txt

model_base_dir = 'saved_models'

dirs = os.listdir(model_base_dir)
for e in dirs:
    dirs2 = os.listdir(model_base_dir+'/'+e)
    for d in dirs2:
        seed = 1234
        parts = d.split('_')
        if 'base' in parts or 'codl' in parts:
            if len(parts) == 4:
                if parts[-1] == '1' or parts[-1] == '3':
                    seed = 2345
                elif parts[-1] == '2' or parts[-1] == '4':
                    seed = 3456
        else:
            if len(parts) == 3:
                if parts[-1] == '1' or parts[-1] == '3':
                    seed = 2345
                elif parts[-1] == '2' or parts[-1] == '4':
                    seed = 3456

        if e[-1] == 't':
            print('\n#######################')
            print(model_base_dir+'/'+e+'/'+d)
            print('seed: '+str(seed))
            print('transductive')
            print('#######################\n')
            evaluate(model_base_dir+'/'+e+'/'+d, data_dir='dataset/sciERC', seed=seed, gpu_id=0)
        else:
            # test with and without ilp
            print('\n#######################')
            print(model_base_dir+'/'+e+'/'+d)
            print('seed: '+str(seed))
            print('yes ilp')
            print('#######################\n')
            evaluate(model_base_dir+'/'+e+'/'+d, data_dir='dataset/sciERC', seed=seed, gpu_id=0, ilp=True)