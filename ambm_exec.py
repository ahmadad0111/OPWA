import os

def run_exp(num_batch=12, shot=20, query=15, lr1=0.0001, lr2=0.0005, base_lr=0.005, hyper_lr=1e-4, clip_hyper=10.0, layer_filters=['conv1', 'conv2', 'conv3'], subject_agnostic=False, update_step=20, gamma=0.8):
    max_episode = 5
    way = 4
    step_size = 3
    gpu = 0
       
    the_command = 'python3 main.py' \
        + ' --max_episode=' + str(max_episode) \
        + ' --num_batch=' + str(num_batch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --meta_lr1=' + str(lr1) \
        + ' --meta_lr2=' + str(lr2) \
        + ' --step_size=' + str(step_size) \
        + ' --gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --update_step=' + str(update_step) 

    os.system(the_command + ' --phase=meta_train')
    os.system(the_command + ' --phase=meta_eval')

run_exp(num_batch=12, shot=20, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, hyper_lr=1e-4, clip_hyper=10.0, layer_filters=['conv1', 'conv2', 'conv3'], update_step=20, gamma=0.8)



