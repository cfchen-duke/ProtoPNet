experiment_task = 'CBIS_massBenignMalignant' #'ADNI' 
# base_architecture = 'vgg19'
base_architecture = 'resnet18' #'resnet50'
#base_architecture = 'densenet121'


img_size = 224 #124# 224 #336
num_classes = 2 
num_prots_per_class = 10
num_filters = 512 #256 #128 #256
prototype_shape = (num_classes*num_prots_per_class, num_filters, 1, 1) #40 #60 #16 #40
prototype_activation_function = 'log'
add_on_layers_type = 'regular'


data_path = './datasets/' #
train_dir = data_path + 'push_augmented/' #
# train_dir = data_path + 'push/' # TODO


test_dir = data_path + 'valid' #'valid/' #
# test_dir = data_path + 'valid_augmented' #'valid/' #
# test_dir = data_path + 'test/' #'valid/' #TODO

train_push_dir = data_path + 'push/' #
train_batch_size = 40 #25 #50 #35 #80 #20 #30 #25 #12 #20 #
test_batch_size = 2 #20 #60 #40 #100 #20 #25 #20 #10 #
train_push_batch_size = 40 #25 #50 #35 #75 #20 #30 #25 #15 #20 # 10 #

joint_optimizer_lrs = {'features': 1e-06,#1e-4 #TODO
                       'add_on_layers': 1e-06, #3e-3,
                       'prototype_vectors': 1e-06} #3e-3}
joint_lr_step_size = 10 #5 #TODO

warm_optimizer_lrs = {'add_on_layers': 1e-06, #3e-3,
                      'prototype_vectors': 1e-06} #3e-3}

last_layer_optimizer_lr = 1e-06 #1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4
}

num_train_epochs = 1000 #TODO
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

from time import gmtime,strftime
experiment_run = f'{experiment_task}_{strftime("%a_%d_%b_%Y_%H:%M:%S", gmtime())}'
