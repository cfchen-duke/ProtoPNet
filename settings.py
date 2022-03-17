# base_architecture = 'vgg19'
base_architecture = 'resnet50'
#base_architecture = 'densenet121'


img_size = 224 #336
prototype_shape = (30, 512, 1, 1) #40 #60 #16 #40
num_classes = 2 
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '02_breastCBIS_mass_bm_224IMGSIZE_15PROTPERCLASS_512FILTERS'

data_path = './datasets/' #
train_dir = data_path + 'push_augmented/' #
test_dir = data_path + 'valid_augmented' #'valid/' #
train_push_dir = data_path + 'push/' #
train_batch_size = 20 #30 #25 #12 #20 #
test_batch_size = 20 #25 #20 #10 #
train_push_batch_size = 20 #30 #25 #15 #20 # 10 #

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.95, #0.95
    'sep': -0.08, #0.05
    'l1': 1e-4,
}

num_train_epochs = 300
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
