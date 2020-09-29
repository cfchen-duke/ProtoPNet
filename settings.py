import torch
import numpy as np

base_architecture = 'resnet152'
img_size = 224
prototype_shape = (15, 1024, 1, 1)
num_classes = 3
# def prototype_activation_function(distance):
#     return -torch.log(distance + 1e-4)
#
# def prototype_activation_function_in_numpy(distance):
#     return -np.log(distance + 1e-4)

prototype_activation_function = "log"
prototype_activation_function_in_numpy = prototype_activation_function

class_specific = True

add_on_layers_type = 'regular'

experiment_run = 'PPNETLesionOrNot0229'
data_path = "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/"
train_dir = data_path + 'binary_train_spiculated_augmented_morer_with_rot/'
# train_dir ='binary_train_spiculated_augmented_crazy/'
test_dir = data_path + 'binary_test_spiculated/'
train_push_dir = data_path + 'binary_train_spiculated/'



train_batch_size = 50
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 2e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 2e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-3

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
