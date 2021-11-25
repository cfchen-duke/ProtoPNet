base_architecture = 'resnet152'
img_size = 224
num_prototypes = 4
num_classes = 16
prototype_shape = (num_prototypes * num_classes, 128, 1, 1)
top_k_percentage = 10
prototype_activation_function = "log"
add_on_layers_type = "regular"

experiment_run = '003'

data_csv_path = 'one_hot_dataset.csv'
data_img_path = '../images'
data_path = '../images'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

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
