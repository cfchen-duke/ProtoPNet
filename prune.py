import os
import shutil
from collections import Counter
import numpy as np
import torch

from helpers import makedir
import find_nearest

def prune_prototypes(dataloader,
                     prototype_network_parallel,
                     k,
                     prune_threshold,
                     preprocess_input_function,
                     original_model_dir,
                     epoch_number,
                     #model_name=None,
                     log=print,
                     copy_prototype_imgs=True):
    ### run global analysis
    nearest_train_patch_class_ids = \
        find_nearest.find_k_nearest_patches_to_prototypes(dataloader=dataloader,
                                                          prototype_network_parallel=prototype_network_parallel,
                                                          k=k,
                                                          preprocess_input_function=preprocess_input_function,
                                                          full_save=False,
                                                          log=log)

    ### find prototypes to prune
    original_num_prototypes = prototype_network_parallel.module.num_prototypes
    
    prototypes_to_prune = []
    for j in range(prototype_network_parallel.module.num_prototypes):
        class_j = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
        nearest_train_patch_class_counts_j = Counter(nearest_train_patch_class_ids[j])
        # if no such element is in Counter, it will return 0
        if nearest_train_patch_class_counts_j[class_j] < prune_threshold:
            prototypes_to_prune.append(j)

    log('k = {}, prune_threshold = {}'.format(k, prune_threshold))
    log('{} prototypes will be pruned'.format(len(prototypes_to_prune)))

    ### bookkeeping of prototypes to be pruned
    class_of_prototypes_to_prune = \
        torch.argmax(
            prototype_network_parallel.module.prototype_class_identity[prototypes_to_prune],
            dim=1).numpy().reshape(-1, 1)
    prototypes_to_prune_np = np.array(prototypes_to_prune).reshape(-1, 1)
    prune_info = np.hstack((prototypes_to_prune_np, class_of_prototypes_to_prune))
    makedir(os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number,
                                          k,
                                          prune_threshold)))
    np.save(os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number,
                                          k,
                                          prune_threshold), 'prune_info.npy'),
            prune_info)

    ### prune prototypes
    prototype_network_parallel.module.prune_prototypes(prototypes_to_prune)
    #torch.save(obj=prototype_network_parallel.module,
    #           f=os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number,
    #                                              k,
    #                                              prune_threshold),
    #                          model_name + '-pruned.pth'))
    if copy_prototype_imgs:
        original_img_dir = os.path.join(original_model_dir, 'img', 'epoch-%d' % epoch_number)
        dst_img_dir = os.path.join(original_model_dir,
                                   'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number,
                                                                               k,
                                                                               prune_threshold),
                                   'img', 'epoch-%d' % epoch_number)
        makedir(dst_img_dir)
        prototypes_to_keep = list(set(range(original_num_prototypes)) - set(prototypes_to_prune))
        
        for idx in range(len(prototypes_to_keep)):
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img%d.png' % prototypes_to_keep[idx]),
                            dst=os.path.join(dst_img_dir, 'prototype-img%d.png' % idx))
            
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img-original%d.png' % prototypes_to_keep[idx]),
                            dst=os.path.join(dst_img_dir, 'prototype-img-original%d.png' % idx))
            
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img-original_with_self_act%d.png' % prototypes_to_keep[idx]),
                            dst=os.path.join(dst_img_dir, 'prototype-img-original_with_self_act%d.png' % idx))
            
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-self-act%d.npy' % prototypes_to_keep[idx]),
                            dst=os.path.join(dst_img_dir, 'prototype-self-act%d.npy' % idx))


            bb = np.load(os.path.join(original_img_dir, 'bb%d.npy' % epoch_number))
            bb = bb[prototypes_to_keep]
            np.save(os.path.join(dst_img_dir, 'bb%d.npy' % epoch_number),
                    bb)

            bb_rf = np.load(os.path.join(original_img_dir, 'bb-receptive_field%d.npy' % epoch_number))
            bb_rf = bb_rf[prototypes_to_keep]
            np.save(os.path.join(dst_img_dir, 'bb-receptive_field%d.npy' % epoch_number),
                    bb_rf)
    
    return prune_info
