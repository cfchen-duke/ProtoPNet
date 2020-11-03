import os
import copy
import time

import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop


# XXX my two ideas here are weight averaging for push and a memory bank. I think memory bank is the most
# interesting item to implement first, but they both have similar implementation in code
class Pusher(object):
    def __init__(self,
                 dataloader,
                 prototype_network_parallel, # pytorch network with prototype_vectors
                 bank_size,
                 class_specific=True,
                 preprocess_input_function=None, # normalize if needed
                 prototype_layer_stride=1,
                 dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                 prototype_img_filename_prefix=None,
                 prototype_self_act_filename_prefix=None,
                 proto_bound_boxes_filename_prefix=None,
                 save_prototype_class_identity=True, # which class the prototype image comes from
                 log=print,
                 prototype_activation_function_in_numpy=None):
        self.dataloader = dataloader
        self.prototype_network_parallel = prototype_network_parallel
        self.class_specific = class_specific
        self.preprocess_input_function = preprocess_input_function
        self.prototype_layer_stride = prototype_layer_stride
        self.dir_for_saving_prototypes = dir_for_saving_prototypes
        self.prototype_img_filename_prefix = prototype_img_filename_prefix
        self.prototype_self_act_filename_prefix = prototype_self_act_filename_prefix
        self.save_prototype_class_identity = save_prototype_class_identity
        self.log = log
        self.prototype_activation_function_in_numpy = prototype_activation_function_in_numpy
        self.bank_size = bank_size

    def push_orig(self, epoch_number):
        self.prototype_network_parallel.eval()
        self.log('\tpush')

        start = time.time()
        prototype_shape = self.prototype_network_parallel.module.prototype_shape
        n_prototypes = self.prototype_network_parallel.module.num_prototypes
        # saves the closest distance seen so far
        global_min_proto_dist = np.full(n_prototypes, np.inf)
        # saves the patch representation that gives the current smallest distance
        global_min_fmap_patches = np.zeros(
            [n_prototypes,
             prototype_shape[1],
             prototype_shape[2],
             prototype_shape[3]])

        '''
        proto_rf_boxes and proto_bound_boxes column:
        0: image index in the entire dataset
        1: height start index
        2: height end index
        3: width start index
        4: width end index
        5: (optional) class identity
        '''
        if self.save_prototype_class_identity:
            proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                        fill_value=-1)
            proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                                fill_value=-1)
        else:
            proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                        fill_value=-1)
            proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                                fill_value=-1)

        if self.dir_for_saving_prototypes != None:
            if epoch_number != None:
                proto_epoch_dir = os.path.join(self.dir_for_saving_prototypes,
                                               'epoch-'+str(epoch_number))
                makedir(proto_epoch_dir)
            else:
                # XXX I think dir_for_saving_proto and root_dir are actually
                # different variables and it wasnt a misnaming. Oh well
                # I'll come back to this later
                proto_epoch_dir = self.dir_for_saving_prototypes
        else:
            proto_epoch_dir = None

        search_batch_size = self.dataloader.batch_size

        num_classes = self.prototype_network_parallel.module.num_classes

        for push_iter, (search_batch_input, search_y) in enumerate(self.dataloader):
            '''
            start_index_of_search keeps track of the index of the image
            assigned to serve as prototype
            '''
            start_index_of_search_batch = push_iter * search_batch_size

            self.update_prototypes_on_batch(search_batch_input,
                                            start_index_of_search_batch,
                                            global_min_proto_dist,
                                            global_min_fmap_patches,
                                            proto_rf_boxes,
                                            proto_bound_boxes,
                                            search_y=search_y,
                                            num_classes=num_classes)

        if proto_epoch_dir != None and self.proto_bound_boxes_filename_prefix != None:
            np.save(os.path.join(proto_epoch_dir, self.proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                    proto_rf_boxes)
            np.save(os.path.join(proto_epoch_dir, self.proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                    proto_bound_boxes)

        # XXX push here is different because we're choosing top K vectors.
        self.log('\tExecuting push ...')
        prototype_update = np.reshape(global_min_fmap_patches,
                                      tuple(prototype_shape))
        self.prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
        # prototype_network_parallel.cuda()
        end = time.time()
        self.log('\tpush time: \t{0}'.format(end -  start))

    def push_protobank(self, epoch_number):
        self.prototype_network_parallel.eval()
        self.log('\tpush')

        start = time.time()
        prototype_shape = self.prototype_network_parallel.module.prototype_shape
        protobank_shape = list(prototype_shape)
        protobank_shape[0] = protobank_shape[0] * self.bank_size
        n_prototypes = self.prototype_network_parallel.module.num_prototypes
        # XXX skip proto_rf impl
        #
        # XXX skip dir save impl
        search_batch_size = self.dataloader.batch_size
        num_classes = self.prototype_network_parallel.module.num_classes
        all_proto_dist = np.full((self.bank_size, n_prototypes), np.inf)
        # saves the patch representation that gives the current smallest distance
        # this is really the only var we care about in the entire update process
        all_fmap_patches = np.zeros([
            self.bank_size, n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]
        ])

        for push_iter, (search_batch_input, search_y) in enumerate(self.dataloader):
            '''
            start_index_of_search keeps track of the index of the image
            assigned to serve as prototype
            '''
            start_index_of_search_batch = push_iter * search_batch_size

            self.generic_update_search(search_batch_input,
                                       start_index_of_search_batch,
                                       all_proto_dist,
                                       all_fmap_patches,
                                       search_y=search_y,
                                       num_classes=num_classes)

        # XXX didnt impl bound boxes filename
        self.log('\tExecuting push ...')
        prototype_update = np.reshape(all_fmap_patches, tuple(protobank_shape))
        self.prototype_network_parallel.module.protobank_tensor.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
        # prototype_network_parallel.cuda()
        end = time.time()
        self.log('\tTotal push time: \t{0}'.format(end -  start))

    def generic_update_search(self,
                              search_batch_input,
                              start_index_of_search_batch,
                              all_proto_dist,
                              all_fmap_patches,
                              search_y,
                              num_classes):
        """
        Make generic update on prototype vectors based on some kind of
        memory bank available to us. This follows a top-K method
        """
        self.prototype_network_parallel.eval()
        # XXX no preprocess func implemented
        with torch.no_grad():
            search_batch_input = search_batch_input.cuda()
            # you could run this across multi-gpu but you'd have to make mod
            # to the DataParallel class because it doesn't allow you to
            # call other methods besides forward func
            conv_features, distances = self.prototype_network_parallel.module.protobank_distances(search_batch_input)

        pool = nn.MaxPool2d(distances.size()[-1], distances.size()[-1], return_indices=True)
        protoL_input_ = np.copy(conv_features.detach().cpu().numpy())
        proto_dist_ = np.copy(distances.detach().cpu().numpy())
        pooled, pooled_idx = pool(-distances)
        pooled = -pooled.detach().cpu().numpy()

        # XXX non-class specific not implemented
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

        prototype_shape = self.prototype_network_parallel.module.prototype_shape
        n_prototypes = prototype_shape[0]
        proto_h = prototype_shape[2]
        proto_w = prototype_shape[3]
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        for j in range(n_prototypes):
            #if n_prototypes_per_class != None:
            # target_class is the class of the class_specific prototype
            # XXX non-class specific not implemented
            target_class = torch.argmax(self.prototype_network_parallel.module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j]
            # XXX well apparently this function is more complex than i realized. Its not
            # just finding a specific prototype input, but it finds a specific pixel
            # that they're trying to set. This complicates the actual setting of
            # the conv value because in my impl i was doing max pooling and setting
            # by row.
            #
            # XXX So this in turn leads me to question the whole thing about push. I mean
            # at the end, its just focused on minimizing the l2 distance to 0
            # after min pooling. But why pick the min for the push? Why not a mean?
            # I should read thru the math again to see if theres rationale for the min
            # I mean, setting it to anything could have the desired outcome of
            # minimizing l2
            for i in range(self.bank_size):
                batch_min_proto_dist_j = np.amin(proto_dist_j)
                # this works originally because the pooling done is basically an
                # argmin across the 7x7. If you want to simplfiy this you can
                # directly perform the argmin here.
                which_less_than = batch_min_proto_dist_j < all_proto_dist[:, j]
                if which_less_than.any():
                    batch_argmin_proto_dist_j = \
                        list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                              proto_dist_j.shape))
                    # XXX no non-class specific
                    img_index_in_batch = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

                    # retrieve the corresponding feature map patch
                    fmap_height_start_index = batch_argmin_proto_dist_j[1] * self.prototype_layer_stride
                    fmap_height_end_index = fmap_height_start_index + proto_h
                    fmap_width_start_index = batch_argmin_proto_dist_j[2] * self.prototype_layer_stride
                    fmap_width_end_index = fmap_width_start_index + proto_w

                    batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                           :,
                                                           fmap_height_start_index:fmap_height_end_index,
                                                           fmap_width_start_index:fmap_width_end_index]

                    # find the bank idx to update first and then do it
                    at_bank_idx = np.searchsorted(which_less_than, True)
                    # do insert instead of update, and then truncate
                    dist_vals = all_proto_dist[:, j]
                    all_proto_dist[:, j] = np.insert(dist_vals, at_bank_idx, batch_min_proto_dist_j)[:-1]
                    fmap_vals = all_fmap_patches[:, j]
                    all_fmap_patches[:, j] = np.insert(fmap_vals, at_bank_idx, batch_min_fmap_patch_j,axis=0)[:-1]
                    # remove the rest of this. I'd like to separate it out into another
                    # function. Currently we dont need it and we can always refer to the
                    # original function if we want the code

    def update_prototypes_on_batch(self,
                                   search_batch_input,
                                   start_index_of_search_batch,
                                   global_min_proto_dist, # this will be updated
                                   global_min_fmap_patches, # this will be updated
                                   proto_rf_boxes, # this will be updated
                                   proto_bound_boxes, # this will be updated
                                   search_y=None, # required if class_specific == True
                                   num_classes=None): # required if class_specific == True

        self.prototype_network_parallel.eval()

        if self.preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = self.preprocess_input_function(search_batch_input)

        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.cuda()
            # this computation currently is not parallelized
            conv_features, distances = self.prototype_network_parallel.module.prototype_distances(search_batch)

        protoL_input_ = np.copy(conv_features.detach().cpu().numpy())
        proto_dist_ = np.copy(distances.detach().cpu().numpy())

        del conv_features, distances

        if self.class_specific:
            class_to_img_index_dict = {key: [] for key in range(num_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(search_y):
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)

        prototype_shape = self.prototype_network_parallel.module.prototype_shape
        n_prototypes = prototype_shape[0]
        proto_h = prototype_shape[2]
        proto_w = prototype_shape[3]
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        for j in range(n_prototypes):
            #if n_prototypes_per_class != None:
            if self.class_specific:
                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(self.prototype_network_parallel.module.prototype_class_identity[j]).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
            else:
                # if it is not class specific, then we will search through
                # every example
                proto_dist_j = proto_dist_[:,j,:,:]

            batch_min_proto_dist_j = np.amin(proto_dist_j)
            if batch_min_proto_dist_j < global_min_proto_dist[j]:
                batch_argmin_proto_dist_j = \
                    list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                          proto_dist_j.shape))
                if self.class_specific:
                    '''
                    change the argmin index from the index among
                    images of the target class to the index in the entire search
                    batch
                    '''
                    batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmin_proto_dist_j[0]
                fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
                fmap_height_end_index = fmap_height_start_index + proto_h
                fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
                fmap_width_end_index = fmap_width_start_index + proto_w

                batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                       :,
                                                       fmap_height_start_index:fmap_height_end_index,
                                                       fmap_width_start_index:fmap_width_end_index]

                global_min_proto_dist[j] = batch_min_proto_dist_j
                global_min_fmap_patches[j] = batch_min_fmap_patch_j

                # get the receptive field boundary of the image patch
                # that generates the representation
                protoL_rf_info = self.prototype_network_parallel.module.proto_layer_rf_info
                rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)

                # get the whole image
                original_img_j = search_batch_input[rf_prototype_j[0]]
                original_img_j = original_img_j.numpy()
                original_img_j = np.transpose(original_img_j, (1, 2, 0))
                original_img_size = original_img_j.shape[0]

                # crop out the receptive field
                rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                          rf_prototype_j[3]:rf_prototype_j[4], :]

                # save the prototype receptive field information
                proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
                proto_rf_boxes[j, 1] = rf_prototype_j[1]
                proto_rf_boxes[j, 2] = rf_prototype_j[2]
                proto_rf_boxes[j, 3] = rf_prototype_j[3]
                proto_rf_boxes[j, 4] = rf_prototype_j[4]
                if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                    proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

                # find the highly activated region of the original image
                proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
                if self.prototype_network_parallel.module.prototype_activation_function == 'log':
                    proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + self.prototype_network_parallel.module.epsilon))
                elif self.prototype_network_parallel.module.prototype_activation_function == 'linear':
                    proto_act_img_j = max_dist - proto_dist_img_j
                else:
                    proto_act_img_j = self.prototype_activation_function_in_numpy(proto_dist_img_j)
                upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                                 interpolation=cv2.INTER_CUBIC)
                proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
                # crop out the image patch with high activation as prototype image
                proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                             proto_bound_j[2]:proto_bound_j[3], :]

                # save the prototype boundary (rectangular boundary of highly activated region)
                proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
                proto_bound_boxes[j, 1] = proto_bound_j[0]
                proto_bound_boxes[j, 2] = proto_bound_j[1]
                proto_bound_boxes[j, 3] = proto_bound_j[2]
                proto_bound_boxes[j, 4] = proto_bound_j[3]
                if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                    proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

                if self.dir_for_saving_prototypes is not None:
                    if self.prototype_self_act_filename_prefix is not None:
                        # save the numpy array of the prototype self activation
                        np.save(os.path.join(self.dir_for_saving_prototypes,
                                             self.prototype_self_act_filename_prefix + str(j) + '.npy'),
                                proto_act_img_j)
                    if self.prototype_img_filename_prefix is not None:
                        # save the whole image containing the prototype as png
                        plt.imsave(os.path.join(self.dir_for_saving_prototypes,
                                                self.prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                                   original_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        # overlay (upsampled) self activation on original image and save the result
                        rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                        heatmap = np.float32(heatmap) / 255
                        heatmap = heatmap[...,::-1]
                        overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                        plt.imsave(os.path.join(self.dir_for_saving_prototypes,
                                                self.prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                                   overlayed_original_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

                        # if different from the original (whole) image, save the prototype receptive field as png
                        if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                            plt.imsave(os.path.join(self.dir_for_saving_prototypes,
                                                    self.prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                       rf_img_j,
                                       vmin=0.0,
                                       vmax=1.0)
                            overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                          rf_prototype_j[3]:rf_prototype_j[4]]
                            plt.imsave(os.path.join(self.dir_for_saving_prototypes,
                                                    self.prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                       overlayed_rf_img_j,
                                       vmin=0.0,
                                       vmax=1.0)

                        # save the prototype image (highly activated region of the whole image)
                        plt.imsave(os.path.join(self.dir_for_saving_prototypes,
                                                self.prototype_img_filename_prefix + str(j) + '.png'),
                                   proto_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

        if class_specific:
            del class_to_img_index_dict
