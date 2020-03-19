import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")

# check shape, zeros, visualization, and distribution

test_dir = "/usr/project/xtmp/mammo/binary_Feb/train_context_roi_augByWin/spiculated/" \
           "DP_ABAC_L_MLO_3#0#3.npy"
arr = np.load(test_dir)
print(arr.shape)
print(np.count_nonzero(arr))
plt.imsave("visualization", arr, cmap="gray")
arr = arr.flatten()
arr.sort()
plt.plot(arr)
plt.savefig("distribution")
plt.close()

#DP_AAPR_R_MLO_3#0#4.npy
#DP_AAXS_L_MLO_5#0#1.npy
#DP_AAYY_R_CC_1#0#1.npy
#DP_ABAC_L_MLO_3#0#3.npy
