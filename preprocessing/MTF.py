from GAF_MTF_utils import *
import numpy as np

def MTF(X, paa_size=64, Q=16):
    output_imgs = np.array([None] * X.shape[0] * paa_size * paa_size * X.shape[-1], dtype=float)
    output_imgs = output_imgs.reshape(X.shape[0], paa_size, paa_size, X.shape[-1])
    for i, raw_img in enumerate(X):
        output_img = np.array([None] * X.shape[-1] * paa_size * paa_size, dtype=float)
        output_img = output_img.reshape(paa_size, paa_size, X.shape[-1])
        for index, channel in enumerate(raw_img.T):
            std_data = channel
            paalist = paa(std_data, paa_size, None)
            mat, matindex, level = QMeq(std_data, Q)
            paamatindex = paaMarkovMatrix(paalist, level)
            paacolumn = []
            for p in range(paa_size):
                for q in range(paa_size):
                    paacolumn.append(mat[paamatindex[p]][paamatindex[(q)]])

            output_img[:,:,index] = np.array(paacolumn).reshape(paa_size, paa_size)
        output_imgs[i] = output_img
    return output_imgs
