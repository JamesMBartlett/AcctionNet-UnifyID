from GAF_MTF_utils import *
import numpy as np

def GAF(img, rescale_type='zero', GAF_type='GADF', paa_size=64):
    output_img = np.array([None] * img.shape[-1] * paa_size * paa_size, dtype=float)
    output_img = output_img.reshape(paa_size, paa_size, img.shape[-1])
    for index, channel in enumerate(img.T):
        if rescale_type == 'zero':
            std_data = rescale(channel)
        elif rescale_type == 'minusone':
            std_data = rescaleminus(channel)
        else:
            raise Exception("Unknown rescaling type")
        paacos = np.array(paa(std_data, paa_size, None))
        paasin = np.sqrt(1-paacos**2)
        paacos = np.matrix(paacos)
        paasin = np.matrix(paasin)
        if GAF_type == 'GASF':
            paamatrix = paacos.T*paacos - paasin.T*paasin
        elif GAF_type == 'GADF':
            paamatrix = paasin.T * paacos - paacos.T*paasin
        else:
            raise Exception("Unknown GAF Type")
        output_img[:,:,index] = paamatrix
    return output_img

