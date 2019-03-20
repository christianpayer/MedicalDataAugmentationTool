
from scipy.ndimage import gaussian_filter
import numpy as np
from utils.sitk_np import sitk_to_np_no_copy, np_to_sitk
from utils.io.image import write_np


class FrangiFilter:

    def __call__(self, img_sitk):

        img = sitk_to_np_no_copy(img_sitk)
        #write_np(img, 'img.nii')

        out = []
        for i in range(8):
            s = img_sitk.GetSpacing()[0] + 0.25 * float(i)
            img_yy = gaussian_filter(img, sigma=s, order=[0, 2, 0])
            I = np.maximum(-img_yy, 0)
            I /= np.maximum(np.amax(I), 1e-20)
            out.append(I)
            #write_np(out[i], 'plate' + str(i) + '.nii')

        OUT = np.amax(out, axis=0)
        #write_np(OUT, "plate.nii")

        out_sitk = np_to_sitk(OUT)

        return out_sitk

