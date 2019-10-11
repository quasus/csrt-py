import numpy as np
import cv2
from . import _gradient

def mround(x):
    x_ = x.copy()
    idx = (x - np.floor(x)) >= 0.5
    x_[idx] = np.floor(x[idx]) + 1
    idx = ~idx
    x_[idx] = np.floor(x[idx])
    return x_


class OTBHCConfig:
    fhog_params = {'fname': 'fhog',
                   'num_orients': 9,
                   'cell_size': 6,
                   'compressed_dim': 10,
                   }

    cn_params = {"fname": 'cn',
                 "table_name": "CNnorm",
                 "use_for_color": True,
                 "cell_size": 4,
                 "compressed_dim": 3,
                 }

    ic_params = {'fname': 'ic',
                 "table_name": "intensityChannelNorm6",
                 "use_for_color": False,
                 "cell_size": 4,
                 "compressed_dim": 3,
                 }

    features = [fhog_params, cn_params, ic_params]

    # feature parameters
    normalize_power = 2                 # Lp normalization with this p
    normalize_size = True               # also normalize with respect to the spatial size of the feature
    normalize_dim = True                # also normalize with respect to the dimensionality of the feature
    square_root_normalization = False   #

    # image sample parameters
    search_area_shape = 'square'        # the shape of the samples
    search_area_scale = 4.0             # the scaling of the target size to get the search area
    min_image_sample_size = 150 ** 2    # minimum area of image samples
    max_image_sample_size = 200 ** 2    # maximum area of image samples

    # detection parameters
    refinement_iterations = 1           # number of iterations used to refine the resulting position in a frame
    newton_iterations = 5               # the number of Netwon iterations used for optimizing the detection score
    clamp_position = False              # clamp the target position to be inside the image

    # learning parameters
    output_sigma_factor = 1 / 14.       # label function sigma
    learning_rate = 0.009               # learning rate
    num_samples = 30                    # maximum number of stored training samples
    sample_replace_startegy = 'lowest_prior' # which sample to replace when the memory is full
    lt_size = 0                         # the size of the long-term memory (where all samples have equal weight)
    train_gap = 5                       # the number of intermediate frames with no training (0 corresponds to the training every frame)
    skip_after_frame = 10                # after which frame number the sparse update scheme should start (1 is directly)
    use_detection_sample = True         # use the sample that was extracted at the detection stage also for learning

    # factorized convolution parameters
    use_projection_matrix = True        # use projection matrix, i.e. use the factorized convolution formulation
    update_projection_matrix = True     # whether the projection matrix should be optimized or not
    proj_init_method = 'pca'            # method for initializing the projection matrix
    projection_reg = 1e-7               # regularization parameter of the projection matrix

    # generative sample space model parameters
    use_sample_merge = True             # use the generative sample space model to merge samples
    sample_merge_type = 'merge'         # strategy for updating the samples
    distance_matrix_update_type = 'exact' # strategy for updating the distance matrix

    # CG paramters
    CG_iter = 5                         # the number of Conjugate Gradient iterations in each update after the first time
    init_CG_iter = 5 * 15              # the total number of Conjugate Gradient iterations used in the first time
    init_GN_iter = 5                   # the number of Gauss-Netwon iterations used in the first frame (only if the projection matrix is updated)
    CG_use_FR = False                   # use the Fletcher-Reeves or Polak-Ribiere formula in the Conjugate Gradient
    CG_standard_alpha = True            # use the standard formula for computing the step length in Conjugate Gradient
    CG_forgetting_rate = 50             # forgetting rate of the last conjugate direction
    precond_data_param = 0.75           # weight of the data term in the preconditioner
    precond_reg_param = 0.25            # weight of the regularization term in the preconditioner
    precond_proj_param = 40             # weight of the projection matrix part in the preconditioner

    # regularization window paramters
    use_reg_window = True               # use spatial regularizaiton or not
    reg_window_min = 1e-4               # the minimum value of the regularization window
    reg_window_edge = 10e-3             # the impace of the spatial regularization
    reg_window_power = 2                # the degree of the polynomial to use (e.g. 2 is q quadratic window)
    reg_sparsity_threshold = 0.05       # a relative threshold of which DFT coefficients of the kernel

    # interpolation parameters
    interp_method = 'bicubic'           # the kind of interpolation kernel
    interp_bicubic_a = -0.75            # the parameter for the bicubic interpolation kernel
    interp_centering = True             # center the kernel at the feature sample
    interp_windowing = False            # do additional windowing on the Fourier coefficients of the kernel

    # scale parameters
    # number_of_scales = 5              # number of scales to run the detector
    # scale_step = 1.01                 # the scale factor
    use_scale_filter = True             # use the fDSST scale filter or not

    # only used if use_scale_filter == true
    scale_sigma_factor = 1 / 16.        # scale label function sigma
    scale_learning_rate = 0.025         # scale filter learning rate
    number_of_scales_filter = 17        # number of scales
    number_of_interp_scales = 33        # number of interpolated scales
    scale_model_factor = 1.0            # scaling of the scale model
    scale_step_filter = 1.02            # the scale factor of the scale sample patch
    scale_model_max_area = 32 * 16      # maximume area for the scale sample patch
    scale_feature = 'HOG4'              # features for the scale filter (only HOG4 supported)
    s_num_compressed_dim = 'MAX'        # number of compressed feature dimensions in the scale filter
    lamBda = 1e-2                       # scale filter regularization
    do_poly_interp = True               # do 2nd order polynomial interpolation to obtain more accurate scale


    vis = True


class Feature:
    def __init__(self,config=OTBHCConfig()):
        self.config=config

    def init_size(self, img_sample_sz, cell_size=None):
        if cell_size is not None:
            max_cell_size = max(cell_size)
            new_img_sample_sz = (1 + 2 * mround(img_sample_sz / ( 2 * max_cell_size))) * max_cell_size
            feature_sz_choices = np.array([(new_img_sample_sz.reshape(-1, 1) + np.arange(0, max_cell_size).reshape(1, -1)) // x for x in cell_size])
            num_odd_dimensions = np.sum((feature_sz_choices % 2) == 1, axis=(0,1))
            best_choice = np.argmax(num_odd_dimensions.flatten())
            img_sample_sz = mround(new_img_sample_sz + best_choice)

        self.sample_sz = img_sample_sz
        self.data_sz = [img_sample_sz // self._cell_size]
        return img_sample_sz

    def _sample_patch(self, im, pos, sample_sz, output_sz):
        pos = np.floor(pos)
        sample_sz = np.maximum(mround(sample_sz), 1)
        xs = np.floor(pos[1]) + np.arange(0, sample_sz[1]+1) - np.floor((sample_sz[1]+1)/2)
        ys = np.floor(pos[0]) + np.arange(0, sample_sz[0]+1) - np.floor((sample_sz[0]+1)/2)
        xmin = max(0, int(xs.min()))
        xmax = min(im.shape[1], int(xs.max()))
        ymin = max(0, int(ys.min()))
        ymax = min(im.shape[0], int(ys.max()))
        # extract image
        im_patch = im[ymin:ymax, xmin:xmax, :]
        left = right = top = down = 0
        if xs.min() < 0:
            left = int(abs(xs.min()))
        if xs.max() > im.shape[1]:
            right = int(xs.max() - im.shape[1])
        if ys.min() < 0:
            top = int(abs(ys.min()))
        if ys.max() > im.shape[0]:
            down = int(ys.max() - im.shape[0])
        if left != 0 or right != 0 or top != 0 or down != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, down, left, right, cv2.BORDER_REPLICATE)
        # im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])))
        im_patch = cv2.resize(im_patch, (int(output_sz[1]), int(output_sz[0])), cv2.INTER_CUBIC)
        if len(im_patch.shape) == 2:
            im_patch = im_patch[:, :, np.newaxis]
        return im_patch

    def _feature_normalization(self, x):
        if hasattr(self.config, 'normalize_power') and self.config.normalize_power > 0:
            if self.config.normalize_power == 2:
                x = x * np.sqrt((x.shape[0]*x.shape[1]) ** self.config.normalize_size * (x.shape[2] ** self.config.normalize_dim) / (x ** 2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0]*x.shape[1]) ** self.config.normalize_size) * (x.shape[2] ** self.config.normalize_dim) / ((np.abs(x) ** (1. / self.config.normalize_power)).sum(axis=(0, 1, 2)))

        if self.config.square_root_normalization:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x.astype(np.float32)




class TableFeature(Feature):

    tables = {}

    def __init__(self, fname, compressed_dim, table_name, use_for_color, cell_size=1,config=OTBHCConfig()):
        super(TableFeature,self).__init__(config)
        self.fname = fname
        self._table_name = table_name
        self._color = use_for_color
        self._cell_size = cell_size
        self._compressed_dim = [compressed_dim]
        self._factor = 32
        self._den = 8
        self._table = TableFeature.tables[self._table_name]

        self.num_dim = [self._table.shape[1]]
        self.min_cell_size = self._cell_size
        self.penalty = [0.]
        self.sample_sz = None
        self.data_sz = None

    def integralVecImage(self, img):
        w, h, c = img.shape
        intImage = np.zeros((w+1, h+1, c), dtype=img.dtype)
        intImage[1:, 1:, :] = np.cumsum(np.cumsum(img, 0), 1)
        return intImage

    def average_feature_region(self, features, region_size):
        region_area = region_size ** 2
        if features.dtype == np.float32:
            maxval = 1.
        else:
            maxval = 255
        intImage = self.integralVecImage(features)
        i1 = np.arange(region_size, features.shape[0]+1, region_size).reshape(-1, 1)
        i2 = np.arange(region_size, features.shape[1]+1, region_size).reshape(1, -1)
        region_image = (intImage[i1, i2, :] - intImage[i1, i2-region_size,:] - intImage[i1-region_size, i2, :] + intImage[i1-region_size, i2-region_size, :])  / (region_area * maxval)
        return region_image

    def get_features(self, img, pos, sample_sz, scales,normalization=True):
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            h, w, c = patch.shape
            if c == 3:
                RR = patch[:, :, 0].astype(np.int32)
                GG = patch[:, :, 1].astype(np.int32)
                BB = patch[:, :, 2].astype(np.int32)
                index = RR // self._den + (GG // self._den) * self._factor + (BB // self._den) * self._factor * self._factor
                features = self._table[index.flatten()].reshape((h, w, self._table.shape[1]))
            else:
                features = self._table[patch.flatten()].reshape((h, w, self._table.shape[1]))
            if self._cell_size > 1:
                features = self.average_feature_region(features, self._cell_size)
            feat.append(features)
        feat=np.stack(feat, axis=3)
        if normalization is True:
            feat = self._feature_normalization(feat)
        return [feat]



def fhog(I, bin_size=8, num_orients=9, clip=0.2, crop=False):
    soft_bin = -1
    M, O = _gradient.gradMag(I.astype(np.float32), 0, True)
    H = _gradient.fhog(M, O, bin_size, num_orients, soft_bin, clip)
    return H


def extract_hog_feature(img, cell_size=4):
    fhog_feature=fhog(img.astype(np.float32),cell_size,num_orients=9,clip=0.2)[:,:,:-1]
    return fhog_feature


def extract_cn_feature(img,cell_size=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    cn = TableFeature(fname='cn', cell_size=cell_size, compressed_dim=11, table_name="CNnorm",
                      use_for_color=True)

    if np.all(img[:, :, 0] == img[:, :, 1]):
        img = img[:, :, :1]
    else:
        # # pyECO using RGB format
        img = img[:, :, ::-1]
    h,w=img.shape[:2]
    cn_feature = \
    cn.get_features(img, np.array(np.array([h/2,w/2]), dtype=np.int16), np.array([h,w]), 1, normalization=False)[
        0][:, :, :, 0]
    # print('cn_feature.shape:', cn_feature.shape)
    # print('cnfeature:',cn_feature.shape,cn_feature.min(),cn_feature.max())
    gray = cv2.resize(gray, (cn_feature.shape[1], cn_feature.shape[0]))[:, :, np.newaxis]
    out = np.concatenate((gray, cn_feature), axis=2)
    return out
