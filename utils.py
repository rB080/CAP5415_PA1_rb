import numpy as np
import math
import cv2


def load_img(path, normalize=True, resize_shape=None):
    #########################################################################################
    # Loads Image, optionally converts BGR to greyscale
    # Parameters:
    # - path: Path for image (String)
    # - normalize: True if img is to be normalized to [0,1] (bool)
    # - resize_shape: Tuple for new image shape (int-tuple)
    #########################################################################################
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize_shape is not None:
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
    if normalize:
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - img.min()) / img.max()
    else:
        img = np.array(img, dtype=np.float32)
    return img


def save_img(filename, img, normalized=True):
    #########################################################################################
    # Saves numpy matrices.
    # Parameters:
    # - filename: Save-path for image (String)
    # - img: Image matrix to be saved (numpy, float/uint8)
    # - normalized: True if img is normalized to [0,1] (bool)
    #########################################################################################
    if normalized:
        img = (img*255.0).astype(np.uint8)
    else:
        img = (img)
    cv2.imwrite(filename, img)


def gaussian_mask_1d(size, center, sigma=1.0, scale=1.0):
    #########################################################################################
    # Generates 1-D Gaussian Mask.
    # Parameters:
    # - size: Size of mask. Should be odd integer for even padding (int)
    # - center: Center/Mean of Gaussian function, give desired center index (int)
    # - sigma: Variance of Gaussian function (float)
    # - scale: Further scales function as a linear multiplier (float)
    #########################################################################################
    assert size % 2 != 0, "Please assign odd numbered filter size. Image can't be evenly padded otherwise"
    mask = []
    for i in range(size):
        mask.append(i)
    mask = np.array(mask) - center
    gmask = np.exp(- (mask*mask)/(2 * sigma * sigma)) / \
        (sigma * math.sqrt(2 * math.pi))

    gmask_ddx = - gmask * 2 * mask / (2 * sigma * sigma)
    return gmask * scale, gmask_ddx * scale


def convolve(array, kernel):
    #########################################################################################
    # Convolves 1-D kernel with 1-D array.
    # Parameters:
    # - array: Array to be convolved (numpy, float)
    # - kernel: Convolution Kernel (numpy, float)
    #########################################################################################
    output = np.zeros_like(array)
    array = array.tolist()
    size = kernel.shape[0]
    if size != 0:
        pad = [0 for _ in range(size // 2)]
        array = pad + array + pad
    array = np.array(array)
    for i in range(array.shape[0] + 1 - kernel.shape[0]):
        inner_product = kernel * array[i:i+kernel.shape[0]]
        output[i] = inner_product.sum()
    return output


def directional_convolve(img, mask, direction='both'):
    #########################################################################################
    # Convolves 1-D kernel with 2-D Image matrix in specified direction.
    # Parameters:
    # - img: Image to be blurred (numpy, float)
    # - mask: 1-D Mask (numpy, float)
    # - direction: Direction of convolution (String, 'x' or 'y' or 'both')
    #########################################################################################
    assert direction in ['x', 'y', 'both'], "Invalid direction"

    Ix, Iy = [], []

    # Convolution in x-direction
    if direction == 'x' or direction == 'both':
        for i in range(img.shape[0]):
            conv = convolve(img[i, :], mask)
            Ix.append(np.array(conv))
        Ix = np.array(Ix)

    # Convolution in y-direction
    if direction == 'y' or direction == 'both':
        for i in range(img.shape[1]):
            conv = convolve(img[:, i], mask)
            Iy.append(np.array(conv))
        Iy = np.transpose(np.array(Iy))

    if direction == 'both':
        return Ix, Iy
    elif direction == 'x':
        return Ix
    elif direction == 'y':
        return Iy


def polarise(Ix, Iy, renormalize=False, degrees=True):
    #########################################################################################
    # Converts Ix and Iy pixels to polar format from rectilinear parameters
    # Parameters:
    # - Ix, Iy: Blurred images in x and y directions respectively (numpy, float)
    # - renormalize: True if normalize magnitude matrix (bool)
    # - degrees: True if return angle in degrees else radians (bool)
    #########################################################################################

    # calculate magnitudes
    M = Ix**2 + Iy**2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i,j] = math.sqrt(M[i,j])

    # calculate angles
    theta = np.zeros_like(M)
    for i in range(Ix.shape[0]):
        for j in range(Ix.shape[1]):
            if degrees:
                theta[i, j] = math.atan2(Iy[i, j], Ix[i, j]) * 180/math.pi
            else:
                theta[i, j] = math.atan2(Iy[i, j], Ix[i, j])

    if renormalize:
        M = (M - 0) / M.max()
    return M, theta


def non_max_suppression(M, theta):
    #########################################################################################
    # Inplace implementation of Non-Maximum Suppression Algorithm (NMS)
    # Parameters:
    # - M: Magnitude map M(x,y) (numpy, float)
    # - theta: Angle map theta(x,y) (numpy, float)
    #########################################################################################

    # Discretizing angles to closest multiples of 45 degrees
    theta_D = np.round(theta/45.0) * 45.0

    # NMS Step
    for i in range(M.shape[0]):
        if i != 0 and i != M.shape[0]-1:

            for j in range(M.shape[1]):
                if j != 0 and j != M.shape[1]-1:

                    if theta_D[i, j] == 0.0 or theta_D[i, j] == -180.0 or theta_D[i, j] == 180.0:
                        if M[i, j] < M[i, j+1] or M[i, j] < M[i, j-1]:
                            M[i, j] = 0
                    if theta_D[i, j] == 45.0 or theta_D[i, j] == -135.0:
                        if M[i, j] < M[i+1, j-1] or M[i, j] < M[i-1, j+1]:
                            M[i, j] = 0
                    if theta_D[i, j] == 90.0 or theta_D[i, j] == -90.0:
                        if M[i, j] < M[i-1, j] or M[i, j] < M[i+1, j]:
                            M[i, j] = 0
                    if theta_D[i, j] == 135.0 or theta_D[i, j] == -45.0:
                        if M[i, j] < M[i+1, j+1] or M[i, j] < M[i-1, j-1]:
                            M[i, j] = 0
    return M


def hysteresis_thresh(M, low=0.1, mid=0.3, high=0.35):
    #########################################################################################
    # Implements double thresholds on computed NMS map
    # Parameters:
    # - M: Magnitude map M(x,y) (numpy, float)
    # - low: Lower threshold (float)
    # - mid: Middle layer value (float)
    # - high: Higher threshold (float)
    #########################################################################################
    M[M < low] = 0.0
    M[M > high] = 1.0
    M[np.logical_and(M >= low, M <= high)] = mid
    return M


def connected_comps(M_thresh, conn=8):
    #########################################################################################
    # Connects Isolated pixel clusters to generate final edge-map (POST_PROCESSING_STEP)
    # Parameters:
    # - M_thresh: Double thresholded magnitude map M(x,y) (numpy, float)
    # - conn: Connectivity parameter for cv2 library function of connected components
    #########################################################################################
    M_thresh = np.array(M_thresh*255.0, dtype=np.uint8)
    M_check = M_thresh.copy()
    M_check[M_check != 255] = 0
    _, labels = cv2.connectedComponents(M_thresh, connectivity=conn)
    high_list = set()

    # Determine high clusters
    for i in range(M_thresh.shape[0]):
        for j in range(M_thresh.shape[1]):
            if M_check[i, j] == 255:
                high_list.add(labels[i, j])

    output = np.zeros_like(M_thresh)

    # Suppress non-connected clusters
    for i in range(M_thresh.shape[0]):
        for j in range(M_thresh.shape[1]):
            if M_thresh[i, j] != 255.0 and M_thresh[i, j] != 0.0 and labels[i, j] in high_list:
                output[i, j] = 1.0
            else:
                output[i, j] = 0.0

    output = np.array(output * 255.0, dtype=np.uint8)
    return output


def evaluate(input, target, type="l1"):
    #########################################################################################
    # Evaluates edge-map computation wrt ground truths
    # Parameters:
    # - input: Main output to be compared (numpy, float)
    # - target: Ground truth for comparison (numpy, float)
    # - type: Type of evaluation, L-1 norm or L-2 norm (String, 'l1' or 'l2')
    #########################################################################################
    assert type in ["l1", "l2"], "Can only evaluate L1 or L2 norm"
    if type == "l1":
        score = np.abs(input - target)
        score = score.sum()
    else:
        score = (input - target) ** 2
        score = score.sum()
        score = math.sqrt(score)

    return score
