from utils import *
import os.path as osp
import os
import argparse
import json


def canny_edge_detect(img_path, sigma=1.0, mask_size=5, thresh_low=0.1, thresh_high=0.3, mid=0.4, conn=8, save_path=None):
    #########################################################################################
    # Canny Edge Filter Driver Function
    # Parameters:
    # - img_path: Path for image (String)
    # - sigma: Variance of Gaussian Mask (float)
    # - mask_size: Mask size of Gaussian Mask (int)
    # - thresh_low: Lower Threshold for Hysteresis thresholding (float)
    # - thresh_high: Higher Threshold for Hysteresis thresholding (float)
    # - mid: Middle zone value upon thresholding (float)
    # - conn: connectivity parameter for cv2 connected components (int)
    # - save_path: Path to an output directory to dump intermediate results (String)
    #########################################################################################

    # Load image
    img = load_img(img_path, normalize=True)

    # Generate Gaussian Mask
    gmask, gmask_ddx = gaussian_mask_1d(
        size=mask_size, center=mask_size//2, sigma=sigma)

    # Generate Ix, Iy, Ixx, Iyy
    Ix, Iy = directional_convolve(img, gmask)
    Ixx, Iyy = directional_convolve(
        Ix, gmask_ddx, direction='x'), directional_convolve(Iy, gmask_ddx, direction='y')

    # Generate M, theta by polar conversion
    M, theta = polarise(Ixx, Iyy, renormalize=True)

    # Save Image, Ix, Iy, Ixx, Iyy, M
    if save_path is not None:
        save_img(osp.join(save_path, "Image.jpg"), img)
        save_img(osp.join(save_path, "Ix.jpg"), Ix)
        save_img(osp.join(save_path, "Iy.jpg"), Iy)
        save_img(osp.join(save_path, "Ixx.jpg"), Ixx)
        save_img(osp.join(save_path, "Iyy.jpg"), Iyy)
        save_img(osp.join(save_path, "M.jpg"), M)

    # Compute Non-max Suppression and save M_nms
    M_nms = non_max_suppression(M, theta)
    if save_path is not None:
        save_img(osp.join(save_path, "M_nms.jpg"), M_nms)

    # Hysteresis thresholding and save M_thresh
    M_thresh = hysteresis_thresh(M_nms, low=thresh_low, high=thresh_high)
    if save_path is not None:
        save_img(osp.join(save_path, "M_thresh.jpg"), M_thresh)

    # Connected Components computation and save M_cc
    M_cc = connected_comps(M_thresh, conn)
    if save_path is not None:
        save_img(osp.join(save_path, "M_cc.jpg"), M_cc, normalized=False)

    return M_cc


def parse_args():
    #########################################################################################
    #  Argument parsing for driver program
    #########################################################################################

    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument('--run_name', default="exp", type=str)
    parser.add_argument('--input_img', default="assets/119082.jpg", type=str)
    parser.add_argument('--GT', default="assets/119082_gt.jpg", type=str)

    # Gaussian Mask details
    parser.add_argument('--size', default=5, type=int)
    parser.add_argument('--sigma', default=1.0, type=float)

    # Thresh values
    parser.add_argument('--thresh_low', default=0.1, type=float)
    parser.add_argument('--thresh_mid', default=0.4, type=float)
    parser.add_argument('--thresh_high', default=0.3, type=float)
    parser.add_argument('--thresh_conn', default=8, type=int)

    args = parser.parse_args()
    return args


def launch():
    #########################################################################################
    #  Driver Program
    #########################################################################################
    args = parse_args()

    # Save run parameters
    configs = vars(args)
    os.makedirs(osp.join("outputs", args.run_name), exist_ok=True)
    file = open(os.path.join("outputs", args.run_name, 'configs.json'), 'w')
    file.write(json.dumps(configs))
    file.close()

    # Run Canny Edge filter
    cne = canny_edge_detect(img_path=args.input_img, sigma=args.sigma, mask_size=args.size,
                            thresh_low=args.thresh_low, thresh_high=args.thresh_high, mid=args.thresh_mid,
                            conn=args.thresh_conn, save_path=osp.join("outputs", args.run_name))
    # use cne as output edge maps for further tasks...


if __name__ == '__main__':
    launch()
