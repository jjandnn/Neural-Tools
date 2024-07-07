# Standalone script by github.com/htoyryla, and github.com/ProGamerGov
# 2024 fixed by github.com/jjandnn

import numpy as np
import argparse
import imageio.v2 as imageio
from skimage import io, transform, img_as_float
from skimage.io import imread, imsave
from PIL import Image
from numpy import eye 

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target_image', type=str, help="The image you are transferring color to. Ex: target.png", required=True)
parser.add_argument('-s', '--source_image', type=str, help="The image you are transferring color from. Ex: source.png", required=True)
parser.add_argument('-o', '--output_image', default='output.png', help="The name of your output image. Ex: output.png", type=str)
parser.add_argument('-m', '--mode', default='pca', help="The color transfer mode. Options are pca, chol, or sym.", type=str)
parser.add_argument('-e', '--eps', default='1e-5', help="Your epsilon value in scientific notation or normal notation. Ex: 1e-5 or 0.00001", type=float)
parser.parse_args()
args = parser.parse_args()

Image.MAX_IMAGE_PIXELS = 1000000000  # Support gigapixel images

def main():   
    target_img = imageio.imread(args.target_image, pilmode="RGB").astype(np.float32) / 256
    source_img = imageio.imread(args.source_image, pilmode="RGB").astype(np.float32) / 256

    output_img = match_color(target_img, source_img, mode=args.mode, eps=args.eps)

    # 确保输出图像的值在 0 到 255 之间，并转换为 uint8 类型
    output_img = np.clip(output_img * 256, 0, 255).astype(np.uint8)
    imsave(args.output_image, output_img)

def match_color(target_img, source_img, mode='pca', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c) and float in [0,1].
    Modes are chol, pca or sym for different choices of basis.
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2, 0, 1).reshape(3, -1)
    Ct = t.dot(t.T) / t.shape[1] + eps * eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2, 0, 1).reshape(3, -1)
    Cs = s.dot(s.T) / s.shape[1] + eps * eye(s.shape[0])
    
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    elif mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    elif mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
        
    matched_img = ts.reshape(*target_img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += mu_s
    matched_img[matched_img > 1] = 1
    matched_img[matched_img < 0] = 0
    return matched_img

if __name__ == "__main__":
    main()
