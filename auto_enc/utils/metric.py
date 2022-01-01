import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def calc_ssim(gts, preds):
	metric = 0.5*(1+np.array([ssim(gt.transpose(1,2,0), pred.transpose(1,2,0), multichannel=True, data_range=1) for (gt,pred) in zip(gts, preds)]))
	return metric


def test():
	img_path = '../source.png'

	img = Image.open(img_path).convert('RGB')
	img = np.array(img)

	img_path2 = '../test.png'

	img2 = Image.open(img_path2).convert('RGB')
	img2 = np.array(img2)

	metric = calc_ssim(img, img2)
	print(metric)

