### Test reading images / annotations from GlaS dataset

# from skimage.io import imread
from matplotlib import pyplot as plt
# import os

# directory = "d:/Adrien/dataset/GlaS/train"
# image_files = [os.path.join(directory, f'train_{i}.bmp') for i in range(1, 86)]
# annotation_files = [os.path.join(directory, f'train_{i}_anno.bmp') for i in range(1, 86)]

# for imf,annof in zip(image_files, annotation_files):
# 	im,anno = imread(imf), imread(annof)
# 	print(im.shape, im.dtype, anno.shape, anno.dtype)
# 	plt.figure()
# 	plt.subplot(1,2,1)
# 	plt.imshow(im)
# 	plt.subplot(1,2,2)
# 	plt.imshow(anno)
# 	plt.show()
# 	break

from DataGenerator import DataGenerator

dg = DataGenerator(5, 10, "d:/Adrien/dataset/GlaS/train")

for batch_x, batch_y in dg.next_batch(1):
	print(batch_x.shape, batch_y.shape)