import cv2
import numpy as np
from matplotlib import pyplot as plt
from processer import Processer

img_path = '../data/img/25_glass005_0081.jpg'
ori_img_data = cv2.imread(img_path, )
processer = Processer(ori_img_data)
masked_img = processer.create_mask_with_options(ori_img_data, 495, 0, 300, 200,outer_color=255, inverse=False)
img_inv = cv2.bitwise_not(masked_img )
img_bin = cv2.threshold(img_inv , 110, 255, cv2.THRESH_BINARY)
# 距离变换
dist_transform = cv2.distanceTransform(img_bin, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
kernel = np.ones((3,3), np.uint8)  # 定义一个小的卷积核
# 找到背景
sure_bg = cv2.dilate(img_inv, kernel, iterations=3)

# 分水岭算法
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)

# 应用分水岭
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR), markers)

processer.plt_plot_with_rectangle([img_inv, sure_fg, sure_bg, unknown, markers])
plt.imshow(markers, cmap='jet')
plt.show()