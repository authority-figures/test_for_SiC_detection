import cv2
import numpy as np

image = cv2.imread(r'F:\python\object_detection\data\img\25_glass005_0001.jpg', cv2.IMREAD_UNCHANGED)

ori_image = image


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers_copy = markers.copy()
markers_copy[markers == 0] = 150  # 灰色表示背景
markers_copy[markers == 1] = 0  # 黑色表示背景
markers_copy[markers > 1] = 255  # 白色表示前景

markers_copy = np.uint8(markers_copy)

final_img = ori_image.copy()
# 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
markers = cv2.watershed(final_img, markers)
final_img[markers == -1] = [255, 0, 0]  # 将边界标记为红色