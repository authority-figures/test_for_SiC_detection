import cv2
from test.processer import Processer
import numpy as np

processer = Processer('../../data/img/25_glass005_0080.jpg')
img_path = '../../data/img/25_glass005_0080.jpg'
ori_image = cv2.imread(img_path)

image = processer.create_mask_with_options(ori_image, 495, 0, 300, 200, outer_color=255,
                                                            inverse=False)  # 得到的结果已经是灰度图

image = cv2.bitwise_not(image)
ret,image = cv2.threshold(image , 110, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)    # 开运算出去噪点
# cv2.imshow('opening',opening)

# 获取前景
sure_fg = cv2.erode(opening, kernel, iterations=2)  # sure foreground area
sure_fg = np.uint8(sure_fg)
# cv2.imshow('sure_fg',sure_fg)

# 获取背景
sure_bg = cv2.dilate(opening, kernel, iterations=3)
sure_bg = np.uint8(sure_bg)
# cv2.imshow('sure_bg',sure_bg)

unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area
cv2.imshow('unknown',unknown)



# Perform the distance transform algorithm
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# Normalize the distance image for range = {0.0, 1.0}
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
# Finding sure foreground area
ret, sure_fg = cv2.threshold(dist_transform, 0.15*dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('distance transform algorithm sure_fg',sure_fg)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers_copy = markers.copy()
markers_copy[markers==0] = 150  # 灰色表示背景
markers_copy[markers==1] = 0    # 黑色表示背景
markers_copy[markers>1] = 255   # 白色表示前景

markers_copy = np.uint8(markers_copy)
cv2.imshow('markers',markers_copy ,)


# 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
markers = cv2.watershed(ori_image, markers)
ori_image[markers==-1] = [0,0,255]  # 将边界标记为红色

# 排除背景（标签为1），计数前景区域
object_count = len(np.unique(markers)) - 2  # -2 是因为我们不计算背景和分水岭线
print(f"Number of objects: {object_count}")

# 显示处理后的图像
cv2.imshow('Binary Image with Different Thresholds',ori_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
