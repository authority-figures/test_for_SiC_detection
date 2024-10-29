import cv2
import numpy as np

# 读取彩色图像
img_path = '../data/img/25_glass005_0080.jpg'
ori_img_data = cv2.imread(img_path)

# 分离图像的三个通道
b_channel, g_channel, r_channel = cv2.split(ori_img_data)

# 对每个通道应用不同的二值化阈值
ret_b, b_binary = cv2.threshold(b_channel, 35, 255, cv2.THRESH_BINARY)  # 对蓝色通道
ret_g, g_binary = cv2.threshold(g_channel, 35, 255, cv2.THRESH_BINARY)  # 对绿色通道
ret_r, r_binary = cv2.threshold(r_channel, 25, 255, cv2.THRESH_BINARY)  # 对红色通道

# 合并处理后的三个通道
merged_img = cv2.merge([b_binary, g_binary, r_binary])

# 显示处理后的图像
cv2.imshow('Binary Image with Different Thresholds', merged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
