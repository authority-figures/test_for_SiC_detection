import cv2
import numpy as np
from matplotlib import pyplot as plt
from img_process import plt_plot

# 1. 加载二值化图像
img_path = '../data/img/25_glass005_0040.jpg'  # 请确保路径正确
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"无法找到或打开图像文件: {img_path}")

# 2. 预处理
# 2.1 噪声去除（开运算）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

# 2.2 填充孔洞（闭运算）
img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel, iterations=2)


ret,img_data=cv2.threshold(img_close, 127, 255, cv2.THRESH_BINARY);
# 3. 连通区域分析
# 3.1 反转图像：在连通区域分析中，白色为前景
img_inv = cv2.bitwise_not(img_data)


# 3.2 确保反转后的图像是单通道8位图像
if img_inv.ndim != 2:
    raise ValueError("反转后的图像不是单通道的。请检查图像处理步骤。")
if img_inv.dtype != np.uint8:
    img_inv = img_inv.astype(np.uint8)
# 3.2 查找连通区域
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_inv, connectivity=8)

print(f"检测到 {num_labels - 1} 个斑块。")  # 减去背景

# 4. 过滤与分离
# 4.1 设置面积阈值，过滤小斑块（根据具体情况调整）
min_area = 50  # 最小面积阈值
max_area = 5000  # 最大面积阈值（可选）

# 4.2 创建一个空白图像用于存储过滤后的斑块
filtered_labels = np.zeros_like(labels)

for i in range(1, num_labels):  # 从1开始，0是背景
    area = stats[i, cv2.CC_STAT_AREA]
    # if min_area <= area <= max_area:
    #     # 将符合条件的斑块标记为255
    #     filtered_labels[labels == i] = 255
    filtered_labels[labels == i] = 255

# 4.3 确保 `filtered_labels` 是单通道8位图像
if filtered_labels.ndim != 2:
    raise ValueError("filtered_labels 不是单通道图像。")
if filtered_labels.dtype != np.uint8:
    filtered_labels = filtered_labels.astype(np.uint8)

plt_plot([img, img_data, img_inv, filtered_labels])
plt.show()

# 4.3 查找过滤后的斑块的轮廓
contours, hierarchy = cv2.findContours(filtered_labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"过滤后保留 {len(contours)} 个斑块。")

# 5. 可视化与输出
# 5.1 在原始图像上绘制轮廓
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for idx, contour in enumerate(contours):
    # 绘制轮廓
    cv2.drawContours(img_color, [contour], -1, (0, 0, 255), 2)  # 红色轮廓

    # 可选：绘制边界框
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色边界框

    # 可选：标注斑块编号
    cv2.putText(img_color, f"{idx+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1, cv2.LINE_AA)

# 5.2 显示结果
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("分离后的碳化硅斑块")
plt.axis('off')
plt.show()

# 5.3 （可选）保存分离后的斑块图像
# cv2.imwrite('separated_sic_blocks.jpg', img_color)
