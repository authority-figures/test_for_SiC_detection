import cv2
import numpy as np
from matplotlib import pyplot as plt
from img_process import plt_plot

def separate_sic_blocks_with_morphology(img_path, min_area=100, max_area=5000):
    # 1. 加载二值化图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法找到或打开图像文件: {img_path}")

    # # 2. 预处理
    # # 2.1 噪声去除（开运算）
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    #
    # # 2.2 填充孔洞（闭运算）
    # img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel, iterations=2)
    # img_close = cv2.bitwise_not(img_close)

    ret, img_close = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY);
    img_inv = cv2.bitwise_not(img_close)
    # 3. 断开狭窄连接
    # 3.1 定义用于腐蚀的结构元素，大小取决于狭窄连接的宽度
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_eroded = cv2.erode(img_inv , erosion_kernel, iterations=1)

    # 3.2 膨胀恢复主要结构，但狭窄连接已被断开
    img_dilated = cv2.dilate(img_eroded, erosion_kernel, iterations=1)


    # 4. 连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilated, connectivity=8)

    # 5. 过滤与分离
    filtered_labels = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered_labels[labels == i] = 255
        # filtered_labels[labels == i] = 255
    plt_plot([img, img_close, img_eroded, img_dilated,filtered_labels])
    plt.show()

    # 6. 查找轮廓
    contours, hierarchy = cv2.findContours(filtered_labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 可视化与输出
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if x>500 and y<130:
            cv2.drawContours(img_color, [contour], -1, (0, 0, 255), 2)  # 红色轮廓

            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色边界框
            cv2.putText(img_color, f"{idx + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("分离后的碳化硅斑块（形态学操作）")
    plt.axis('off')


    return img_color, contours



if __name__ == '__main__':

    # 示例调用
    result_img, sic_contours = separate_sic_blocks_with_morphology('../data/img/25_glass005_0040.jpg')
    for i in range(10):
        separate_sic_blocks_with_morphology(f'../data/img/25_glass005_{i+100:04}.jpg',min_area=200)
        plt.show()
        plt.close()
