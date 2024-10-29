import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = '../data/img/25_glass005_0040.jpg'
ori_img_data = cv2.imread(img_path)

ret,img_data=cv2.threshold(ori_img_data, 100, 255, cv2.THRESH_BINARY)

def plt_plot(img_list=[]):
    plt.figure(figsize=(9, 6))
    for i, img in enumerate(img_list):
        plt.subplot(1, len(img_list), i+1), plt.axis('off'), plt.title(f"img-{i}")
        ax = plt.gca()
        ax.add_patch(plt.Rectangle((500, 0), 200, 150, color="red", fill=False, linewidth=1,linestyle='--'))
        ax.add_patch(plt.Rectangle((580, 300), 200, 200, color="blue", fill=False, linewidth=1, linestyle='--'))
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    pass

def plt_plot_rgb(ori_img_data):
    # Convert BGR image to RGB for correct color channel representation in matplotlib
    img_rgb = cv2.cvtColor(ori_img_data, cv2.COLOR_BGR2RGB)

    # Split the image into its three channels
    r_channel, g_channel, b_channel = cv2.split(img_rgb)

    # Plot the three channels separately
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(r_channel, cmap='Reds')
    plt.title('Red Channel')

    plt.subplot(1, 3, 2)
    plt.imshow(g_channel, cmap='Greens')
    plt.title('Green Channel')

    plt.subplot(1, 3, 3)
    plt.imshow(b_channel, cmap='Blues')
    plt.title('Blue Channel')

    plt.tight_layout()




if __name__ == '__main__':
    img_path = '../data/img/25_glass005_0081.jpg'
    ori_img_data = cv2.imread(img_path,)

    ret, img_data = cv2.threshold(ori_img_data, 25, 255, cv2.THRESH_BINARY)
    img_data = cv2.GaussianBlur(img_data, (5, 5), 0)
    edges = cv2.Canny(img_data, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓到空白图像上
    contour_img = np.zeros_like(img_data)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 创建一个副本来绘制直线
    line_img = np.copy(contour_img)

    # 绘制检测到的直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 使用绿色线绘制

    # 显示结果图像
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title('Canny Edges')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines')

    # 显示轮廓图
    # plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    # cv2.imshow('img', img_data)
    # cv2.waitKey(0)
    # plt_plot_rgb(img_data)
    # plt.imshow(img_data, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # cv2.imshow('img', img_data)

    # # 1.78：图像锐化：拉普拉斯算子 (Laplacian)
    # img = cv2.imread(img_path, flags=0)  # NASA 月球影像图
    #
    # # 使用函数 filter2D 实现 Laplace 卷积算子
    # kernLaplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplacian kernel
    # imgLaplace1 = cv2.filter2D(img, -1, kernLaplace, borderType=cv2.BORDER_REFLECT)
    #
    # # 使用 cv2.Laplacian 实现 Laplace 卷积算子
    # imgLaplace2 = cv2.Laplacian(img, -1, ksize=3)
    # imgRecovery = cv2.add(img, imgLaplace2)  # 恢复原图像
    #
    # # 二值化边缘图再卷积
    # ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    # imgLaplace3 = cv2.Laplacian(binary, cv2.CV_64F)
    # imgLaplace3 = cv2.convertScaleAbs(imgLaplace3)
    #
    # plt.figure(figsize=(9, 6))
    # plt.subplot(131), plt.axis('off'), plt.title("Original")
    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(132), plt.axis('off'), plt.title("cv.Laplacian")
    # plt.imshow(imgLaplace2, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(133), plt.axis('off'), plt.title("thresh-Laplacian")
    # plt.imshow(imgLaplace3, cmap='gray', vmin=0, vmax=255)
    # plt.tight_layout()
    # plt.show()
    #
    # pass