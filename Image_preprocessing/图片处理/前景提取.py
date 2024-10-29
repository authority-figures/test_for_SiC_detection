import cv2
import numpy as np

# 加载 PNG 图像（包括 Alpha 通道）
image = cv2.imread(r'img1.png', cv2.IMREAD_UNCHANGED)

# 设置缩小比例，例如缩小 50%
scale_percent = 50  # 设置为你想缩小的百分比

# 计算新的宽度和高度
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# 使用 cv2.resize 函数调整图像大小
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
image = resized_image
# 检查图像是否具有 Alpha 通道
if image.shape[2] == 4:
    # 分离图像的 RGBA 通道
    b, g, r, alpha = cv2.split(image)

    # 创建一个三通道的 BGR 图像
    bgr_image = cv2.merge([b, g, r])

    # 创建背景掩码，Alpha 通道为0的像素将被视为背景
    mask = alpha == 0

    # 将背景设置为白色（也可以改为其他颜色）
    bgr_image[mask] = [255, 255, 255]

    # 转换为灰度图像
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊减少噪点
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白的图像，用于绘制轮廓
    output_image = bgr_image.copy()

    # 使用蓝色（BGR: (255, 0, 0)）绘制轮廓
    cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('Original Image', bgr_image)
    cv2.imshow('Contours', output_image)

    # 保存结果
    cv2.imwrite('output_contours.png', output_image)

    # 等待按键，然后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("该图像不包含 Alpha 通道。请提供具有透明背景的 PNG 图像。")
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊减少噪点
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白的图像，用于绘制轮廓
    output_image = image.copy()

    # 使用蓝色（BGR: (255, 0, 0)）绘制轮廓
    cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Contours', output_image)

    # 保存结果
    cv2.imwrite('output_contours.jpg', output_image)

    # 等待按键，然后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
