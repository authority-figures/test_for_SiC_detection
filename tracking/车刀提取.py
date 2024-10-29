import cv2
import numpy as np

# 全局变量
drawing = False         # 是否在绘制中
mode = None             # 当前模式，1：前景，2：背景
ix, iy = -1, -1         # 鼠标点击的位置
marker_image = None     # 标记矩阵
img = None              # 原始图像
img_copy = None         # 用于绘制的图像副本
polygon_points = []     # 存储多边形顶点的列表
last_command = None     # 记录上一次按键

def mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # 在窗口标题中显示鼠标坐标
        coords = f"Mouse coordinates: ({x}, {y})"
        cv2.setWindowTitle(param, coords)
# 鼠标回调函数
def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, mode, img_copy, marker_image

    if mode == 'm':
        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加多边形顶点
            polygon_points.append((x, y))
            # 在图像上绘制小圆点
            cv2.circle(img_copy, (x, y), 3, (0, 255, 255), -1)
            if len(polygon_points) > 1:
                # 绘制线段连接顶点
                cv2.line(img_copy, polygon_points[-2], polygon_points[-1], (0, 255, 255), 2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键点击完成多边形
            if len(polygon_points) > 2:
                # 闭合多边形
                cv2.line(img_copy, polygon_points[-1], polygon_points[0], (0, 255, 255), 2)
                # 创建遮罩
                mask = np.zeros(marker_image.shape, dtype=np.uint8)
                pts = np.array(polygon_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 3)  # 使用标记值3表示多边形遮罩
                # 更新标记矩阵
                marker_image[mask == 3] = 3
            polygon_points.clear()
    else:

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if mode == 1:
                    color = (0, 255, 0)  # 前景用绿色表示
                    cv2.line(img_copy, (ix, iy), (x, y), color, 5)
                    cv2.line(marker_image, (ix, iy), (x, y), 255, 5)  # 前景标记为255
                    ix, iy = x, y
                elif mode == 2:
                    color = (0, 0, 255)  # 背景用红色表示
                    cv2.line(img_copy, (ix, iy), (x, y), color, 5)
                    cv2.line(marker_image, (ix, iy), (x, y), 128, 5)  # 背景标记为128
                    ix, iy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == 1:
                color = (0, 255, 0)
                cv2.line(img_copy, (ix, iy), (x, y), color, 5)
                cv2.line(marker_image, (ix, iy), (x, y), 255, 5)
            elif mode == 2:
                color = (0, 0, 255)
                cv2.line(img_copy, (ix, iy), (x, y), color, 5)
                cv2.line(marker_image, (ix, iy), (x, y), 128, 5)

# 主程序
def main():
    global img, img_copy, marker_image, mode

    # 读取图像
    img = cv2.imread(r'F:\python\object_detection\data\imgs\14-glass005-1\14_glass005_1_0001.jpg')
    # img = cv2.bitwise_not(img)
    # img = cv2.resize(img, (800, 600))
    img_copy = img.copy()

    # 初始化标记矩阵
    marker_image = np.zeros(img.shape[:2], dtype=np.uint8)
    window_name = 'Image'
    # 创建窗口并设置回调函数
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_mask)

    # 设置鼠标移动事件的回调函数
    cv2.setMouseCallback(window_name, mouse_move, window_name)

    print("按'1'键：绘制前景（车刀）掩码")
    print("按'2'键：绘制背景掩码")
    print("按'm'键：创建多边形遮罩（左键添加顶点，右键完成）")
    print("按's'键：保存当前绘制的掩码")
    print("按'c'键：执行分水岭算法")
    print("按'r'键：重置")
    print("按'q'键：退出程序")

    while True:
        cv2.imshow(window_name, img_copy)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('1'):
            mode = 1
            print("进入前景绘制模式")
        elif k == ord('2'):
            mode = 2
            print("进入背景绘制模式")
        elif k == ord('m'):
            mode = 'm'
            last_command = 'm'
            print("模式：创建多边形遮罩")
            # 重置多边形顶点
            polygon_points.clear()
        elif k == ord('s'):
            if last_command == '1':
                # 保存前景掩码
                cv2.imwrite('foreground_mask.png', (marker_image == 1).astype(np.uint8) * 255)
                print("前景掩码已保存为'foreground_mask.png'")
            elif last_command == '2':
                # 保存背景掩码
                cv2.imwrite('background_mask.png', (marker_image == 2).astype(np.uint8) * 255)
                print("背景掩码已保存为'background_mask.png'")
            elif last_command == 'm':
                # 保存多边形遮罩
                cv2.imwrite('polygon_mask.png', (marker_image == 3).astype(np.uint8) * 255)
                print("多边形遮罩已保存为'polygon_mask.png'")
            else:
                print("没有可保存的命令结果。")
        elif k == ord('c'):
            print("执行分水岭算法...")
            # 创建 markers
            markers = np.zeros_like(marker_image, dtype=np.int32)
            markers[marker_image == 255] = 1  # 前景标记为1
            markers[marker_image == 128] = 2  # 背景标记为2

            # 执行分水岭算法
            cv2.watershed(img, markers)

            # 创建结果图像
            img_result = np.zeros_like(img)
            # 将前景区域保留原始图像内容
            img_result[markers == 1] = img[markers == 1]
            # 将背景区域填充为纯色，例如白色
            img_result[markers != 1] = [0, 0, 0]
            img_result[markers == -1] = [0, 0, 255]  # 边界标记为红色

            # 保存并显示结果
            cv2.imshow('result', img_result)
            cv2.imwrite('segmentation_result.png', img_result)
            print("分割结果已显示并保存为'segmentation_result.png'")
        elif k == ord('r'):
            img_copy = img.copy()
            marker_image = np.zeros(img.shape[:2], dtype=np.uint8)
            print("已重置")
        elif k == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
