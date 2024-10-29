
import cv2
import numpy as np


class Visualization:
    def __init__(self, image_size=(700, 700)):
        self.image_size = image_size  # 白色背景图像的大小
        self.tool_mask = cv2.imread(r'F:\python\object_detection\test\tracking\polygon_mask.png', cv2.IMREAD_UNCHANGED)

    def create_blank_image(self):
        """创建一个白色背景的空图像"""
        return np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 255

    def show_single_history_entry(self, history_entry):
        """显示单个历史记录条目"""
        img = self.create_blank_image()

        # 获取历史记录条目的信息
        bbox = history_entry['bbox']
        center = history_entry['center']
        contour = history_entry['contour']

        # 绘制边界框
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色边界框

        # 绘制中心点
        cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)  # 绿色中心点

        # 绘制轮廓
        contour = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)  # 红色轮廓

        # 显示图像
        cv2.imshow('Single History Entry', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_block_trajectory(self, block_data):
        """显示block的历史轨迹，标记越线和消失位置"""
        img = self.create_blank_image()

        history_data = block_data['block_history_data']
        when_crossed_state = block_data['when_crossed_state']
        when_disappeared_state = block_data['when_disappeared_state'] if block_data['when_disappeared_state'] is not None else history_data[-1]

        # 获取中心点轨迹
        points = [entry['center'] for entry in history_data]

        # 绘制轨迹线
        for i in range(1, len(points)):
            pt1 = (int(points[ i -1][0]), int(points[ i -1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)  # 黄色轨迹线

        # 标记越线时的轮廓
        if when_crossed_state:
            self._draw_special_state(img, when_crossed_state, color=(0, 0, 255), text="Crossed")

        # 标记消失时的轮廓
        if when_disappeared_state:
            self._draw_special_state(img, when_disappeared_state, color=(255, 0, 0), text="Disappeared")


        # 显示图像
        cv2.imshow(f'Block Trajectory - ID: {block_data["show_id"]}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_block_trajectory_dynamic(self, blocks_data, delay=500):
        """动态显示每个block的轨迹，每个block展示半秒"""
        window_name = 'Block Trajectory'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 100)  # 将窗口移动到屏幕位置 (100, 100)
        cv2.resizeWindow(window_name, 800, 600)  # 将窗口大小设置为 800x600

        for block_data in blocks_data:
            img = self.create_blank_image()

            history_data = block_data['block_history_data']
            when_crossed_state = block_data['when_crossed_state']
            when_disappeared_state = block_data['when_disappeared_state'] if block_data['when_disappeared_state'] is not None else history_data[-1]

            # 获取中心点轨迹
            points = [entry['center'] for entry in history_data]

            # 绘制轨迹线
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(img, pt1, pt2, (0, 255, 255), 2)  # 黄色轨迹线

            # 标记越线时的轮廓
            if when_crossed_state:
                self._draw_special_state(img, when_crossed_state, color=(0, 0, 255), text="Crossed")

            # 标记消失时的轮廓
            if when_disappeared_state:
                self._draw_special_state(img, when_disappeared_state, color=(255, 0, 0), text="Disappeared")
                bbox = when_disappeared_state['bbox']
                x, y, w, h = bbox
                rect_color = { 'cut_in':(255, 128, 0),'cut_out':(0, 255, 128),'cutting':(128, 0, 255),'normal':(0, 0, 0)}[when_disappeared_state['classification']]
                cv2.putText(img, when_disappeared_state['classification'], (x, y +h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            rect_color, 2)

            cv2.line(img, (0, 100), (700, 100), (0, 255, 0), 2)  # 添加检测线
            cv2.line(img, (0, 350), (700, 350), (0, 0, 255), 2)  # 添加检测线

            # 调用 display_tool 函数，在图像上绘制车刀
            self.display_tool(img, None, (128, 128, 128))

            # 在图像上显示您需要的文字信息
            text = f'Block ID: {block_data["show_id"]}'
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # # 显示图像并保持delay毫秒
            # cv2.imshow(f'Block Trajectory - ID: {block_data["show_id"]}', img)
            # cv2.waitKey(delay)  # 等待 delay 毫秒
            # cv2.destroyAllWindows()  # 关闭当前显示的图像窗口

            # 显示图像并保持 delay 毫秒
            cv2.imshow(window_name, img)
            key = cv2.waitKey(delay)  # 等待 delay 毫秒
            if key == 27:  # 按下 ESC 键退出
                break
        # 循环结束后再销毁窗口
        cv2.destroyAllWindows()
    def display_tool(self, img, mask_image=None, color=(0, 255, 0)):
        """
        在图像上绘制车刀的多边形遮罩。

        参数：
        - img: 要绘制的图像。
        - polygon_mask: 车刀的多边形遮罩，顶点列表，例如 [(x1, y1), (x2, y2), ...]。
        - color: 绘制多边形的颜色，默认为绿色 (0, 255, 0)。
        """
        # 确保多边形遮罩是numpy数组，并具有正确的形状
        if mask_image is None:
            mask_image = self.tool_mask
            # 确保遮罩图像与目标图像大小一致
            if mask_image.shape[:2] != img.shape[:2]:
                # 如果尺寸不一致，可以根据需要调整遮罩图像的尺寸
                mask_image = cv2.resize(mask_image, (img.shape[1], img.shape[0]))

        # 创建一个彩色的遮罩，颜色为指定的 color
        color_mask = np.zeros_like(img)
        color_mask[mask_image > 0] = color  # 将遮罩区域设置为指定颜色

        # 创建掩码，只在遮罩区域为 True
        mask = mask_image > 0

        # 将彩色遮罩叠加到目标图像上，仅在遮罩区域
        alpha = 0.5  # 透明度，可根据需要调整
        img[mask] = cv2.addWeighted(color_mask[mask], alpha, img[mask], 1 - alpha, 0)


    def _draw_special_state(self, img, state, color, text):
        """在图像上绘制特定状态（如越线或消失）的轮廓和标记"""
        bbox = state['bbox']
        center = state['center']
        contour = state['contour']

        # 绘制边界框
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # 绘制轮廓
        contour = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(img, [contour], -1, color, 2)

        # 标记状态
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
