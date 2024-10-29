import cv2
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from processer import *

# 初始化ORB特征提取器
orb = cv2.ORB_create()

# 读取连续的两帧图像
frame1 = cv2.imread('../../data/img/25_glass005_0081.jpg')
frame2 = cv2.imread('../../data/img/25_glass005_0082.jpg')


class Block:
    def __init__(self, block_id, bbox, contour):
        self.id = block_id
        self.bbox = bbox  # (x, y, w, h)
        self.center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        self.area = bbox[2] * bbox[3]
        self.contour = contour
        self.crossed = False  # 标记是否已跨越线
        self.show_id = self.id  # 用于显示的ID

    def update(self, bbox):
        self.bbox = bbox
        self.center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        self.area = bbox[2] * bbox[3]

class TrackedBlob:
    def __init__(self, blob_id, bbox, center):
        self.id = blob_id
        self.bbox = bbox  # (x, y, w, h)
        self.center = center  # 中心点 (x, y)
        self.expected_center = center  # 期望的下一个中心位置
        self.crossed = False  # 是否已越过检测线
        self.lost_frames = 0  # 跟踪丢失的帧数

    def update_position(self, bbox, center):
        """更新斑块的位置和边界框"""
        self.bbox = bbox
        self.center = center
        self.expected_center = center  # 更新期望位置
        self.lost_frames = 0  # 恢复跟踪

    def predict_next_position(self):
        """如果斑块丢失，在下一帧预测其期望位置"""
        # 可以进一步扩展为基于运动模型预测的期望位置
        return self.expected_center

# class BlobTracker:
#     def __init__(self, detection_line_y, max_lost_frames=5):
#         self.detection_line_y = detection_line_y  # 检测线的y坐标
#         self.tracked_blobs = []  # 存储已跟踪的斑块
#         self.total_blobs_count = 0  # 记录通过检测线的斑块总数
#         self.blob_id_counter = 0  # 分配唯一ID
#         self.max_lost_frames = max_lost_frames  # 最大丢失帧数，超过此帧数不再跟踪
#
#     def add_blob(self, block):
#         """为新检测的斑块分配ID并存储"""
#         new_blob = Block(self.blob_id_counter, block.bbox, block.center)
#         self.tracked_blobs.append(new_blob)
#         self.blob_id_counter += 1
#
#     def calculate_similarity(self, tracked_blob:Block, block, shift):
#         """计算当前斑块和期望斑块位置的相似度"""
#         expected_center = (tracked_blob.center[0]+shift[0],tracked_blob.center[1]+shift[1])
#         current_center = (block.center[0], block.center[1] )
#         distance = np.linalg.norm(np.array(expected_center) - np.array(current_center))
#         size_diff = abs(tracked_blob.bbox[2] * tracked_blob.bbox[3] - block.area)
#         return distance < 50 and size_diff < 500  # 距离和大小变化是否可接受
#
#     def track_blobs(self, current_blocks, image_shift, frame):
#         """根据当前帧斑块与上一帧的位移信息进行斑块跟踪，并显示跟踪过程"""
#         new_tracked_blobs = []
#
#         # 绘制检测线
#         cv2.line(frame, (0, self.detection_line_y), (frame.shape[1], self.detection_line_y), (0, 255, 0), 2)
#
#         for block in current_blocks:
#             matched = False
#
#             # 遍历所有正在跟踪的斑块
#             for tracked_blob in self.tracked_blobs:
#                 if self.calculate_similarity(tracked_blob, block, image_shift):
#                     matched = True
#                     # 更新斑块的位置
#                     tracked_blob.update_position(block.bbox, block.center)
#                     new_tracked_blobs.append(tracked_blob)
#
#                     # 检查斑块是否越过检测线
#                     if tracked_blob.center[1] > self.detection_line_y and not tracked_blob.crossed:
#                         self.total_blobs_count += 1
#                         tracked_blob.crossed = True
#
#                     # 在图像上绘制斑块的边界框、ID 和中心点
#                     self.draw_blob(frame, tracked_blob)
#                     break
#
#             # 如果未找到匹配的斑块，添加为新斑块
#             if not matched:
#                 self.add_blob(block)
#                 self.draw_blob(frame, self.tracked_blobs[-1])
#
#         # 处理丢失帧超过阈值的斑块
#         for tracked_blob in self.tracked_blobs:
#             tracked_blob.lost_frames += 1
#             if tracked_blob.lost_frames > self.max_lost_frames or self.out_of_bounds(tracked_blob):
#                 continue  # 不再跟踪该斑块
#
#         self.tracked_blobs = [blob for blob in new_tracked_blobs if blob.lost_frames <= self.max_lost_frames]
#
#     def out_of_bounds(self, blob):
#         """检查斑块是否超出图像边界"""
#         x, y, w, h = blob.bbox
#         return x < 0 or y < 0 or x + w > 640 or y + h > 480  # 假设图像大小为640x480
#
#     def draw_blob(self, frame, blob):
#         """在图像上绘制斑块的边界框、中心点和ID"""
#         x, y, w, h = blob.bbox
#         # 绘制边界框
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         # 绘制中心点
#         cv2.circle(frame, (int(blob.center[0]), int(blob.center[1])), 5, (0, 0, 255), -1)
#         # 显示ID
#         cv2.putText(frame, f'ID: {blob.id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#
#     def get_total_blobs_count(self):
#         return self.total_blobs_count


def calculate_image_shift(frame1, frame2, roi_coords):
    """
    计算两张图像中感兴趣区域的相对平移量。

    参数：
        image1_path (str): 第一张图片的文件路径。
        image2_path (str): 第二张图片的文件路径。
        roi_coords (tuple): 感兴趣区域的坐标 (x, y, w, h)。

    返回：
        tuple: 相对平移量 (dx, dy)。
    """

    # 提取感兴趣区域（ROI）
    x, y, w, h = roi_coords
    roi1 = frame1[y:y + h, x:x + w]
    roi2 = frame2[y:y + h, x:x + w]

    # 将ROI区域转换为灰度图像
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # 使用相位相关法计算图像的相对平移量
    shift = cv2.phaseCorrelate(np.float32(gray1), np.float32(gray2))

    # 解析平移量
    dx, dy = shift[0]

    # 在原始图像上绘制感兴趣区域框（可选，可视化）
    frame1_with_roi = frame1.copy()
    frame2_with_roi = frame2.copy()
    cv2.rectangle(frame1_with_roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.rectangle(frame2_with_roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示标记了ROI的图像（用于调试，可注释掉）
    # cv2.imshow('Frame 1 with ROI', frame1_with_roi)
    # cv2.imshow('Frame 2 with ROI', frame2_with_roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return dx, dy


def track_sic_blocks(processor, start_frame, end_frame, roi_coords, min_area=100, max_area=5000,show_dynamic=True, delay=0.1):
    """
    通过连续帧跟踪碳化硅斑块并统计流过ROI的碳化硅数量。

    参数：
        processor (Processer): Processer类的实例。
        start_frame (int): 起始帧编号。
        end_frame (int): 结束帧编号。
        roi_coords (tuple): 感兴趣区域的坐标 (x, y, w, h)。
        min_area (int): 碳化硅斑块的最小面积。
        max_area (int): 碳化硅斑块的最大面积。

    返回：
        int: 通过ROI的碳化硅斑块数量。
    """
    if show_dynamic:
        plt.ion()  # 启用 Matplotlib 的交互模式
    prev_frame = None
    total_blocks_passed = 0
    prev_blocks = []

    for idx in range(start_frame, end_frame):
        # 读取当前帧
        img_path = f"../data/img/25_glass005_{idx + 1:04}.jpg"
        current_frame = cv2.imread(img_path)
        if current_frame is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 对当前帧的ROI进行碳化硅斑块检测
        masked_img = processor.create_mask_with_options(current_frame, *roi_coords, outer_color=255, inverse=False)
        separated_img,contours, bounding_boxes = processor.separate_sic_blocks_with_morphology(masked_img, min_area, max_area)

        # 统计当前帧的碳化硅斑块
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.cvtColor(separated_img, cv2.COLOR_BGR2GRAY),
                                                                        connectivity=8)
        current_blocks = [stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1] for i in range(1, num_labels)]

        if prev_frame is not None:
            # 计算帧之间的平移量
            dx, dy = calculate_image_shift(prev_frame, current_frame, roi_coords)
            print(f"帧 {idx} 到 {idx + 1} 的平移量: dx = {dx:.2f}, dy = {dy:.2f}")

            # 进行斑块的相关性分析
            matched_blocks = set()
            for current_block in current_blocks:
                current_center = (current_block[0] + current_block[2] / 2, current_block[1] + current_block[3] / 2)
                for prev_block in prev_blocks:
                    prev_center = (prev_block[0] + prev_block[2] / 2 + dx, prev_block[1] + prev_block[3] / 2 + dy)
                    dist = distance.euclidean(current_center, prev_center)
                    area_diff = abs(current_block[2] * current_block[3] - prev_block[2] * prev_block[3])

                    # 如果中心点距离和面积差异在一定范围内，则认为是同一斑块
                    if dist < 20 and area_diff < 500:
                        matched_blocks.add(tuple(current_center))
                        break

            # 计算新出现的斑块数量
            new_blocks_count = len(current_blocks) - len(matched_blocks)
            total_blocks_passed += new_blocks_count
        else:
            # 第一帧的所有斑块都计为新出现的
            total_blocks_passed += len(current_blocks)

        # 更新上一帧信息
        prev_frame = current_frame
        prev_blocks = current_blocks

        if show_dynamic:
            processor.display_single_image(separated_img, title=[f"Processed {idx + 1}"], delay=delay)

    return total_blocks_passed


def track_sic_blocks1(processor, start_frame, end_frame, roi_coords, min_area=100, max_area=5000):
    """
    通过连续帧跟踪碳化硅斑块并统计流过ROI的碳化硅数量。

    参数：
        processor (Processer): Processer类的实例。
        start_frame (int): 起始帧编号。
        end_frame (int): 结束帧编号。
        roi_coords (tuple): 感兴趣区域的坐标 (x, y, w, h)。
        min_area (int): 碳化硅斑块的最小面积。
        max_area (int): 碳化硅斑块的最大面积。

    返回：
        int: 通过ROI的碳化硅斑块数量。
    """
    prev_frame = None
    total_blocks_passed = 0
    blocks = []
    next_block_id = 1
    crossing_line_y = roi_coords[1] + roi_coords[3]  # 在ROI下方设置一条线

    for idx in range(start_frame, end_frame):
        # 读取当前帧
        img_path = f"../data/img/25_glass005_{idx + 1:04}.jpg"
        current_frame = cv2.imread(img_path)
        if current_frame is None:
            print(f"无法读取图像: {img_path}")
            continue

        # # 对当前帧的ROI进行碳化硅斑块检测
        # masked_img = processor.create_mask_with_options(current_frame, *roi_coords, outer_color=255, inverse=False)
        # separated_img, contours, bounding_boxes = processor.separate_sic_blocks_with_morphology(masked_img, min_area, max_area)

        blocks = processor.watershed_method(current_frame,mask_options=(495, 0, 300, 200,), fg_method='', ifshow=False)

        # current_blocks = []
        # for bbox, contour in zip(bounding_boxes, contours):
        #     current_blocks.append(Block(next_block_id, bbox, contour))
        #     next_block_id += 1

        current_blocks = blocks.copy()

        # 动态显示当前帧
        display_frame = current_frame.copy()
        for block in current_blocks:
            cv2.circle(display_frame, (int(block.center[0]), int(block.center[1])), 5, (0, 255, 0), -1)

        if prev_frame is not None:
            # 计算帧之间的平移量
            dx, dy = calculate_image_shift(prev_frame, current_frame, roi_coords)
            print(f"帧 {idx} 到 {idx + 1} 的平移量: dx = {dx:.2f}, dy = {dy:.2f}")

            # 进行斑块的相关性分析，判断是否为同一斑块
            for current_block in current_blocks:
                matched = False
                for prev_block in blocks:
                    prev_center_shifted = (prev_block.center[0] + dx, prev_block.center[1] + dy)
                    dist = distance.euclidean(current_block.center, prev_center_shifted)
                    area_diff = abs(current_block.area - prev_block.area)

                    # 如果中心点距离和面积差异在一定范围内，则认为是同一斑块
                    if dist < 50 and area_diff < 1000:
                        matched = True
                        current_block.id = prev_block.id
                        current_block.crossed = prev_block.crossed

                        # 更新跨越状态
                        if not current_block.crossed and prev_block.center[1] <= crossing_line_y < current_block.center[1]:
                            current_block.crossed = True
                            total_blocks_passed += 1

                        # 更新显示中的ID
                        cv2.putText(display_frame, f"ID: {current_block.id}", (int(current_block.center[0]), int(current_block.center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        break
                if not matched:
                    # 分配新的ID，只考虑从上方进入的斑块
                    if current_block.center[1] < roi_coords[1] + 20:  # 只允许从顶部新进入的斑块
                        current_block.id = next_block_id
                        next_block_id += 1
                        # 更新显示中的ID
                        cv2.putText(display_frame, f"ID: {current_block.id}", (int(current_block.center[0]), int(current_block.center[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            # 第一帧的所有斑块都计为新出现的
            for current_block in current_blocks:
                current_block.id = next_block_id
                next_block_id += 1
                cv2.putText(display_frame, f"ID: {current_block.id}", (int(current_block.center[0]), int(current_block.center[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 显示动态更新的帧
        cv2.imshow('Tracking', display_frame)
        cv2.waitKey(30)  # 设置延时以便于观察

        # 更新上一帧信息
        prev_frame = current_frame
        blocks = current_blocks

    cv2.destroyAllWindows()
    return total_blocks_passed



def process_image_sequence(image_sequence, tracker, roi_coords_for_displament,roi_for_mask=(495, 0, 300, 400,),save_video=False):
    if save_video:
        output_video_path = './video_out/output_video.mp4'
        # 获取视频帧的尺寸
        height, width = image_sequence[0].shape[:2]
        frame_rate = 30
        # 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
        video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    previous_frame = None
    unet = Unet()
    for i, frame in enumerate(image_sequence):
        # 使用水岭算法检测当前帧的斑块
        processor = Processer(frame)
        blocks = processor.watershed_method(frame, mask_options=roi_for_mask, fg_method='', ifshow=False,
                                            bg_iter=3, fg_iter=8,
                                            opening_iter=3,
                                            min_area=200,
                                            unet=unet,
                                            )

        if previous_frame is not None:
            # 计算当前帧与上一帧的位移
            image_shift = processor.calculate_image_shift(previous_frame, frame, roi_coords_for_displament)

            # 进行斑块跟踪和计数，并显示跟踪结果
            tracker.track_blobs(blocks, image_shift, frame)

        # 显示当前帧的跟踪结果

        cv2.putText(frame, f'Num SiC: {tracker.get_total_blobs_count()}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # 按下 'ESC' 键退出
            break
        if save_video:
            # 写入当前帧到视频文件
            video_writer.write(frame)

        previous_frame = frame
        # 释放 VideoWriter 对象

    if save_video:
        video_writer.release()


    print(f"总共有 {tracker.get_total_blobs_count()} 个斑块越过了检测线.")
    cv2.destroyAllWindows()



def process_image_sequence_with_kalman(image_sequence, tracker:BlobTracker, roi_coords_for_displament,roi_for_mask=(495, 0, 300, 400,),save_video=False):
    if save_video:
        output_video_path = './video_out/output_video.mp4'
        # 获取视频帧的尺寸
        height, width = image_sequence[0].shape[:2]
        frame_rate = 30
        # 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
        video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    previous_frame = None
    unet = Unet()
    for i, frame in enumerate(image_sequence):
        # 使用水岭算法检测当前帧的斑块
        processor = Processer(frame)
        blocks = processor.watershed_method(frame, mask_options=roi_for_mask, fg_method='', ifshow=False,
                                            bg_iter=3, fg_iter=8,
                                            opening_iter=3,
                                            min_area=200,
                                            unet=unet,
                                            )

        if previous_frame is not None:
            # 计算当前帧与上一帧的位移
            image_shift = processor.calculate_image_shift(previous_frame, frame, roi_coords_for_displament)

            # 进行斑块跟踪和计数，并显示跟踪结果
            tracker.track_blobs_with_kalman_kdtree(blocks, image_shift, frame)

        # 显示当前帧的跟踪结果

        cv2.putText(frame, f'Num SiC: {tracker.get_total_blobs_count()}', (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(frame, f'Current frame: {i+1}/{len(image_sequence)+1}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 3)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # 按下 'ESC' 键退出
            break
        if save_video:
            # 写入当前帧到视频文件
            video_writer.write(frame)

        previous_frame = frame
        # 释放 VideoWriter 对象

    if save_video:
        video_writer.release()

    tracker.save_history_to_npy(r'F:\python\object_detection\test\data', 'blocks_data3600')
    print(f"总共有 {tracker.get_total_blobs_count()} 个斑块越过了检测线.")
    cv2.destroyAllWindows()



def main():
    # 感兴趣区域的坐标 (x, y, w, h)
    roi_coords = (600, 0, 100, 600)

    total_dy = 0
    for i in range(80,92,1):
        img_path1 = f'../../data/img/25_glass005_{i:04}.jpg'
        img_path2 = f'../../data/img/25_glass005_{i+1:04}.jpg'
        # 读取输入的两张图片
        frame1 = cv2.imread(img_path1)
        frame2 = cv2.imread(img_path2)

        dx, dy = calculate_image_shift(frame1,frame2, roi_coords)
        total_dy += dy
        print(f'相对平移量：dx = {dx}, dy = {dy}')
    print(f'总共平移量：dy = {total_dy}')

    img_path1 = '../../data/img/25_glass005_0080.jpg'
    img_path2 = '../../data/img/25_glass005_0091.jpg'
    # 读取输入的两张图片
    frame1 = cv2.imread(img_path1)
    frame2 = cv2.imread(img_path2)
    # 计算两张图像中感兴趣区域的相对平移量
    dx, dy = calculate_image_shift(frame1,frame2, roi_coords)

    print(f'间隔10张相对平移量：dx = {dx}, dy = {dy}')






if __name__ == '__main__':
    main()
    pass