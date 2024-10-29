import cv2
import numpy as np
from matplotlib import pyplot as plt
from img_process import plt_plot
import sys,os
import time
from scipy.spatial import KDTree
import copy
import concurrent.futures
# 添加 'UNet_Demo' 目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UNet_Demo')))

from UNet_Demo.unet import Unet

class Block:
    def __init__(self, block_id, bbox, contour):
        self.id = block_id
        self.bbox = bbox  # (x, y, w, h)
        self.center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        self.area = bbox[2] * bbox[3]
        self.contour = contour
        self.birthed_before_line = False
        self.crossed = False  # 标记是否已跨越线
        self.show_id = -1  # 用于显示的ID
        self.lost_frames = 0  # 丢失帧数
        self.kalman_tracker = None
        self.has_been_tracked_in_this_frame = False
        self.classification = None
        self.sure_for = None # 用于存储分类的确定性

        self.history = []  # 用于记录历史信息

        # 初始化时保存初始状态到历史记录中
        self.record_history()
        self.when_crossed_state = None
        self.when_disappeared_state = None

    def __repr__(self):
        return f"Block(ID={self.id}, BBox={self.bbox}, Center={self.center}, Area={self.area})"

    def apply_shift(self,shift):
        self.center = (self.center[0]+shift[0],self.center[1]+shift[1])
        self.bbox = (self.bbox[0]+shift[0],self.bbox[1]+shift[1],self.bbox[2],self.bbox[3])
        pass

    def update_by_clone(self,block):
        self.id = block.id
        self.bbox = block.bbox
        self.center = block.center
        self.area = block.area
        self.contour = block.contour

        pass

    def update_by_concate(self,block):
        self.id = block.id
        # Merge bounding boxes
        x1 = min(self.bbox[0], block.bbox[0])
        y1 = min(self.bbox[1], block.bbox[1])
        x2 = max(self.bbox[0] + self.bbox[2], block.bbox[0] + block.bbox[2])
        y2 = max(self.bbox[1] + self.bbox[3], block.bbox[1] + block.bbox[3])
        self.bbox = (x1, y1, x2 - x1, y2 - y1)

        # Combine contours
        if isinstance(self.contour, list):
            self.contour.extend(block.contour)
        else:
            self.contour = [self.contour] + [block.contour]

        # Sum areas
        self.area += block.area

        # Average centers
        self.center = (
            (self.center[0] + block.center[0]) / 2,
            (self.center[1] + block.center[1]) / 2
        )
        pass


    def record_history(self):
        """记录当前状态到历史记录"""
        history_entry = {
            'type': 'normal',
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'contour': self.contour,
            'classification': self.classification,
        }
        self.history.append(history_entry)

    def save_crossed_state(self):
        self.when_crossed_state = {
            'type': 'crossed',
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'contour': self.contour,
            'classification': self.classification,
        }

    def save_disappeared_state(self):
        self.when_disappeared_state = {
            'type': 'disappeared',
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'contour': self.contour,
            'classification': self.classification,
        }

class BlobTracker:
    def __init__(self, detection_line_y):
        self.detection_line_y = detection_line_y  # 检测线的y坐标
        self.tracked_blobs : list[Block] = []  # 用于存储已跟踪的斑块
        self.total_blobs_count = 0  # 记录通过检测线的斑块总数
        self.blob_id_counter = 0  # 为新检测的斑块分配唯一ID
        self.max_lost_frames = 10  # 允许的最大丢失帧数
        self.out_of_bounds_line = 400  # 超出边界的y坐标阈值
        self.tool_point_center = (565,300)
        self.tool_point_end = (0,455)
        self.history_tracked_blobs :list[Block] = []  # 用于存储历史跟踪斑块

    def save_history_to_npy(self, file_path,filename='blocks_data.npy'):
        """将记录的历史斑块保存为npy文件"""
        # 将 Block 对象转换为字典格式，方便保存
        history_data = []
        for block in self.history_tracked_blobs:
            block_data = {
                'show_id': block.show_id,
                'block_history_data': block.history,
                'when_crossed_state': block.when_crossed_state,
                'when_disappeared_state': block.when_disappeared_state,
            }
            history_data.append(block_data)
        date = time.strftime('%Y年%m月%d日%Hh%Mm%Ss', time.localtime())
        save_data_name = os.path.join(file_path, f"{filename}_{date}.npy")
        # 保存到 .npy 文件
        np.save(save_data_name, history_data)
        print(f"历史数据已保存至 {file_path}")

    def add_blob(self, block:Block,initial_velocity=(0, 0)):
        """为新检测的斑块分配ID并存储"""
        block.show_id = self.blob_id_counter
        block.kalman_tracker = KalmanTracker(initial_position=block.center,
                                             initial_velocity=initial_velocity)
        self.tracked_blobs.append(block)

        self.blob_id_counter += 1

    def calculate_similarity(self, block1, block2,):
        """通过帧间位移信息判断斑块的相关性"""
        distance = np.linalg.norm(np.array(block1.center) - np.array(block2.center))
        size_diff = abs(block1.area - block2.area)

        # 判断距离和大小变化是否在可接受范围内
        return distance < 10 and size_diff < 1000

    def calculate_similarity_with_para(self, block1, block2,distance_thresh=10,size_diff_thresh=1000):
        """通过帧间位移信息判断斑块的相关性"""
        distance = np.linalg.norm(np.array(block1.center) - np.array(block2.center))
        size_diff = abs(block1.area - block2.area)

        # 判断距离和大小变化是否在可接受范围内
        return distance < distance_thresh and size_diff < size_diff_thresh

    def out_of_bounds(self, block,bound_line=200):
        """检查斑块是否超出图像边界"""
        if block.center[1] > bound_line:
            return True
        else:
            return False

    def remove_blob(self, block):
        """删除指定的斑块"""
        self.tracked_blobs.remove(block)




    def track_blobs(self, current_blocks:list[Block], image_shift, frame):
        """根据当前帧斑块与上一帧的位移信息进行斑块跟踪，并显示跟踪过程"""
        new_tracked_blobs = []

        # 绘制检测线
        cv2.line(frame, (0, self.detection_line_y), (frame.shape[1], self.detection_line_y), (0, 255, 0), 2)
        cv2.line(frame, (0, self.out_of_bounds_line), (frame.shape[1], self.out_of_bounds_line), (0, 0, 255), 2)

        for tracked_blob in self.tracked_blobs:
            tracked_blob.apply_shift(image_shift)

        for block in self.tracked_blobs:
            # self.draw_blob(frame, block,center_color=(255,255,255),rect_color=(255,255,255))  # 绘制新斑块
            pass

        # 遍历当前帧中的斑块
        for block in current_blocks:
            matched = False

            # 尝试与之前帧的斑块进行匹配
            for tracked_blob in self.tracked_blobs:
                if self.calculate_similarity(tracked_blob, block):
                    matched = True

                    # 更新跟踪斑块的位置
                    tracked_blob.update_by_clone(block)
                    tracked_blob.lost_frames = 0
                    # new_tracked_blobs.append(tracked_blob)

                    # 检查斑块是否越过检测线
                    if tracked_blob.center[1] > self.detection_line_y and not tracked_blob.crossed and tracked_blob.birthed_before_line:
                        self.total_blobs_count += 1
                        tracked_blob.crossed = True

                    # 在图像上绘制斑块的边界框、ID 和中心点
                    # self.draw_blob(frame, tracked_blob)
                    break

            # 如果未找到匹配的斑块，添加为新斑块
            if not matched:
                if block.center[1] < self.detection_line_y:
                    self.add_blob(block)
                    block.birthed_before_line = True
                # new_tracked_blobs.append(block)
                # self.draw_blob(frame, block)  # 绘制新斑块

        # self.tracked_blobs = new_tracked_blobs

        for tracked_blob in self.tracked_blobs:
            tracked_blob.lost_frames += 1
            if tracked_blob.lost_frames > self.max_lost_frames or self.out_of_bounds(tracked_blob,bound_line=self.out_of_bounds_line):
                self.remove_blob(tracked_blob)

        for block in self.tracked_blobs:
            if block.lost_frames<=1:
                self.draw_blob(frame, block)  # 绘制新斑块
        return self.tracked_blobs
        pass

        # 处理丢失帧超过阈值的斑块

    def track_blobs_with_kalman(self, current_blocks:list[Block], image_shift, frame):
        new_tracked_blobs = []
        # 绘制检测线
        cv2.line(frame, (0, self.detection_line_y), (frame.shape[1], self.detection_line_y), (0, 255, 0), 2)
        cv2.line(frame, (0, self.out_of_bounds_line), (frame.shape[1], self.out_of_bounds_line), (0, 0, 255), 2)

        # 更新已跟踪斑块的位置（预测）
        for tracked_blob in self.tracked_blobs:
            predicted_position = tracked_blob.kalman_tracker.predict()
            tracked_blob.center = predicted_position  # 更新斑块中心
            pass

        # 匹配并更新斑块的实际位置
        for block in current_blocks:
            matched = False
            for tracked_blob in self.tracked_blobs:

                if self.calculate_similarity_with_para(tracked_blob, block,distance_thresh=20,size_diff_thresh=10000):
                    matched = True
                    if tracked_blob.show_id == 16 and len(tracked_blob.history) ==312:
                        pass


                    # 使用检测到的新位置校正卡尔曼滤波器
                    correct_pos = tracked_blob.kalman_tracker.correct(block.center)
                    block.center = correct_pos
                    tracked_blob.lost_frames = 0
                    tracked_blob.update_by_clone(block)
                    tracked_blob.has_been_tracked_in_this_frame = True
                    tracked_blob.record_history()

                    # 检查斑块是否越过检测线
                    if tracked_blob.center[1] > self.detection_line_y and not tracked_blob.crossed and tracked_blob.birthed_before_line:
                        self.total_blobs_count += 1
                        tracked_blob.crossed = True
                        self.history_tracked_blobs.append(tracked_blob)
                        tracked_blob.save_crossed_state()



                    break

            # 如果没有匹配的斑块，将其视为新斑块并加入跟踪
            if not matched:
                if block.center[1] < self.detection_line_y:
                    self.add_blob(block, initial_velocity=image_shift)
                    block.birthed_before_line = True


        # 移除超过最大丢失帧数的斑块
        for tracked_blob in self.tracked_blobs:
            tracked_blob.lost_frames += 1
            tracked_blob.classification = self.judge_for_classify(tracked_blob)
            tracked_blob.save_disappeared_state()
            if tracked_blob.lost_frames > self.max_lost_frames or self.out_of_bounds(tracked_blob,
                                                                                     bound_line=self.out_of_bounds_line):
                if tracked_blob.show_id == 16:
                    pass
                # tracked_blob.save_disappeared_state()
                self.remove_blob(tracked_blob)


        # 绘制当前帧中斑块的边界框
        for tracked_blob in self.tracked_blobs:
            if tracked_blob.lost_frames <= 1:
                self.draw_blob(frame, tracked_blob)

        return self.tracked_blobs
    def track_blobs_with_kalman_kdtree(self, current_blocks:list[Block], image_shift, frame):
        new_tracked_blobs = []

        # 绘制检测线
        cv2.line(frame, (0, self.detection_line_y), (frame.shape[1], self.detection_line_y), (0, 255, 0), 2)
        cv2.line(frame, (0, self.out_of_bounds_line), (frame.shape[1], self.out_of_bounds_line), (0, 0, 255), 2)

        # 初始化情况：若没有已跟踪斑块，将当前检测到的所有斑块添加到跟踪中
        if not self.tracked_blobs:
            for block in current_blocks:
                if block.center[1] < self.detection_line_y:
                    self.add_blob(block, initial_velocity=image_shift)
                    block.birthed_before_line = True
            return self.tracked_blobs

        # 使用当前帧的检测结果构建 KD 树，基于中心点坐标
        block_centers = [block.center for block in current_blocks]
        kd_tree = KDTree(block_centers) if block_centers else None

        # 更新已跟踪斑块的位置（预测）
        for tracked_blob in self.tracked_blobs:
            predicted_position = tracked_blob.kalman_tracker.predict()
            tracked_blob.center = predicted_position  # 更新斑块中心

        # 外层循环遍历已跟踪的斑块
        unmatched_blocks = set(range(len(current_blocks)))  # 用于记录未匹配的检测结果
        for tracked_blob in self.tracked_blobs:
            matched = False

            # 在 KD 树中找到靠近当前 tracked_blob 的检测结果
            if kd_tree:
                nearby_indices = kd_tree.query_ball_point(tracked_blob.center, r=20)  # 查找半径为20的邻近点
                nearby_blocks = [current_blocks[i] for i in nearby_indices]
            else:
                nearby_blocks = []

            # 内层循环遍历邻近的检测结果，进行匹配
            for idx, block in zip(nearby_indices, nearby_blocks):
                if idx in unmatched_blocks and self.calculate_similarity_with_para(tracked_blob, block,
                                                                                   distance_thresh=20,
                                                                                   size_diff_thresh=10000):
                    matched = True
                    unmatched_blocks.discard(idx)  # 标记该检测结果已匹配

                    # 使用检测到的新位置校正卡尔曼滤波器
                    correct_pos = tracked_blob.kalman_tracker.correct(block.center)
                    block.center = correct_pos
                    tracked_blob.lost_frames = 0
                    tracked_blob.update_by_clone(block)
                    tracked_blob.has_been_tracked_in_this_frame = True
                    tracked_blob.record_history()

                    # 检查斑块是否越过检测线
                    if tracked_blob.center[
                        1] > self.detection_line_y and not tracked_blob.crossed and tracked_blob.birthed_before_line:
                        self.total_blobs_count += 1
                        tracked_blob.crossed = True
                        self.history_tracked_blobs.append(tracked_blob)
                        tracked_blob.save_crossed_state()
                    break

            # 如果当前 tracked_blob 没有找到匹配的检测结果，增加新的斑块
            tracked_blob.lost_frames += 1
            tracked_blob.classification = self.judge_for_classify(tracked_blob)
            tracked_blob.save_disappeared_state()
            if tracked_blob.lost_frames > self.max_lost_frames or self.out_of_bounds(tracked_blob,
                                                                                     bound_line=self.out_of_bounds_line):
                # tracked_blob.save_disappeared_state()
                self.remove_blob(tracked_blob)

        # 将所有未匹配的当前帧斑块添加为新斑块进行跟踪
        for idx in unmatched_blocks:
            block = current_blocks[idx]
            if block.center[1] < self.detection_line_y:
                self.add_blob(block, initial_velocity=image_shift)
                block.birthed_before_line = True

        # 绘制当前帧中斑块的边界框
        for tracked_blob in self.tracked_blobs:
            if tracked_blob.lost_frames <= 1:
                self.draw_blob(frame, tracked_blob)

        return self.tracked_blobs

    def draw_blob(self, frame, blob,rect_color=(255,0,0),center_color=(0,0,255)):
        """在图像上绘制斑块的边界框、中心点和ID"""
        x, y, w, h = blob.bbox
        x,y,w,h = round(x),round(y),round(w),round(h)
        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
        # 绘制中心点
        cv2.circle(frame, (int(blob.center[0]), int(blob.center[1])), 2, center_color, -1)
        cv2.drawContours(frame, blob.contour, -1, (0, 0, 255), 2)  # 红色轮廓
        # 显示ID
        cv2.putText(frame, f'ID: {blob.show_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if blob.crossed:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if blob.classification is not None:
            rect_color = { 'cut_in':(255, 128, 0),'cut_out':(0, 255, 128),'cutting':(128, 0, 255),'flow_out':(0,128,255),'normal':(0, 0, 0)}[blob.classification]
            cv2.putText(frame, f'{blob.classification}', (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)


    def get_total_blobs_count(self):
        return self.total_blobs_count

    def is_point_below_offset_line(self,point, offset_line):
        # 判断点是否在偏移线下方
        (x1, y1), (x2, y2) = offset_line
        px, py = point
        # 计算偏移线的斜率和截距
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            # 计算点在偏移线上的投影y值
            y_on_line = slope * px + intercept
            return py > y_on_line
        else:
            # 竖直线的情况下，只需比较x坐标
            return px > x1

    def judge_for_classify(self,block):
        tool_point_center = self.tool_point_center
        tool_point_end = self.tool_point_end
        cut_in_threshold = 15   # 车刀刀尖向右切入判断的阈值
        cut_out_threshold = 10  # 车刀刀尖向左切出判断的阈值
        flow_out_threshold = 10  # 流出和切出的y方向判断阈值
        tool_line = (tool_point_center,tool_point_end)
        tool_line_y_threshold = 5
        offset_line = ((tool_point_center[0],tool_point_center[1] - tool_line_y_threshold),(tool_point_end[0],tool_point_end[1]-tool_line_y_threshold))

        if block.sure_for is not None:
            return block.sure_for

        if block.bbox[0] > tool_point_center[0] + cut_in_threshold:
            return 'normal'


        if block.bbox[0] + block.bbox[2] < tool_point_center[0] - cut_out_threshold :
            if self.is_point_below_offset_line((block.bbox[0]+block.bbox[2],block.bbox[1]+block.bbox[3]),offset_line)\
                    and block.bbox[0] + block.bbox[2]>tool_point_center[0] - cut_out_threshold - 50:
                if block.classification != 'cut_out' and block.history[-2]['area']*0.8 > block.area:
                    block.sure_for = 'cutting'
                    print(f'block {block.show_id} is sure for cutting')
                    return 'cutting'
                else:
                    return 'cut_out'
            else:
                if block.classification != 'cut_out' or block.bbox[0] + block.bbox[2]<tool_point_center[0] - cut_out_threshold - 50:
                    return 'flow_out'
                else:
                    block.sure_for = 'cut_out'
                    return 'cut_out'

        elif ((block.bbox[0] < tool_point_center[0] + cut_in_threshold and block.bbox[0] > tool_point_center[0] - cut_in_threshold) or
                block.bbox[1] > tool_point_center[1]):
            if block.classification != 'cut_in' and block.history[-2]['area'] * 0.8 > block.area:
                block.sure_for = 'cutting'
                print(f'block {block.show_id} is sure for cutting')
                return 'cutting'
            else:
                return 'cut_in'

        else:
            return 'cutting'






class KalmanTracker:
    def __init__(self, initial_position, initial_velocity=(0, 0)):
        self.kalman = cv2.KalmanFilter(4, 2)  # State: [x, y, vx, vy], Measurement: [x, y]

        # Transition matrix for constant velocity model
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)

        # Measurement matrix
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # Initialize state
        self.kalman.statePre = np.array([[initial_position[0]],
                                         [initial_position[1]],
                                         [initial_velocity[0]],
                                         [initial_velocity[1]]], dtype=np.float32)
        self.kalman.statePost = self.kalman.statePre.copy()

    def predict(self):
        """Predict the next position of the blob"""
        predicted_state = self.kalman.predict()
        predicted_position = (float(predicted_state[0]), float(predicted_state[1]))
        return predicted_position

    def correct(self, measured_position):
        """Update the Kalman filter with the new measurement"""
        measured = np.array([[np.float32(measured_position[0])],
                             [np.float32(measured_position[1])]])
        corrected_state = self.kalman.correct(measured)
        corrected_position = (float(corrected_state[0]), float(corrected_state[1]))
        return corrected_position



class Processer:
    def __init__(self,img):
        self.ori_img = img
        self.fig, self.ax = plt.subplots(figsize=(8, 6))  # 创建一个图形和坐标轴
        self.axs = None  # 子图坐标轴
        pass

    def process(self, data):
        return data

    def create_mask_with_options(self,img, x, y, w, h, outer_color=0, inverse=False):
        """
        将矩形框以外的区域置为黑色或白色，并可根据需要反转感兴趣区域与外围区域。

        参数:
        - img: 输入的BGR图像
        - x, y, w, h: 感兴趣区域（ROI）的左上角坐标(x, y)及宽度和高度
        - outer_color: 指定外围区域的颜色，0为黑色，255为白色
        - inverse: 是否反转感兴趣区域与外围区域，True表示反转

        返回:
        - 处理后的图像
        """
        # 转换为灰度图
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img

        # 创建一个全白的掩码
        mask = np.zeros_like(gray_img)

        if inverse:
            # 如果inverse为True，保留外围部分，感兴趣区域设为outer_color
            mask[y:y + h, x:x + w] = 255  # 感兴趣区域为白色
            result = np.ones_like(gray_img) * outer_color  # 整体设为outer_color
            result[mask == 0] = gray_img[mask == 0]  # 保留外围区域内容
        else:
            # 否则，保留感兴趣区域，外围设为outer_color
            mask[y:y + h, x:x + w] = 255  # 感兴趣区域设为255
            result = cv2.bitwise_and(gray_img, mask)  # 保留ROI内容
            result[mask == 0] = outer_color  # 外围区域设为outer_color

        return result


    def watershed_method(self,img,mask_options=None,min_area=500,max_area=5000,fg_method='distance',opening_iter=2,bg_iter=3,fg_iter=3,dist_rate=0.15,unet:(None|Unet)=None,ifshow=False):
        ori_image = img
        if mask_options is not None:
            x, y, w, h = mask_options
        else:
            x,y,w,h = 0,0,ori_image.shape[1],ori_image.shape[0]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if unet is not None:
            Unet_predict = unet.get_mask(ori_image, count=False, name_classes=["background", "SiC"], mask_name='SiC')
            image = self.create_mask_with_options(Unet_predict, x, y, w, h, outer_color=0,
                                                   inverse=False)
            opening = image

        else:
            image = self.create_mask_with_options(ori_image, x, y, w, h, outer_color=255,
                                                   inverse=False)  # 得到的结果已经是灰度图
            image = cv2.bitwise_not(image)
            ret, image = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
            opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=opening_iter)  # 开运算出去噪点

        # 获取背景
        sure_bg = cv2.dilate(opening, kernel, iterations=bg_iter)
        sure_bg = np.uint8(sure_bg)

        if fg_method == 'distance':
            # Perform the distance transform algorithm
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            # Normalize the distance image for range = {0.0, 1.0}
            cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
            # Finding sure foreground area
            ret, sure_fg = cv2.threshold(dist_transform, dist_rate * dist_transform.max(), 255, 0)
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

        else:

            # 获取前景
            sure_fg = cv2.erode(opening, kernel, iterations=fg_iter)  # sure foreground area
            sure_fg = np.uint8(sure_fg)

            unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area

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


        # 获取所有标签并初始化筛选后区域的计数
        unique_labels = np.unique(markers)
        independent_regions = unique_labels[(unique_labels > 1)]  # 排除背景和分水岭线
        filtered_regions_count = 0  # 筛选后区域计数
        blocks : list[Block] = []
        # 统计每个区域的面积并进行筛选
        for label in independent_regions:

            area = np.sum(markers == label)  # 计算当前标签对应区域的像素数（面积）
            if min_area <= area <= max_area:  # 如果区域面积在指定范围内

                # 查找区域的轮廓
                mask = np.uint8(markers == label)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 计算边界框 (bounding box)
                if contours and len(contours) > 0:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    block = Block(block_id=label, bbox=(x, y, w, h), contour=contours[0])
                    blocks.append(block)
                filtered_regions_count += 1
                if ifshow:
                    # 可视化：将满足条件的区域标记为绿色
                    final_img[markers == label] = [0, 255, 0]  # 标记满足条件的区域为绿色
                    pass


        # 打印符合面积条件的区域数量
        # print(f"符合面积范围的区域数量: {filtered_regions_count}")



        if ifshow:
            self.plt_plot([ori_image,sure_fg, markers_copy, final_img],['Ori Image','sure_fg', 'Markers', 'Segmented Image'])

        return blocks

    @staticmethod
    def process_blob(label, markers, min_area, max_area):
        """处理每个斑块的面积计算和轮廓查找"""
        area = np.sum(markers == label)  # 计算当前标签对应区域的面积

        if min_area <= area <= max_area:  # 如果区域面积在指定范围内
            # 查找区域的轮廓
            mask = np.uint8(markers == label)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 计算边界框 (bounding box)
            if contours and len(contours) > 0:
                x, y, w, h = cv2.boundingRect(contours[0])
                return Block(block_id=label, bbox=(x, y, w, h), contour=contours[0]), area
        return None, 0

    def separate_sic_blocks_with_morphology(self,img, min_area=100, max_area=5000):

        img_inv = cv2.bitwise_not(img)
        ret,img_bin = cv2.threshold(img_inv , 110, 255, cv2.THRESH_BINARY)
        # 3. 断开狭窄连接
        # 3.1 定义用于腐蚀的结构元素，大小取决于狭窄连接的宽度
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_eroded = cv2.erode(img_bin, erosion_kernel, iterations=2)

        # 3.2 膨胀恢复主要结构，但狭窄连接已被断开
        img_dilated = cv2.dilate(img_eroded, erosion_kernel, iterations=2)

        # 4. 连通区域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilated, connectivity=8)

        # 5. 过滤与分离
        filtered_labels = np.zeros_like(labels, dtype=np.uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                filtered_labels[labels == i] = 255
            # filtered_labels[labels == i] = 255

        # 6. 查找轮廓
        contours, hierarchy = cv2.findContours(filtered_labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        print(f"过滤后保留 {len(contours)} 个斑块。")
        # 7. 可视化与输出
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            cv2.drawContours(img_color, [contour], -1, (255, 0, 0), 2)  # 红色轮廓

            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色边界框
            cv2.putText(img_color, f"{idx + 1}", (x+10, y - 10+h), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # self.plt_plot([img_inv, img_dilated, filtered_labels,img_color])
        return img_color,contours, bounding_boxes

    def remove_highlight_and_equalize(self,img, highlight_thresh=200):
        """
        移除图像的高亮部分，并对剩余部分进行直方图均衡化。

        参数:
        - img: 输入的BGR图像
        - highlight_thresh: 高亮部分的阈值，超过该值的部分被去除

        返回:
        - 去除高亮并均衡化处理后的图像
        """
        # 转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 创建掩码，将高亮部分设为0
        mask = gray_img < highlight_thresh  # 找到低于阈值的部分
        low_light_img = np.copy(gray_img)  # 复制灰度图
        low_light_img[~mask] = 0  # 将高亮部分设为0（黑色）

        # 对剩下的部分进行直方图均衡化
        equalized_img = cv2.equalizeHist(low_light_img)

        return equalized_img

    @staticmethod
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


    def show_gray_img(self,img):
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
        pass

    def plt_plot(self,img_list=[],name_list=[]):
        # plt.figure(figsize=(9, 6))
        if len(name_list) != len(img_list):
            name_list = [f"img-{i}" for i in range(len(img_list))]
        # 清除当前绘制
        self.ax.clear()
        self.ax.axis('off')
        # 创建新的子图，放在 self.fig 上，但不创建新的 fig 对象
        num_imgs = len(img_list)
        self.axs = self.fig.subplots(1, num_imgs) if num_imgs > 1 else [self.ax]

        # 如果只有一个图像，需要包裹为列表以便处理
        if num_imgs == 1:
            self.axs = [self.ax]

        # 遍历并绘制图像
        for i, (img, ax) in enumerate(zip(img_list, self.axs)):
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(name_list[i])
            ax.axis('off')

        # 更新画布，显示绘制
        self.fig.canvas.draw()
        # for i, img in enumerate(img_list):
        #     plt.subplot(1, len(img_list), i + 1), plt.axis('off'), plt.title(name_list[i])
        #     plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        # pass

    def plt_plot_with_rectangle(self, img_list=[]):
        plt.figure(figsize=(9, 6))
        for i, img in enumerate(img_list):
            plt.subplot(1, len(img_list), i + 1), plt.axis('off'), plt.title(f"img-{i}")
            ax = plt.gca()
            ax.add_patch(plt.Rectangle((500, 0), 200, 150, color="red", fill=False, linewidth=1, linestyle='--'))
            ax.add_patch(plt.Rectangle((580, 300), 200, 200, color="blue", fill=False, linewidth=1, linestyle='--'))
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        pass

    def display_image_stream(self, images, titles=None, delay=1):
        """
        动态显示一系列图像。

        参数:
        - images: 要显示的图像列表
        - titles: 图像标题列表，可选
        - delay: 每张图像显示的时间（秒）
        """
        plt.figure(figsize=(8, 6))
        for idx, img in enumerate(images):
            plt.clf()  # 清除当前图像
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if titles and idx < len(titles):
                plt.title(titles[idx])
            else:
                plt.title(f"Image {idx + 1}")
            plt.axis('off')
            plt.pause(delay)  # 暂停以显示图像
        plt.ioff()  # 关闭交互模式
        plt.show()

    def display_single_image(self, img, title=None, delay=1):
        """
        动态显示单张图像。

        参数:
        - img: 要显示的图像
        - title: 图像标题，可选
        - delay: 显示的时间（秒）
        """
        self.ax.clear()  # 清除当前图像
        if len(img.shape) == 2:
            self.ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title("Image")
        self.ax.axis('off')
        plt.pause(delay)  # 暂停以显示图像

    def process_and_display_stream(self, start,len, min_area=100, max_area=5000, delay=1,show_dynamic=True):
        """
        处理一系列图像并动态显示处理结果。

        参数:
        - image_paths: 图像路径列表
        - min_area: 分离斑块的最小面积
        - max_area: 分离斑块的最大面积
        - delay: 每张图像显示的时间（秒）
        """
        if show_dynamic:
            plt.ion()  # 启用 Matplotlib 的交互模式
        processed_images = []
        titles = []
        for idx in range(len):
            # 读取图像
            img = cv2.imread(f"../data/img/25_glass005_{start+idx + 1:04}.jpg",)
            if img is None:
                print(f"无法读取图像: {idx+1}")
                continue

            masked_img = self.create_mask_with_options(img, 495, 0, 300, 200, outer_color=255,
                                                            inverse=False)
            # 分离斑块
            img_separated = self.separate_sic_blocks_with_morphology(masked_img, min_area, max_area)
            # 添加到列表
            processed_images.append(img_separated)
            titles.append(f"Processed {idx + 1}")

            if show_dynamic:
                self.display_single_image(img_separated, title=[f"Processed {idx + 1}"], delay=delay)
        # 动态显示
        # self.display_image_stream(processed_images, titles=titles, delay=delay)
        return processed_images


if __name__ == '__main__':
    img_path = '../data/img/25_glass005_0081.jpg'
    ori_img_data = cv2.imread(img_path, )
    processer = Processer(ori_img_data)
    masked_img = processer.create_mask_with_options(ori_img_data, 495, 0, 300, 200,outer_color=255, inverse=False)
    processer.separate_sic_blocks_with_morphology(masked_img, min_area=200)

    # processer.show_gray_img(masked_img)

    plt.show()
    pass