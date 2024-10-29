import cv2

from processer import *
from tracking.test_tracking import *

import concurrent.futures

def load_image(idx):
    """单独的图像读取函数"""
    img_path = f"../data/img/25_glass005_{idx + 1:04}.jpg"
    current_frame = cv2.imread(img_path)
    if current_frame is None:
        print(f"无法读取图像: {img_path}")
    return current_frame


def load_image_sequence(start_frame=80, end_frame=200):
    start_frame = start_frame
    end_frame = end_frame
    image_sequence = []

    # 使用多线程加速读取图片，并保持顺序
    with concurrent.futures.ThreadPoolExecutor() as executor:
        image_sequence = list(executor.map(load_image, range(start_frame, end_frame)))

    # 过滤掉读取失败的图像
    image_sequence = [img for img in image_sequence if img is not None]

    return image_sequence


def debug_1():

    img_path = '../data/img/25_glass005_0081.jpg'
    img = cv2.imread(img_path,)
    processer = Processer(img)
    # 移除高亮部分并进行直方图均衡化
    highlight_thresh = 20  # 设定亮度阈值
    result_img = processer.remove_highlight_and_equalize(img, highlight_thresh)

    processer.separate_sic_blocks_with_morphology(binary_img)
    pass

def debug_2():
    processer = Processer('../data/img/25_glass005_0081.jpg')
    processer.process_and_display_stream(300,500, min_area=100, max_area=5000, delay=0.1)
    pass

def debug_watered():

    img_path = '../data/img/25_glass005_0081.jpg'
    ori_img_data = cv2.imread(img_path)
    processor = Processer(ori_img_data)

    # 通过水岭算法分割碳化硅斑块
    blocks = processor.watershed_method(ori_img_data,mask_options=(495, 0, 300, 400,),fg_method='',ifshow=True,
                                        bg_iter=3,fg_iter=6,
                                        opening_iter=3,
                                        min_area=200,
                                        dist_rate=0.1,
                                        unet=Unet(),
                                        )

    ax = processor.axs[3]  # 获取第四个子图的轴对象
    for block in blocks:
        # 绘制边界框等
        x,y,w,h = block.bbox
        id = block.id
        center = block.center

        # 添加边框
        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='yellow', facecolor='none', lw=1))  # 绘制边框

        # 添加ID
        ax.text(x+10, y+10, f'ID: {id}', color='red', fontsize=5)

        # 添加中心点
        ax.plot(center[0], center[1], 'ro',markersize=1)  # 绘制红色中心点
        print(block)

    plt.show()


def debug_watered_sequence():
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as patches

    image_sequence = load_image_sequence(start_frame=1, end_frame=3600)
    fig, ax = plt.subplots()

    processor = None  # 处理器对象占位符
    unet = Unet()  # 初始化Unet检测模型

    # 先处理第一帧以初始化展示
    first_frame = image_sequence[0]
    processor = Processer(first_frame)
    ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))  # 初始显示第一帧

    def update(i):
        """更新函数，用于每一帧处理和显示"""
        frame = image_sequence[i]
        ax.clear()  # 清除前一帧的显示
        processor = Processer(frame)

        # 使用水岭算法分割碳化硅斑块
        blocks = processor.watershed_method(frame, mask_options=(0, 0, 700, 700,), fg_method='', ifshow=False,
                                            bg_iter=3, fg_iter=6,
                                            opening_iter=3,
                                            min_area=200,
                                            dist_rate=0.1,
                                            unet=unet,
                                            )

        # 显示当前帧
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.text(10, 10, f'Frame {i + 1}/{len(image_sequence)}', color='red', fontsize=10)

        # 遍历所有检测到的斑块，绘制边界框、ID 和中心点
        for block in blocks:
            x, y, w, h = block.bbox
            id = block.id
            center = block.center

            # 添加边框
            ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='yellow', facecolor='none', lw=1))

            # 添加ID
            ax.text(x + 10, y + 10, f'ID: {id}', color='red', fontsize=5)

            # 绘制轮廓 (contour 是一系列坐标点)
            polygon = patches.Polygon(block.contour.reshape(-1, 2), closed=True, edgecolor='blue', fill=False, lw=1)
            ax.add_patch(polygon)

            # 添加中心点
            ax.plot(center[0], center[1], 'ro', markersize=1)  # 绘制红色中心点

        ax.set_title(f'Frame {i + 1}/{len(image_sequence)}')  # 显示帧编号
        plt.axis('off')  # 隐藏坐标轴

    # 创建动画，逐帧调用update函数
    anim = FuncAnimation(fig, update, frames=len(image_sequence), interval=10, repeat=False)

    # 显示动画
    plt.show()




def debug_3():
    img_path = '../data/img/25_glass005_0081.jpg'
    ori_img_data = cv2.imread(img_path)
    processor = Processer(ori_img_data)
    roi_coords = (495, 0, 300, 200)  # 示例感兴趣区域坐标

    # 跟踪碳化硅斑块并统计数量11
    total_blocks = track_sic_blocks1(processor, start_frame=80, end_frame=200, roi_coords=roi_coords, min_area=200,)
    print(f'通过ROI的碳化硅斑块总数量: {total_blocks}')

def debug_tracker():



    image_sequence = load_image_sequence(start_frame=80, end_frame=2000)

    roi_coords_for_displament = (400, 0, 200, 400)  # 示例感兴趣区域坐标
    roi_for_mask = (0, 0, 700, 700)
    tracker = BlobTracker(detection_line_y=100)  # 在y=300的位置添加检测线
    # 跟踪碳化硅斑块并统计数量11
    process_image_sequence(image_sequence, tracker, roi_coords_for_displament=roi_coords_for_displament,roi_for_mask=roi_for_mask)

    pass


def debug_kalman_tracker():
    image_sequence = load_image_sequence(start_frame=1, end_frame=3600)

    roi_coords_for_displament = (400, 0, 200, 400)  # 示例感兴趣区域坐标
    roi_for_mask = (0, 0, 700, 700)
    tracker = BlobTracker(detection_line_y=100)  # 在y=300的位置添加检测线
    # 跟踪碳化硅斑块并统计数量11
    process_image_sequence_with_kalman(image_sequence, tracker, roi_coords_for_displament=roi_coords_for_displament,roi_for_mask=roi_for_mask,save_video=True)



if __name__ == '__main__':
    # debug_3()
    # debug_watered()
    # debug_watered_sequence()
    # debug_tracker()
    debug_kalman_tracker()
    pass