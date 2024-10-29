import numpy as np
import cv2

from Visualize_blocks import Visualization


if __name__ == '__main__':
    img_size = cv2.imread(r'F:\python\object_detection\data\img\25_glass005_0001.jpg', cv2.IMREAD_UNCHANGED).shape[:2]
    visualizer = Visualization(img_size)
    history_data = np.load(r'F:\python\object_detection\test\data\blocks_data3600_2024年10月28日11h05m51s.npy', allow_pickle=True)

    # 单独查看某个历史条目

    # visualizer.show_single_history_entry(history_data[1]['block_history_data'][0])

    # 显示整个block的轨迹

    visualizer.show_block_trajectory(history_data[14])
    visualizer.show_block_trajectory_dynamic(history_data)
