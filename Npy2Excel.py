import numpy as np
import pandas as pd


class NpyToExcelConverter:
    def __init__(self, npy_file_path, output_excel_path):
        """
        初始化转换器类
        :param npy_file_path: .npy 文件的路径
        :param output_excel_path: 输出 Excel 文件的路径
        """
        self.npy_file_path = npy_file_path
        self.output_excel_path = output_excel_path
        self.data = None

    def load_npy_data(self):
        """
        从 npy 文件加载数据
        """
        try:
            self.data = np.load(self.npy_file_path, allow_pickle=True)
            if isinstance(self.data, np.ndarray):
                print(f"加载的数据是包含 {len(self.data)} 个元素的数组。")
            else:
                print("数据格式不正确，期望是一个 ndarray。")
                self.data = None
        except Exception as e:
            print(f"加载 npy 文件时出错: {e}")

    def process_data(self):
        """
        处理加载的数据，并将其转换为 DataFrame 格式
        """
        if self.data is None:
            print("数据尚未加载，请先加载 npy 文件。")
            return

        # 定义要保存的列名
        columns = ["Show ID", "Classification", "When Crossed Center","when_crossed_area", "When Crossed Contour", "Trajectory Points"]
        rows = []

        # 用于存储不同分类的数据
        classified_data = {"class_cut_out": [], "class_cutting": [], "class_cut_in": [], "class_flow_out": [],"class_normal": []}

        # 遍历数据，提取信息
        for show_data in self.data:
            show_id = show_data.get('show_id', None)
            classification = None
            when_crossed_center = None
            when_crossed_contour = None
            when_crossed_area = None
            trajectory_points = []

            # 遍历每个斑块的数据

            if 'when_disappeared_state' in show_data:
                classification = show_data['when_disappeared_state'].get('classification', None)
            if 'when_crossed_state' in show_data:
                when_crossed_center = show_data['when_crossed_state'].get('center', None)
                if when_crossed_center is not None:
                    when_crossed_center = tuple(round(coord, 2) for coord in when_crossed_center)
                when_crossed_contour = show_data['when_crossed_state'].get('contour', None)
                when_crossed_area = show_data['when_crossed_state'].get('area', None)
            for block_data in show_data.get('block_history_data', []):
                trajectory_point = block_data.get('center', None)
                if trajectory_point is not None:
                    trajectory_point = tuple(round(coord, 2) for coord in trajectory_point)
                trajectory_points.append(trajectory_point)


            # 根据分类将数据添加到不同列表中
            if classification is not None:
                row = [show_id, classification, when_crossed_center, when_crossed_area, when_crossed_contour, trajectory_points]

                classified_data[f'class_{classification}'].append(row)

        # 将所有分类的数据按横轴拼接
        for key, data in classified_data.items():
            if data:
                df = pd.DataFrame(data, columns=columns)
                df.insert(0, 'Index', pd.Series(range(1, len(df) + 1)))  # 添加一个列来标识分类类别
                rows.append(df)

        # 转换为 DataFrame
        self.df = pd.concat(rows, axis=1) if rows else None

    def save_to_excel(self):
        """
        将处理后的数据保存到 Excel 文件中
        """
        if not hasattr(self, 'df') or self.df is None:
            print("没有数据可保存，请先处理数据。")
            return

        try:
            # 保存到 Excel 文件
            self.df.to_excel(self.output_excel_path, index=False, engine='openpyxl')
            print(f"数据已成功保存到 {self.output_excel_path}")
        except Exception as e:
            print(f"保存到 Excel 文件时出错: {e}")

    def convert(self):
        """
        执行从 npy 文件到 Excel 文件的转换过程
        """
        self.load_npy_data()
        self.process_data()
        self.save_to_excel()


# 示例用法
if __name__ == "__main__":
    npy_file_path = r"F:\python\object_detection\test\data\blocks_data3600_2024年10月28日17h29m37s.npy"  # 输入 .npy 文件的路径
    output_excel_path = r"./data/output_data.xlsx"  # 输出 Excel 文件的路径

    converter = NpyToExcelConverter(npy_file_path, output_excel_path)
    converter.convert()
