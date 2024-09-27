# 从hdf文件中读取时间，并且将地图网格化求每一个网格的太阳天顶角
import os
import ephem
import numpy as np
import h5py
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timedelta


def read_hdf_time(file_path):
# 读取 HDF 文件
    with h5py.File(file_path, 'r') as hdf_file:
        start_date = hdf_file.attrs['Observing Beginning Date']  # 开始时间
        end_date = hdf_file.attrs['Observing Ending Date']  # 结束时间
        start_time = hdf_file.attrs['Observing Beginning Time']  # 开始时间
        end_time = hdf_file.attrs['Observing Ending Time']  # 结束时间
        # 将字节串转换为字符串
        start_date = start_date.decode('utf-8')
        end_date = end_date.decode('utf-8')
        start_time = start_time.decode('utf-8')
        end_time = end_time.decode('utf-8')
        # 处理日期格式，将 '-' 替换为 '/'
        formatted_start_date = start_date.replace('-', '/')
        formatted_end_date = end_date.replace('-', '/')
        # 处理时间格式，去除毫秒部分（保留到秒）
        formatted_start_time = start_time.split('.')[0]
        formatted_end_time = end_time.split('.')[0] 
        # 拼接日期和时间，形成最终的 observer.date 所需格式
        observer_start = f"{formatted_start_date} {formatted_start_time}"
        observer_end = f"{formatted_end_date} {formatted_end_time}"
        # 转换为 datetime 对象
        start_time = datetime.strptime(observer_start, '%Y/%m/%d %H:%M:%S')
        end_time = datetime.strptime(observer_end, '%Y/%m/%d %H:%M:%S')


    print(f"开始时间: {observer_start}")
    print(f"结束时间: {observer_end}")
    return observer_start, observer_end, start_time, end_time


def calc_sun_altitude(lat, lon, utc_time):
    observer = ephem.Observer()
    observer.lat, observer.lon = str(lat), str(lon)  # 目标区域的纬度和经度
    observer.date = utc_time  # 设置观测时间 (UTC时间)
    
    sun = ephem.Sun()
    sun.compute(observer)  # 计算太阳的位置
    sun_altitude = sun.alt 
    # 太阳高度角
    # sun_sea = sun.alt* 180 / 3.14159
    # 太阳天顶角
    sun_zenith_angle = 90 - (sun_altitude * 180.0 / ephem.pi)

    # 返回太阳天顶角（以角度为单位）
    return sun_zenith_angle


# def process_hdf_files_in_folder(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".HDF"):  # 只处理 HDF 文件
#             file_path = os.path.join(folder_path, filename)
#             print(f"Processing file: {file_path}")
#             # 读取起止时间
#             observer_start, observer_end, start_time, end_time = read_hdf_time(file_path)
#             # 初始化时刻
#             current_time = start_time
#             altitudes = []
#             # 循环计算时间段内的太阳高度角
#             while current_time <= end_time:
#                 altitude = calc_sun_altitude(lat, lon, current_time)
#                 altitudes.append(altitude)
#                 current_time += interval
#             average_altitude = sum(altitudes) / len(altitudes)
#             print(f"Solar angle for {observer_start}: {average_altitude}")

# 计算给定区域的太阳高度角分布
def calculate_solar_altitude_grid(file_path, lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    """
    在指定的地理区域进行网格化，并计算每个网格点的太阳高度角。
    参数:
    - hdf_file: HDF 文件路径
    - lat_min: 最小纬度
    - lat_max: 最大纬度
    - lon_min: 最小经度
    - lon_max: 最大经度
    - lat_step: 纬度网格步长
    - lon_step: 经度网格步长
    返回:
    - result: 包含每个网格点 (纬度, 经度) 和对应太阳高度角的列表
    """
    
    # 获取 HDF 文件中的 UTC 时间
    observer_start, observer_end, start_time, end_time = read_hdf_time(file_path)
    utc_time = observer_start
    print(utc_time)
    
    # 初始化结果列表
    result = []
    
    # 生成纬度和经度的网格点
    latitudes = np.arange(lat_min, lat_max + lat_step, lat_step)
    longitudes = np.arange(lon_min, lon_max + lon_step, lon_step)
    
    # 遍历所有网格点，计算太阳高度角
    for lat in latitudes:
        for lon in longitudes:
            # 计算当前网格点的太阳高度角
            solar_altitude = calc_sun_altitude(lat, lon, utc_time)
            
            # 保存结果 (纬度, 经度, 太阳高度角)
            result.append((lat, lon, solar_altitude))
    
    return result

def save_results_to_file(result, output_file):
    """
    将结果保存到指定的 CSV 文件中。

    参数:
    - result: 包含每个网格点 (纬度, 经度, 太阳高度角) 的列表
    - output_file: 输出的文件路径
    """
    # 打开文件以写入模式 ('w')
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 写入CSV文件头
        writer.writerow(["Latitude", "Longitude", "Solar Altitude"])
        
        # 写入每一行数据
        for row in result:
            writer.writerow(row)


# 主程序入口
if __name__ == "__main__":
    # 给定的区域范围
    lat_min = 28.7  # 最小纬度
    lat_max = 41.5  # 最大纬度
    lon_min = 116.2 # 最小经度
    lon_max = 129.0 # 最大经度
    
    # 网格步长，假设我们每隔 1 度进行网格化
    lat_step = 0.05
    lon_step = 0.05

    # interval = timedelta(minutes=2)

    
    # HDF 文件路径
    folder_path = 'E:\\FY4B\\3'
    for filename in os.listdir(folder_path):
        if filename.endswith(".HDF"):  # 只处理 HDF 文件
            file_path = os.path.join(folder_path, filename)
            date = filename[44:56]
            print(date)
            # file_path = "E:\\实验室\\晨昏海雾\\dynamic\\FY4B\\FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240916000000_20240916001459_4000M_V0001.HDF"
    
            # 计算并打印结果
            grid_solar_altitudes = calculate_solar_altitude_grid(file_path, lat_min, lat_max, lon_min, lon_max, lat_step, lon_step)
            output_file = "./soz_record_3/solar_altitudes"+date+".csv"
            save_results_to_file(grid_solar_altitudes, output_file)
    
            print(f"结果已保存到 {output_file}")
    
            # for lat, lon, solar_altitude in grid_solar_altitudes:
            #     print(f"纬度: {lat}, 经度: {lon}, 太阳高度角: {solar_altitude:.2f}")
            

# if __name__ == "__main__":
# # 示例：目标区域的纬度和经度
#     lat = 5
#     lon = 129.0
#     # 设置时间间隔（例如每隔10分钟计算一次）
#     interval = timedelta(minutes=2)
#     folder_path = 'E:\\实验室\\晨昏海雾\\dynamic\\FY4B_09'
#     process_hdf_files_in_folder(folder_path)

