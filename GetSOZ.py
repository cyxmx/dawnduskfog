import numpy as np
import datetime

def day_of_year(date):
    """计算给定日期是一年中的第几天"""
    return date.timetuple().tm_yday

def solar_declination(day_of_year):
    """根据一年中的第几天计算太阳赤纬"""
    return 23.44 * np.sin(np.deg2rad((360 / 365.0) * (day_of_year - 81)))

def hour_angle(longitude, time):
    """根据经度和当地时间计算太阳时角"""
    # 计算太阳时间（solar time），基于经度调整的当地时间
    solar_time = time.hour + time.minute / 60 + time.second / 3600 + longitude / 15.0
    return np.deg2rad(15 * (solar_time - 12))

def parse_time_string(time_string):
    """将 hhmmss 格式的字符串转换为时间对象"""
    if len(time_string) != 6:
        raise ValueError("时间字符串格式应为 hhmmss,长度应为6位。")
    hour = int(time_string[:2])
    minute = int(time_string[2:4])
    second = int(time_string[4:6])
    return datetime.time(hour=hour, minute=minute, second=second)

def solar_zenith_angle(lat_matrix, lon_matrix, date_time_str):
    """根据二维纬度和经度矩阵计算每个点的太阳天顶角"""
    # 当前日期
    today = datetime.date.today()
    
    # 将时间字符串解析为 time 对象
    time_obj = parse_time_string(date_time_str)
    
    # 将日期和时间组合为 datetime 对象
    date_time = datetime.datetime.combine(today, time_obj)
    
    # 获取一年中的第几天
    day = day_of_year(date_time)

    # 计算太阳赤纬
    delta = solar_declination(day)

    # 计算太阳时角
    H_matrix = hour_angle(lon_matrix, date_time)

    # 将纬度转换为弧度
    lat_rad_matrix = np.deg2rad(lat_matrix)

    # 太阳天顶角公式（矩阵运算）
    cos_theta_matrix = (np.sin(lat_rad_matrix) * np.sin(np.deg2rad(delta)) +
                        np.cos(lat_rad_matrix) * np.cos(np.deg2rad(delta)) * np.cos(H_matrix))
    
    # 计算天顶角并返回结果矩阵
    return np.rad2deg(np.arccos(cos_theta_matrix))

if __name__ == '__main__':
    filename = f'./data-AGRI/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240915100000_20240915101459_4000M_V0001.HDF'
    # filename = f'./data-AGRI/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240915193000_20240915194459_4000M_V0001.HDF'
    geo_desc = [28.7, 41.5, 116.2, 129.0, 0.05] # lat_S, lat_N, lon_W, lon_E, step
    lat_S, lat_N, lon_W, lon_E, step = [1000 * x for x in geo_desc]
    lat = np.arange(lat_N, lat_S, -step) / 1000 # lat.shape = (256,)
    lon = np.arange(lon_W, lon_E, step) / 1000 # lon.shape = (256,)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    # date_time = datetime.datetime.now()  # 当前时间
    date_time_end = filename[-22:-16] # 结束测量时间
    date_time_start = filename[-37:-31] # 开始测量时间
    soz_matrix = solar_zenith_angle(lat_mesh, lon_mesh, date_time_start)
    # soz_matrix = solar_zenith_angle(lat_mesh, lon_mesh, date_time_end)
    # print(type(soz_matrix))
    print(soz_matrix*-0.694)
    print(soz_matrix*-0.694+338.296-3)
