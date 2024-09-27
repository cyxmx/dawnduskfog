# 插值波段数据对应到经纬度网络中，结合对应SOZ，设置阈值，进行海雾识别。
import h5py
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# 定义目标区域的经纬度范围和分辨率
lat_min, lat_max = 28.7, 41.5
lon_min, lon_max = 116.2, 129.0
resolution = 0.05  # 分辨率为0.05度

# 生成0.05度分辨率的经纬度网格
lat_grid = np.arange(lat_min, lat_max, resolution)
lon_grid = np.arange(lon_min, lon_max, resolution)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# 读取查找表文件
lookup_file = "FY4B-_DISK_1050E_GEO_NOM_LUT_20240227000000_4000M_V0001.raw"
# 读取数据，每个纬度和经度相邻存储，5496行，2748列
data = np.fromfile(lookup_file, dtype='float64').reshape(5496, 2748)

# 纬度和经度分别位于奇数行和偶数行
latitudes = data[0::2, :]
longitudes = data[1::2, :]
# 这里的 latitudes 和 longitudes 数据为纬度和经度
points = np.array([latitudes.flatten(), longitudes.flatten()]).T

# 读取HDF文件中不同的通道数据
hdf_file = "E:\\FY4B\\3\\FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240305010000_20240305011459_4000M_V0001.HDF"


# 读取CSV文件中的 Solar Altitude (soz) 数据
csv_file = "./soz_record_3/solar_altitudes202403051000.csv"

# 定义目标区域的经纬度范围和分辨率
lat_min, lat_max = 28.7, 41.5
lon_min, lon_max = 116.2, 129.0
resolution = 0.05  # 分辨率为0.05度

# 生成0.05度分辨率的经纬度网格
lat_grid = np.arange(lat_min, lat_max, resolution)
lon_grid = np.arange(lon_min, lon_max, resolution)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# 读取CSV文件中的 Solar Altitude (soz) 数据
df = pd.read_csv(csv_file)

# 过滤CSV文件中的经纬度范围
df_filtered = df[(df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) & 
                 (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]

# 确保纬度和经度数据的形状一致，并形成points数组
points = np.array([df_filtered['Latitude'], df_filtered['Longitude']]).T

# 检查 points 和 Solar Altitude 的长度是否一致
print(f"points shape: {points.shape}")
print(f"Solar Altitude values shape: {df_filtered['Solar Altitude'].shape}")

# 插值 Solar Altitude 数据到目标区域的经纬度网格
solar_altitude_interp = griddata(points, df_filtered['Solar Altitude'], (lat_mesh, lon_mesh), method='linear')

# 读取HDF文件中不同的通道数据
with h5py.File(hdf_file, 'r') as f:
    B1_data = f['Data/NOMChannel01'][:]
    B3_data = f['Data/NOMChannel02'][:]
    B4_data = f['Data/NOMChannel03'][:]
    B5_data = f['Data/NOMChannel05'][:]
    B7_data = f['Data/NOMChannel07'][:]
    B14_data = f['Data/NOMChannel13'][:]

# 使用查找表中的经纬度进行插值 (points与波段数据大小要匹配)
points_hdf = np.array([latitudes.flatten(), longitudes.flatten()]).T

# 插值波段数据到目标区域的经纬度网格
B1_interp = griddata(points_hdf, B1_data.flatten(), (lat_mesh, lon_mesh), method='linear')
B3_interp = griddata(points_hdf, B3_data.flatten(), (lat_mesh, lon_mesh), method='linear')
B4_interp = griddata(points_hdf, B4_data.flatten(), (lat_mesh, lon_mesh), method='linear')
B5_interp = griddata(points_hdf, B5_data.flatten(), (lat_mesh, lon_mesh), method='linear')
B7_interp = griddata(points_hdf, B7_data.flatten(), (lat_mesh, lon_mesh), method='linear')
B14_interp = griddata(points_hdf, B14_data.flatten(), (lat_mesh, lon_mesh), method='linear')

# 将插值后的数据转换为 DataFrame 格式，方便后续处理
df_grid = pd.DataFrame({
    'Latitude': lat_mesh.flatten(),
    'Longitude': lon_mesh.flatten(),
    'Solar Altitude': solar_altitude_interp.flatten(),  # 将 Solar Altitude 数据加入 DataFrame
    'B1': B1_interp.flatten(),
    'B3': B3_interp.flatten(),
    'B4': B4_interp.flatten(),
    'B5': B5_interp.flatten(),
    'B7': B7_interp.flatten(),
    'B14': B14_interp.flatten()
})

# 计算 R_Sum, R_value, DE, BTD
df_grid['R_Sum'] = df_grid['B1'] + df_grid['B4']
df_grid['R_value'] = abs(df_grid['B4'] - df_grid['B1'])
df_grid['DE'] = df_grid['B3'] - df_grid['B5']
df_grid['BTD'] = df_grid['B7'] - df_grid['B14']

# 阈值计算和判定
df_grid['threshold_1'] = df_grid['Solar Altitude'] * (-0.694) + 338.296 - 3
df_grid['threshold_2'] = df_grid['Solar Altitude'] * (-0.0179) + 1.615 - 0.015
df_grid['threshold_3'] = 0.015
df_grid['threshold_4'] = 0.004
df_grid['threshold_5'] = df_grid['Solar Altitude'] * (-0.633) + 52.451 - 3
df_grid['threshold_6'] = df_grid['Solar Altitude'] * (-0.633) + 52.451 + 3

# 根据阈值进行判断
df_grid['Sea_Fog'] = (df_grid['B7'] > df_grid['threshold_1']) & \
                     ((df_grid['R_value'] < df_grid['threshold_3']) | (df_grid['R_Sum'] > df_grid['threshold_2'])) & \
                     (df_grid['DE'] < df_grid['threshold_4']) & \
                     (df_grid['BTD'].between(df_grid['threshold_5'], df_grid['threshold_6']))

# 保存结果
df_grid.to_csv("output_grid.csv", index=False)


