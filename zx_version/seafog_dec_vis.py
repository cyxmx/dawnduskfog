import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 读取 CSV 文件
output_csv = 'output_grid.csv'
df = pd.read_csv(output_csv)

# 创建一个带有地理背景的地图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# 设置地图的范围（根据你的数据范围设置）
ax.set_extent([116.2, 129.0, 28.7, 41.5], crs=ccrs.PlateCarree())

# 添加海岸线和陆地等地理特征
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.LAKES, edgecolor='blue')

# 绘制 Solar Altitude 在 80-90 之间的区域，标注为黄色
solar_altitude_yellow = df[(df['Solar Altitude'] >= 80) & (df['Solar Altitude'] <= 90)]
ax.scatter(solar_altitude_yellow['Longitude'], solar_altitude_yellow['Latitude'], color='yellow', label='Solar Altitude 80-90', s=1, alpha=0.7, transform=ccrs.PlateCarree())

# 绘制有雾和无雾区域
# 有雾区域
fog_points = df[df['Sea_Fog'] == True]
ax.scatter(fog_points['Longitude'], fog_points['Latitude'], color='red', label='Sea Fog', s=1, alpha=0.7, transform=ccrs.PlateCarree())

# 无雾区域
no_fog_points = df[df['Sea_Fog'] == False]
ax.scatter(no_fog_points['Longitude'], no_fog_points['Latitude'], color='blue', label='No Sea Fog', s=1, alpha=0.7, transform=ccrs.PlateCarree())

# 添加图例
ax.legend()

# 添加标题
plt.title("Sea Fog Detection with Solar Altitude 80-90 Highlighted", fontsize=15)

# 显示地图
plt.show()
