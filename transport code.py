#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/Users/allen/Desktop/transport data222.xlsx'  # Update the file path
data = pd.read_excel(file_path)

# Convert 'Date' to datetime format and set it as the index of the DataFrame
data['Date'] = pd.to_datetime(data['Date'], format='%Y %m-%d', errors='coerce')
data.set_index('Date', inplace=True)

# Filter the data for the period from 2014 to 2023
filtered_data = data['2012':'2022']

# Sum up the quarterly data to get annual totals
annual_data = filtered_data.resample('Y').sum()

# Calculate the annual percentage growth rate
annual_data['Petrol Growth'] = annual_data['Petrol'].pct_change() * 100
annual_data['Diesel Growth'] = annual_data['Diesel'].pct_change() * 100
annual_data['EV Total Growth'] = annual_data['EV  total'].pct_change() * 100

# Plotting the annual growth rates
plt.figure(figsize=(12, 6))
plt.plot(annual_data.index, annual_data['Petrol Growth'], label='Petrol Growth %', marker='o')
plt.plot(annual_data.index, annual_data['Diesel Growth'], label='Diesel Growth %', marker='o')
plt.plot(annual_data.index, annual_data['EV Total Growth'], label='EV Total Growth %', marker='o')
plt.title('Annual Growth Rates of Petrol, Diesel, and EV Total (2014-2023)')
plt.xlabel('Year')
plt.ylabel('Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from an Excel file
data_path = '/Users/allen/Desktop/factor table（after）.xlsx'  # Replace 'path_to_your_file.xlsx' with the actual path to your Excel file
data = pd.read_excel(data_path)

# Set the index to 'Year' if it's part of the DataFrame
data.set_index('Year', inplace=True)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Creating a heatmap using seaborn
plt.figure(figsize=(10, 8))  # Adjust the size of the figure as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Heatmap of Variables Affecting CO2 Emissions')
plt.show()


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '/Users/allen/Desktop/transport data222.xlsx'  # Change this to your file path
data = pd.read_excel(file_path)

# Convert 'Date' to datetime format and set it as the index of the DataFrame
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data.set_index('Date', inplace=True)

# Filter the data for the period from 2014 to 2023
filtered_data = data['2013':'2022']

# Sum up the quarterly data to get annual totals
annual_data = filtered_data.resample('Y').sum()

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(annual_data.index, annual_data['Petrol'], label='Petrol', marker='o')
plt.plot(annual_data.index, annual_data['Diesel'], label='Diesel', marker='o')
plt.plot(annual_data.index, annual_data['EV  total'], label='EV Total', marker='o')
plt.title('Trend of Petrol, Diesel, and EV Total Over Time (in Thousands)')
plt.xlabel('Year')
plt.ylabel('Number of Vehicles (Thousands)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 加载数据
file_path = '/Users/allen/Desktop/factor(before).xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 设置日期为索引
data['Year'] = pd.to_datetime(data['Year'], format='%Y')  # 确保年份格式正确
data.set_index('Year', inplace=True)

# 对'CO2 change'列进行时间序列分解
result = seasonal_decompose(data['CO2 change'], model='additive', period=1)

# 绘制分解结果
fig = result.plot()
fig.set_size_inches(10, 8)
plt.show()


# In[4]:


from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
import pandas as pd

# 假设data_clean是你已经加载和清洗过的数据
# 这里仅为示例，确保你的数据已经被加载和准备好
file_path = '/Users/allen/Desktop/factor(before).xlsx'  # 替换为你的文件路径
data_clean = pd.read_excel(file_path)
data_clean['Year'] = pd.to_datetime(data_clean['Year'], format='%Y')  # 确保年份格式正确
data_clean.set_index('Year', inplace=True)

# 定义训练数据
training_data = data_clean['CO2 change']

# 定义ARIMA模型配置
# 我们通常通过使用ACF和PACF图来选择p, d, q参数，这里我们从一些常见设置开始
p = 1  # 滞后阶数
d = 1  # 差分次数
q = 1  # 移动平均阶数

# 拟合ARIMA模型
arima_model = ARIMA(training_data, order=(p, d, q))
arima_result = arima_model.fit()

# 为接下来几年生成预测（比如从2019年到2024年）
forecast_years = 6
forecast_result = arima_result.get_forecast(steps=forecast_years)
forecast_ci = forecast_result.conf_int()

# 创建DataFrame以查看预测和置信区间
forecast_df = DataFrame({
    'Year': range(2019, 2019 + forecast_years),
    'Forecast': forecast_result.predicted_mean,
    'Lower CO2 Emission': forecast_ci.iloc[:, 0],
    'Upper CO2 Emission': forecast_ci.iloc[:, 1]
})

print(forecast_df)


# In[8]:


import matplotlib.pyplot as plt
import pandas as pd

# 更新的数据定义
data = {
    'Year': pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01']),
    'Forecast': [120.831696, 120.472780, 120.352196, 120.311683, 120.298072, 120.293499],
    'Lower CO2 Emission': [118.282611, 115.486159, 113.382475, 111.695698, 110.269863, 109.018646],
    'Upper CO2 Emission': [123.380781, 125.459401, 127.321916, 128.927668, 130.326281, 131.568353]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 计算误差
df['Lower Error'] = df['Forecast'] - df['Lower CO2 Emission']
df['Upper Error'] = df['Upper CO2 Emission'] - df['Forecast']
errors = [df['Lower Error'].values, df['Upper Error'].values]

# 绘制误差条图
plt.figure(figsize=(10, 6))
plt.errorbar(df['Year'], df['Forecast'], yerr=errors, fmt='o', ecolor='red', capsize=5, linestyle='-', color='blue', label='Forecast with error bars')
plt.title('CO2 Emission Forecasts with Error Bars (2020-2025)')
plt.xlabel('Year')
plt.ylabel('Forecast CO2 Emission')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()



# In[5]:


from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
import pandas as pd

# 假设data_clean是你已经加载和清洗过的数据
# 这里仅为示例，确保你的数据已经被加载和准备好
file_path = '/Users/allen/Desktop/factor(before).xlsx'  # 替换为你的文件路径
data_clean = pd.read_excel(file_path)
data_clean['Year'] = pd.to_datetime(data_clean['Year'], format='%Y')  # 确保年份格式正确
data_clean.set_index('Year', inplace=True)

# 定义训练数据
training_data = data_clean['EV change(Thousands)']

# 定义ARIMA模型配置
# 我们通常通过使用ACF和PACF图来选择p, d, q参数，这里我们从一些常见设置开始
p = 1  # 滞后阶数
d = 1  # 差分次数
q = 1  # 移动平均阶数

# 拟合ARIMA模型
arima_model = ARIMA(training_data, order=(p, d, q))
arima_result = arima_model.fit()

# 为接下来几年生成预测（比如从2019年到2024年）
forecast_years = 6
forecast_result = arima_result.get_forecast(steps=forecast_years)
forecast_ci = forecast_result.conf_int()

# 创建DataFrame以查看预测和置信区间
forecast_df = DataFrame({
    'Year': range(2019, 2019 + forecast_years),
    'Forecast': forecast_result.predicted_mean,
    'Lower EV number': forecast_ci.iloc[:, 0],
    'Upper EV number': forecast_ci.iloc[:, 1]
})

print(forecast_df)


# In[10]:


import matplotlib.pyplot as plt
import pandas as pd

# 数据定义
data = {
    'Year': pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01']),
    'Forecast': [3015.538178, 3563.742316, 4103.737214, 4635.645804, 5159.589176, 5675.686608],
    'Lower CO2 Emission': [2940.634654, 3335.963227, 3683.499209, 3990.792912, 4262.919575, 4503.517372],
    'Upper CO2 Emission': [3090.441702, 3791.521405, 4523.975219, 5280.498695, 6056.258776, 6847.855843]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 计算误差
df['Lower Error'] = df['Forecast'] - df['Lower CO2 Emission']
df['Upper Error'] = df['Upper CO2 Emission'] - df['Forecast']
errors = [df['Lower Error'].values, df['Upper Error'].values]

# 绘制误差条图
plt.figure(figsize=(10, 6))
plt.errorbar(df['Year'], df['Forecast'], yerr=errors, fmt='o', ecolor='red', capsize=5, linestyle='-', color='blue', label='Forecast with error bars')
plt.title('The number of EV change with Error Bars (2020-2025)')
plt.xlabel('Year')
plt.ylabel('The number of Electric Vehicle(Thousand)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件中的数据
file_path = '/Users/allen/Desktop/factor table（after）.xlsx'
data = pd.read_excel(file_path)

# 绘制CO2变化的可视化图
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['CO2 change'], color='green', marker='o', linestyle='-', label='CO2 Change')
plt.title('Change in CO2 Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('CO2 Change')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the sample data
data = {
    'Year': np.arange(2010, 2023),
    'Trips': np.linspace(100000, 120000, 13),  # Public transit trips
    'AvgTripLength': np.random.normal(5, 0.5, 13),  # Average trip length in km
    'ServiceFrequency': np.linspace(0.8, 1, 13),  # Frequency as a fraction of normal
    'EV_Number': np.linspace(1000, 5000, 13),  # Number of EVs
    'EnergyPerEV': np.random.normal(0.2, 0.05, 13),  # kWh per km per EV
    'RenewableEnergyFactor': np.linspace(0.5, 0.8, 13),  # Proportion of renewable energy
    'Population': np.linspace(8000000, 9000000, 13),
    'PerCapitaEmissions': np.random.normal(2, 0.1, 13)  # Tons of CO2 per person per year
}

df = pd.DataFrame(data)

# Calculate influences
df['PTI'] = df['Trips'] * df['AvgTripLength'] * df['ServiceFrequency']
df['EVI'] = df['EV_Number'] * df['EnergyPerEV'] * df['RenewableEnergyFactor']
df['PD'] = df['Population'] * df['PerCapitaEmissions']

# Weight factors (example weights)
weights = {'PTI': 0.4, 'EVI': 0.3, 'PD': 0.3}

# Calculate total CO2 emissions
df['Total_CO2'] = (df['PTI'] * weights['PTI'] +
                   df['EVI'] * weights['EVI'] +
                   df['PD'] * weights['PD'])

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(df['Year'], df['Total_CO2'], marker='o', linestyle='-')
plt.title('Estimated CO2 Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('Total CO2 Emissions')
plt.grid(True)
plt.show()


# In[ ]:




