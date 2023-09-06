#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import mplfinance as mpf
from FinMind.data import DataLoader
import csv
import numpy as np
import time
import os
from os import walk
from os.path import join
import random

# Function to fetch data for a given stock code
def fetch_stock_data(stock_id, start_date='2022-01-01'):
    api = DataLoader()
    api.login(user_id='ftes0980', password='asdfr235')
    stock_data = api.taiwan_stock_daily(stock_id, start_date=start_date)
    stock_data.rename(columns={'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close'}, inplace=True)
    stock_data.index = pd.to_datetime(stock_data.date)
    stock_data = stock_data.drop(['date', 'stock_id', 'Trading_Volume', 'Trading_money', 'spread', 'Trading_turnover'], axis=1)
    return stock_data

# Set parameters
path = r'./imagestock/'

#設定抓取的天數 85天 報酬率 50天
DayM = 85
DayN = 50

#抽樣N次
N = 100
start_date = '2022-01-01'

# 創建您想要獲取數據的股票代碼列表
# stock_ids = ['0050', '2330', '2317', '2454']  # Add more stock codes as needed
stock_ids=[
'2330',
'2317',
'2454',
'2412',
'6505',
'2308',
'2881',
'2882',
'1303',
'2303',
'1301',
'2886',
'2002',
'2891',
'3711',
'1216',
'1326',
'2884',
'5880',
'3045',
'2207',
'2892',
'5871',
'2603',
'2382',
'2880',
'2395',
'2885',
'2912',
'1101',
'3008',
'4904',
'3034',
'5876',
'1590',
'3037',
'2883',
'2609',
'2408',
'2357',
'2327',
'2887',
'6669',
'2890',
'2801',
'4938',
'1605',
'2379',
'6415',
'2615',
'8454',
'8046',
'2301',
'1402',
'9910',
'2345',
'1102',
'6409',
'2618',
'3231',
'1476',
'3443',
'2409',
'2888',
'2474',
'3481',
'2377',
'2105',
'6770',
'2356',
'2610',
'9945',
'2834',
'4958',
'2324',
'2344',
'2347',
'8464',
'3533',
'1504',
'9904',
'2353',
'9941',
'1229',
'2027',
'3661',
'3702',
'2376',
'2049',
'2633',
'3023',
'2360',
'2201',
'5269',
'2354',
'2371',
'2385',
'2542',
'9921',
'2812',
'6239',
'6592',
'2352',
'6789',
'2915',
'1795',
'3653',
'3044',
'3017',
'2337',
'1802',
'3036',
'6781',
'1722',
'2227',
'2838',
'3532',
'2449',
'4919',
'2634',
'2383',
'1477',
'6176',
'1503',
'2313',
'9914',
'1907',
'3189',
'9917',
'2368',
'1513',
'5522',
'2637',
'1434',
'2498',
'2492',
'2606',
'1210',
'2845',
'6531',
'2204',
'2206',
'2889',
'3665',
'2006',
'8478',
'3406',
'6550',
'2404',
'3035',
'2923',
'2015',
'2059',
'2809',
'1314',
'1717',
'6285',
'1227',
'2388',
'6116',
'3005',
'2014',
'2707',
'1904',
'3706',
'3714',
'5434',
'1808',
'6412',
'3576',
'9933',
'8926',
'4763',
'2504',
'9907',
'1773',
'2101',
'2903',
'2441',
'6269',
'2451',
'6491',
'1304',
'2023',
'6005',
'4961',
'3592',
'2607',
'6278',
'9939',
'2458',
'1409',
'2106',
'6670',
'1319',
'2897',
'2723',
'1440',
'2539',
'4915']
# 創建一個目錄來保存圖像
if not os.path.exists(path):
    os.makedirs(path)

# 設置初始數據框來存儲數據
X_df = pd.DataFrame([], columns=['date', 'price0', 'price85', 'price85_50', 'Xreturn'])

# 不帶網格且僅包含價格比例的自定義樣式
# custom_style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=' ', y_on_right=False, rc={'axes.xmargin': 0})
custom_style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=' ', y_on_right=False, rc={'axes.xmargin': 0})
labelname=[ '-16', '-9', '-5', '-2', '0', '1', '4', '6', '10', '14', '23', '999']
labelvalue=[-999, -16.617, -9.753, -5.622, -2.85, -0.524, 1.619, 4.008, 6.856, 10.048, 14.502, 23.612,999]

# 為每個不存在的類別創建目錄
for category_dir in labelname:
    category_dir = os.path.join(path, category_dir)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

# 處理每個起點的數據
for stock_id in stock_ids:
    # 獲取股票數據
    stock_data = fetch_stock_data(stock_id, start_date=start_date)

    # 計算數據長度
    lendata = len(stock_data)

    # 隨機選擇N個起點
    randomD = random.sample(range(0, lendata - DayN - DayM), N)

    # 創建一個列表來存儲新行
    new_rows = []

    # 處理每個起點的數據
    for i in randomD:
        
            
        trainX = stock_data[i:i + DayM]
        trainY = stock_data[i + DayM:i + DayM + DayN]
        X_return = np.log(trainY.Close[-1]) - np.log(trainX.Close[DayM - 1])
        X_returnClass = pd.cut([X_return*100], labelvalue, labels=labelname, include_lowest=True)#紀錄上界

        # 處理價格除以 0 的情況
        if not np.isfinite(X_return):  # 檢查是否為無窮大或無窮小的值
            X_return = 0  # 設置為 0 或其他合適的值，表示沒有返回率
        else:
            new_row = {
                'date': trainX.index[0],
                'price0': trainX.Close[0],
                'price85': trainX.Close[DayM - 1],
                'price85_50': trainY.Close[-1],
                'Xreturn': X_return,
                'XClass': X_returnClass[0]
            }
            new_rows.append(new_row)
            # 如果股票目錄不存在，則創建該目錄
            stock_dir = os.path.join(path, X_returnClass[0])
            if not os.path.exists(stock_dir):
                os.makedirs(stock_dir)
            #mpf.plot(trainX, type="candle", title=stock_id, ylabel="price($)",axisoff=True, style=custom_style,tight_layout=True,
            #         savefig=os.path.join(stock_dir, str(X_return.round(5) * 100) + '.png'))
            mpf.plot(trainX, type="candle", ylabel="price($)",axisoff=True, style=custom_style,tight_layout=True,
                     savefig=os.path.join(stock_dir, X_returnClass[0] +str(X_return)+ '.png'))
    
    # 使用 pandas.concat 將新行追加到 X_df
    if new_rows:
        X_df = pd.concat([X_df, pd.DataFrame(new_rows)], ignore_index=True)

# 將處理後的數據存到 CSV 文件
X_df.to_csv(path + 'processed_data.csv', index=False)


# 根據數據分為 12 類
#resultx = pd.cut(X_df.Xreturn*100, [-999, -100, -70, -40, -20, -3, 3, 20, 40, 70, 100, 999], labels=['-999', '-100', '-70', '-40', '-20', '-3', '3', '20', '40', '70', '100'], include_lowest=True)

#resultx = pd.Categorical(resultx)



# # 將圖像移動到各自的類別
# for stock_id in stock_ids:
#     stock_dir = os.path.join(path, stock_id)
#     images = [f for f in os.listdir(stock_dir) if f.endswith('.png')]
#     for image in images:
#         src = os.path.join(stock_dir, image)
#         # 將返回率（returns）由浮點數轉換成整數
#         return_value = float(image[:-4]) * 10
#         # 使用 np.arange() 創建等間距的 bins
#         bins = np.arange(resultx.codes.min(), resultx.codes.max() + 1)
#         # 使用 np.digitize() 將返回率映射到 bins 的範圍
#         category_idx = np.digitize(return_value, bins)
#         category_code = category_idx - 1
#         category = resultx.categories[category_code]

#         # Get the date from the image filename
#         date = image.split('.')[0]  # 假設圖片檔名是 "2021-09-13.png"，取 "2021-09-13"

#         # Create the new image name using stock code, date, and return value
#         new_image_name = f"{stock_id}-{date}-{str(return_value)}"

#         # 檢查目標目錄是否已經存在相同名稱的檔案
#         dest_dir = os.path.join(path, str(category))
#         dest = os.path.join(dest_dir, f"{new_image_name}.png")
#         counter = 1
#         while os.path.exists(dest):
#             # 在檔案名稱後面加上獨特的識別符號
#             dest = os.path.join(dest_dir, f"{new_image_name}_{counter}.png")
#             counter += 1

#         # 將檔案移到目標目錄
#         os.rename(src, dest)

# # 清理空目錄
# for stock_id in stock_ids:
#     stock_dir = os.path.join(path, stock_id)
#     if not os.listdir(stock_dir):
#         os.rmdir(stock_dir)


