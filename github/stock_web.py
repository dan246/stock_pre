# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 15:39:29 2023

@author: user
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator#, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pathlib
from bs4 import BeautifulSoup
import logging
import shutil
from tensorflow import keras
import numpy as np
from PIL import Image
import mplfinance as mpf
import io


#%% 標題

st.title('股票價格預測')
st.sidebar.info('歡迎使用股票價格預測網站。 在下列選擇你要的選項')
st.sidebar.info('投資有風險，入市需謹慎')
st.sidebar.info('歡迎加入韭菜團')
st.sidebar.info('網站建造者為 Daniel.T.Li')


def main():
    option = st.sidebar.selectbox('選擇功能', ['視覺化','近期數據', '預測'])
    if option == '視覺化':
        tech_indicators()
    elif option == '近期數據':
        dataframe()
    else:
        predict()

#%%  查詢功能

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = st.sidebar.text_input('輸入股票代碼', value='2330.TW')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('請輸入時間範圍(建議60天)', value=60)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('開始時間', value=before)
end_date = st.sidebar.date_input('結束時間', today)
if st.sidebar.button('送出'):
    if start_date < end_date:
        st.sidebar.success('開始時間: `%s`\n\n結束時間: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('錯誤：結束日期必須在開始日期之後')




data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.header('技術指標')
    option = st.radio('選擇要視覺化的技術指標', ['收盤價(Close)', '布林通道(BB)', '指數移動平均(EMA)'])

    # BB
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # 創建一個新的數據框
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    # macd = MACD(data.Close).macd()
    # RSI
    # rsi = RSIIndicator(data.Close).rsi()
    # # SMA
    # sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == '收盤價(Close)':
        st.write('收盤價')
        st.line_chart(data.Close)
    elif option == '布林通道(BB)':
        st.write('布林通道')
        st.line_chart(bb)
    else:
        st.write('指數移動平均')
        st.line_chart(ema)


def dataframe():
    st.header('近期資料')
    data_subset = data.iloc[::5]
    st.dataframe(data_subset)

#%% 模型預測

def predict():
    # model = st.radio('選擇模型', ['線性回歸', '隨機森林回歸', '極端隨機森林回歸', 'K最近鄰回歸', 'XGBoost'])
    model = st.radio('選擇模型', ['線性回歸(短期預測)', 'K最近鄰回歸(短期預測)', 'XGBoost(短期預測)','keras_model(長期預測)'])
    
    model_kera = keras.models.load_model('github/keraV2.h5', compile=False)
    
    num = st.number_input('要預測幾天？', value=5)
    num = int(num)
    input_width  = 224
    input_height = 224    
    
    def preprocess_image(image):
        image = image.resize((input_width, input_height))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
    
    if st.button('預測'):
        if model == '線性回歸(短期預測)':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'K最近鄰回歸(短期預測)':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        elif model == 'XGBoost(短期預測)':
            engine = XGBRegressor()
            model_engine(engine, num)
        else:
            if not data.empty:
                #  mplfinance 版本
                custom_style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=' ', y_on_right=False, rc={'axes.xmargin': 0})
                try:
                    fig, _ = mpf.plot(data, type='candle', ylabel="price($)", axisoff=True, style=custom_style, returnfig=True)
                    st.subheader('當下股價圖')
                except Exception as e:
                    st.error(f"生成股價圖時發生錯誤：{e}")
                
                # 將圖表轉換為圖片
                img_stream = io.BytesIO()
                fig.savefig(img_stream, format='png')
                img = Image.open(img_stream)
                st.image(img)
                
                close_prices = data['Close'].values  # 將數據轉成 NumPy Array
                input_image = Image.fromarray(close_prices, 'RGB')  # RGB 
                preprocessed_data = preprocess_image(input_image)
                
                # 預測
                prediction = model_kera.predict(preprocessed_data)
                
                # 預測對應到漲幅範圍標籤
                amplitude_ranges = ['-16', '-9', '-5', '-2', '0', '1', '4', '6', '10', '14', '23', '999']
                predicted_amplitude_index = np.argmax(prediction)  # 將這個替換成預測漲幅的index
        
                if predicted_amplitude_index is not None and 0 <= predicted_amplitude_index < len(amplitude_ranges):
                    predicted_amplitude = amplitude_ranges[predicted_amplitude_index]
        
                    # 顯示預測結果
                    st.subheader('模型預測結果')
                    st.write(f"預測價格幅度: {predicted_amplitude}")
        
                    if predicted_amplitude == '0':
                        st.info("預測：價格將保持不變")
                    elif predicted_amplitude.startswith('-'):
                        st.error(f"預測：價格將在兩個月後下跌超過 {predicted_amplitude} %")
                    else:
                        st.success(f"預測：價格將在兩個月後上漲超過 {predicted_amplitude} %")
                else:
                    st.warning("無法進行預測，請確認模型和數據準備正確")
                    
                    
        


def model_engine(model, num):
    # 收盤價
    df = data[['Close']]
    # 根據預測天數調整收盤價
    df['preds'] = data.Close.shift(-num)
    # 縮放數據
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    
    if len(x) <= num:
        st.warning("數據不足以支持所選的預測天數，請選擇更短期的預測。")
        return
    
    # 存儲最後 num_days 數據
    x_forecast = x[-num:]
    # 選擇訓練所需的值
    x = x[:-num]
    # 取得 preds 列
    y = df.preds.values
    # 選擇訓練所需的值
    y = y[:-num]

    # 分割數據
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    
    if len(x_train) == 0:
        st.warning("訓練數據不足，請調整劃分參數或選擇更短期的預測。")
        return

    # 訓練模型
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    # st.text(f'r2_score: {r2_score(y_test, preds)} \
    #         \nMAE: {mean_absolute_error(y_test, preds)}')
    
    # 根據天數預測股票價格
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


#%%

if __name__ == '__main__':
    main()
