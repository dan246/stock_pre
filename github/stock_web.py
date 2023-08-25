# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:42:24 2023

@author: user
"""
import streamlit as st
from tensorflow import keras
import numpy as np
import datetime
import yfinance as yf
from PIL import Image
import mplfinance as mpf
import io


def main():
    st.title('股票價格預測')
    st.sidebar.info('歡迎使用股票漲幅預測網站。')
    st.sidebar.info('投資有風險，入市需謹慎')
    st.sidebar.info('歡迎加入韭菜團')
    st.sidebar.info('海水退了就知道誰沒穿褲子')
    st.sidebar.info('支援各國股票代碼')
    st.sidebar.info('本網站之創建者為 Daniel.T.Li。')
    st.sidebar.info('funwriter no girlfriend')
    
    # 載入keras模型
    model = keras.models.load_model('keraV2.h5')
    #圖片size
    input_width =   224
    input_height =  224
    
    def preprocess_image(image):
        image = image.resize((input_width, input_height))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
    def predict(stock_code):
        # 股價數據
        end_date = datetime.date.today()  # 當前日期
        start_date = end_date - datetime.timedelta(days=60)  # 60天前
        data = yf.download(stock_code, start=start_date, end=end_date, progress=False)
        
    
        
        if not data.empty:
            # 畫股價圖 streamlit 版
            st.subheader('當下股價圖')
            st.line_chart(data['Close'])
            st.line_chart(data)
            #  mplfinance 版本
            custom_style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=' ', y_on_right=False, rc={'axes.xmargin': 0})
            fig, _ = mpf.plot(data, type='candle', style=custom_style, volume=True, returnfig=True)
 
            # 將圖表轉換為圖片
            img_stream = io.BytesIO()
            fig.savefig(img_stream, format='png')
            img = Image.open(img_stream)
            st.image(img)
            # 預測
            # 用 data['Close'] 為预测数据，轉成圖像格式進行預測
            #close_prices = data.values
            
            close_prices = data['Close'].values  # 將數據轉成 NumPy Array
            input_image = Image.fromarray(close_prices, 'RGB')  # RGB 
            preprocessed_data = preprocess_image(input_image)
    
            # 預測
            prediction = model.predict(preprocessed_data)
    
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
                    st.error(f"預測：價格將在兩個月後下跌 {predicted_amplitude} 個單位")
                else:
                    st.success(f"預測：價格將在兩個月後上漲 {predicted_amplitude} 個單位")
            else:
                st.warning("無法進行預測，請確認模型和數據準備正確")
    
    
    # 股票代碼輸入框
    stock_code = st.text_input('輸入股票代碼（ex. 2330.TW）')
    
    if st.button('預測'):
        predict(stock_code)
        



#%%
if __name__ == '__main__':
    main()