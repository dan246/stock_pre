# 股票價格預測

這個Demo展示了如何使用Streamlit構建一個簡單的股票價格預測應用程序。由於模型過大無法直接上傳到GitHub，因此我們使用Teachable Machine來代替模型。

## 開始

要使用此應用程序，您可以點擊以下連結以訪問它：[股票價格預測 Demo](https://stockpre-e5jvrmkct9gy45v8ltqwyz.streamlit.app/)

請注意，您需要網絡連接才能訪問Demo。

## 功能

### 視覺化

此功能允許您在圖表中查看不同的技術指標，包括收盤價、布林通道和指數移動平均。

### 近期數據

此功能顯示了最近的股票數據，讓您可以查看特定時間範圍內的價格。

### 預測

此功能允許您使用不同的機器學習模型來預測股票價格。您可以選擇線性回歸、K最近鄰回歸、XGBoost或長期預測模型（Keras模型）。

## 如何運行

要執行此應用程序，您需要安裝所需的Python庫。您可以使用以下命令安裝它們：

```shell
pip install streamlit pandas yfinance ta-lib datetime scikit-learn xgboost matplotlib pillow mplfinance
```
1. 使用stockdata庫處理圖像數據，並將其存儲在imagestock資料夾中。

2. 使用Keras模型訓練處理後的數據。

3. 使用stock_image_class_examV2來驗證模型性能。

4. 最終，在Streamlit上部署應用程式，讓使用者能夠輕鬆預測股票價格。
