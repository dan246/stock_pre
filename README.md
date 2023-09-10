# 股票預測demo
這個Demo展示了如何使用Streamlit建立一個簡單的股票預測應用程式。由於模型過大無法直接上傳到GitHub，因此我們使用Teachable Machine來訓練代替模型。

如何運行：

要運行這個Demo，您可以點擊以下連結：股票預測 Demo

請注意，您需要有網絡連接才能訪問Demo。

專案結構：

這個專案包含以下步驟：

使用stockdata庫處理圖像數據，並將其存儲在imagestock資料夾中。

使用Keras模型訓練處理後的數據。

使用stock_image_class_examV2來驗證模型性能。

最終，在Streamlit上部署應用程式，讓用戶能夠輕鬆預測台灣股市。

技術堆疊：

Streamlit: 用於建立互動性股票預測應用程式的Python庫。
Teachable Machine: 用於模型訓練的在線機器學習工具。
Keras: 用於構建和訓練深度學習模型的深度學習庫。
注意事項
請注意，這只是一個測試Demo，模型使用Teachable Machine代替。在實際情況下，您可能需要使用更大的數據集和更強大的模型來進行台灣股市的預測。
