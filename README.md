# AirPredict AI — PM2.5 智慧空氣品質預測平台

使用環境部真實彰化測站資料，以過去 24 小時 PM2.5 預測下一小時。包含 Persistence Baseline、PyTorch RNN／LSTM、訓練與評估、獨立推論及五頁繁體中文 Streamlit。所有預測均為 AI 模型預測，非官方空氣品質預報。

## Windows 快速啟動

在專案資料夾開啟 PowerShell：

```powershell
cd D:\Users\User\PM25_AI_Project
.\.venv\Scripts\python.exe -m streamlit run app.py
```

也可雙擊 `start.bat`。瀏覽器開啟 http://localhost:8501 。本次交付包含已訓練模型，可直接使用，不需重新訓練。

## 安裝環境

建議 Python 3.11 或以上；此次已驗證 Python 3.13.7。若搬到另一台電腦，需重建虛擬環境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本機可重用已安裝的 PyTorch CUDA 套件，建立 `.venv` 時使用 `python -m venv --system-site-packages .venv`。套件實際版本見 `results/environment.txt`。程式自動選 CUDA，沒有 GPU 時使用 CPU，也可加 `--device cpu`。

## 五個階段與重現指令

以下均在專案根目錄執行（可用 `.venv\Scripts\python.exe` 取代 `python`）。

```powershell
# 1. 官方下載、前處理、資料測試
python -m src.data_loader --station 彰化
python -m src.preprocess --station 彰化 --start 2024-01-01 --end "2025-12-31 23:00"
python -m pytest tests/test_preprocess.py -q
# 2–3. 訓練兩種模型，隨後自動評估並產圖
python -m src.train --epochs 100 --patience 10 --batch-size 32 --learning-rate 0.001
# 重新評估已儲存的最佳模型，不必訓練
python -m src.evaluate
# 獨立推論；預設使用歷史資料最後 24 小時
python -m src.inference --model lstm
python -m src.inference --model rnn --csv your_recent_24_hours.csv
# 4. 網頁系統
python -m streamlit run app.py
# 5. 所有測試；重建英文報告
python -m pytest -q
python -m src.report
```

重新訓練會更新 `models/` 和 `results/`。請先備份欲保留的實驗。不要依照測試集表現反覆調參；若要改超參數，應使用驗證集，並另外保留未使用的最終測試期間。

## 官方資料與下載方式

- 歷史來源：[AQX_P_15](https://data.moenv.gov.tw/dataset/detail/AQX_P_15)。實際檢查日為 2026-10-29。
- 最新觀測：[AQX_P_432](https://data.moenv.gov.tw/dataset/detail/aqx_p_432)。只提供最新觀測，單次回傳不足以建立最近 24 小時輸入。
- [官方 API 使用說明](https://data.moenv.gov.tw/paradigm)、[政府資料開放授權](https://data.gov.tw/license)。

歷史下載器使用官方網站前端公開的 `POST /api/frontstage/datastore.search`，搭配網站列出的月份資源 UUID（`src/resources.json`）、`sitename equal 彰化`、`itemengname equal PM2.5`；依 offset 分頁。此為網站端點，可能隨網站改版，不能保證為長期穩定 API。`data/raw/provenance.json` 記錄來源、擷取時間、資源 UUID、原始列數及每檔 SHA-256。下載內容保持官方日寬表格式，不製造資料。

若自動下載失敗：

1. 開啟 AQX_P_15 頁面，至「歷史資料下載」，按「顯示全部」。
2. 勾選 2024/01～2025/12，選 CSV 下載並解壓縮。
3. 將 CSV 放到 `data/raw/`。可包含全部測站與測項，前處理會篩選。
4. 執行 `python -m src.preprocess` 並檢查 `data/processed/quality.json`。

原始欄位已確認：`sitename`、`itemengname`、`monitordate`、`monitorvalue00`～`monitorvalue23`。也接受官方中文欄名、0～23 時欄名，以及 `station,datetime,pm25` 長表。支援 UTF-8 BOM 與 CP950。時間為臺灣本地整點；長表有時區時換成 Asia/Taipei。

最新 API：複製 `.env.example` 為 `.env`，設定自己的 `MOENV_API_KEY`，可在網頁按「讀取官方最新觀測」。不會把 Key 寫入程式或錯誤 URL。若要用最新資料預測，請上傳最近至少 24 小時的真實 CSV；單筆最新觀測不會被重複填成 24 小時。網頁顯示資料時間及歷史回放狀態。

## 清理與防止資料洩漏

- 先依測站及 PM2.5 篩選，再將日寬表轉成小時長表。
- 官方 `x`、`#`、`*` 等標記、空白、負值、無限值轉為缺值；不去掉註記後猜測數值。不補 0、不內插、不使用未來觀測補值。
- 零濃度保留。大於 500 的數值保留並在品質報告列計數，不因為高值就武斷刪除污染事件。
- 相同時間重複數值去重；不同有效值衝突時設為缺值。缺少整個時段會補齊時間索引並標示 `absent_hour`。官方異常觀測標示 `invalid_or_conflict`。
- 整段小時索引按時間 70%／15%／15% 切分。平均值及母體標準差只取訓練時段有效觀測。
- 視窗為 24 小時輸入＋下一小時目標，25 筆皆有效才保留，絕不跨缺口或測站。依目標時間分配資料集；驗證／測試可使用其目標之前已觀測的歷史。
- 測試為逐小時、單步 rolling forecast，各次輸入可使用先前真實觀測；不是連續多小時遞迴預測。
- 訓練使用 Adam、MSE、梯度裁切 1.0、固定種子 42、驗證集 Early Stopping；測試集不參與選模。RNN／LSTM 各用 2 層、hidden size 64。

## 專案輸出

`src/` 含資料下載、前處理、視窗、模型、訓練、評估、推論及報告產生程式。`app.py` 為五頁 Streamlit；`tests/` 使用明確標示的小型測試資料驗證邊界，不用於正式訓練或實驗指標。

- `data/raw/`：實際下載的月份 CSV、來源證據。
- `data/processed/pm25.csv`、`quality.json`：小時資料、缺值紀錄與品質統計。
- `models/rnn_model.pth`、`lstm_model.pth`：最佳 state_dict；`metadata.json` 含測站、架構、分割時間、標準化、種子、資料雜湊及訓練紀錄。
- `scaler.json` 供安全載入；依需求另附 `scaler.pkl`，推論不讀取 pickle。
- `results/metrics.csv`、`test_predictions.csv`、`largest_errors.csv`、`training_history.csv`、`evaluation.json`：可追溯實驗結果。
- `results/*.png` 與 `report/figures/*.png`：訓練／驗證 loss、實際與預測、MAE 比較、誤差分布。
- `report/MIDTERM_REPORT.md`：使用真實結果自動產生的英文報告。

## 使用限制

預設模型僅針對彰化訓練。若要使用二林、斗六，需另用相同流程訓練並保存至不同 `--root` 專案目錄；介面不會把彰化模型套到其他站。只有 PM2.5 單一特徵，未納入氣象與排放；時序季節性、概念漂移及極端事件會影響泛化。不要保證神經網路一定勝過 Baseline。沒有可信輸入或模型時會顯示原因，不產生假結果。模型未限制輸出範圍，若出現負值會提示超出物理範圍；評估保留原始輸出，避免暗中改動指標。

本系統為課堂專題，預測不是官方 AQI 或健康風險判定；μg/m³ 為 PM2.5 濃度，沒有直接當作 AQI 分數使用。
