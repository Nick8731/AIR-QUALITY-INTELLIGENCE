"""五頁繁體中文介面；所有數字均來自監測 CSV 或已儲存模型。"""
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import latest_observation
from src.inference import predict
from src.preprocess import clean, normalize, read_csv
from src.multistation import registry, entry, station_root, load_history, available
from src.forecasting import forecast as audited_forecast, context_window, input_csv, backtest
import hashlib

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='AirPredict AI | PM2.5', page_icon='🌤️', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;600;700&display=swap');
html,body,[class*="css"] {font-family:'Noto Sans TC',sans-serif;}
.block-container {padding-top:4rem;max-width:1450px;}
[data-testid="stMetric"] {background:white;border:1px solid #e2eaf5;border-radius:14px;padding:18px;}
[data-testid="stSidebar"] {border-right:1px solid #e2eaf5;}
.eyebrow {color:#2563eb;font-size:12px;font-weight:700;letter-spacing:2px;margin-bottom:10px;}
.hero {background:linear-gradient(110deg,#102e59,#175bb0);color:white;padding:28px 32px;border-radius:18px;margin-bottom:24px;}
.hero h1 {color:white;font-size:32px;margin:0;}
.hero p {color:#cfe2fc;margin:12px 0 0;}
@media (max-width:900px) {
  [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {min-width:calc(50% - 1rem);flex:1 1 calc(50% - 1rem)!important;}
  [data-testid="stMetricValue"] {font-size:1.65rem;}
}
@media (max-width:600px) {
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {min-width:100%;flex:1 1 100%!important;}
  .hero {padding:22px;}
  .hero h1 {font-size:28px;}
}
</style>''', unsafe_allow_html=True)


def line_chart(frame, result=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.datetime, y=frame.pm25, mode='lines', name='真實 PM2.5',
                             line={'color': '#2563eb', 'width': 2}, connectgaps=False,
                             fill='tozeroy', fillcolor='rgba(37,99,235,0.07)'))
    if result:
        fig.add_trace(go.Scatter(x=[frame.datetime.iloc[-1], pd.Timestamp(result['prediction_time'])],
                                 y=[frame.pm25.iloc[-1], result['pm25']], name='AI 下一小時預測',
                                 mode='lines+markers', line={'color': '#14b8a6', 'dash': 'dash', 'width': 3},
                                 marker={'size': 9}))
    fig.update_layout(template='plotly_white', height=350, margin={'l': 10, 'r': 15, 't': 15, 'b': 10},
                      yaxis_title='PM2.5（μg/m³）', xaxis_title='臺灣時間',
                      legend={'orientation': 'h', 'y': 1.12}, hovermode='x unified',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def forecast(frame, station, model):
    try:
        result, _ = audited_forecast(ROOT, selected_entry['key'], model.lower(), frame)
        return result, None
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        return None, str(exc)


def show_result(result, error):
    if error:
        st.warning(error)
        return
    st.metric('下一小時 AI 預測', f"{result['pm25']:.2f} μg/m³")
    st.caption(f"完整推論數值：{result['pm25']!r} μg/m³；卡片顯示至小數二位。")
    st.caption(f"預測時間：{result['prediction_time']} · {result['model']} · 臺灣時間")
    if result['pm25'] < 0:
        st.warning('模型輸出為負值，超出物理濃度範圍。此結果不適合用於實際判斷。')
    st.caption('AI 模型預測，非官方空氣品質預報。')
    interval = result.get('prediction_interval', {})
    if interval.get('available'):
        lower_col, upper_col = st.columns(2)
        lower_col.metric('名目 95% 預測區間下限', f"{interval['lower']:.2f} μg/m³")
        upper_col.metric('名目 95% 預測區間上限', f"{interval['upper']:.2f} μg/m³")
        st.caption(f"校準完整日數：{interval['calibration_samples']}；獨立評估小時覆蓋率：{interval['evaluation_coverage']:.2%}；平均寬度：{interval['evaluation_mean_width']:.2f} μg/m³")
        st.info(interval['warning'])
        if interval['lower'] < 0:
            st.caption('對稱區間保留原始數學下限；低於 0 不代表存在負的 PM2.5 濃度。')
    else:
        st.caption('預測區間：' + interval.get('reason', '尚未完成獨立驗證。'))


with st.sidebar:
    st.markdown('### 🌤️ AirPredict AI')
    st.caption('PM2.5 智慧空氣品質預測平台')
    st.divider()
    page = st.radio('功能導覽', ['首頁儀表板', 'AI 預測', '模型比較', '歷史資料分析', '關於專題'], label_visibility='collapsed')
    st.divider()
    upload = st.file_uploader('匯入真實監測 CSV', type=['csv'], help='可用官方日寬表，或 station,datetime,pm25 長表。')
    st.caption('上傳資料只在本次操作使用，不會覆蓋訓練資料與模型。')

data = pd.DataFrame()
source_label = '尚無資料'
with st.sidebar:
    county = st.selectbox('選擇縣市', [e['county'] for e in registry()], key='county_choice')
    selected_entry = entry(county)
    station = st.selectbox('監測站', [selected_entry['station']], key='station_' + selected_entry['key'])
    model_name = st.selectbox('預測模型', ['LSTM', 'RNN'])
    include_newer = st.checkbox('包含 2026 校準／評估觀測', value=True)
    st.caption(f"{county} → {station}測站（站號 {selected_entry['siteid']}）")
    if selected_entry.get('note'):
        st.caption(selected_entry['note'])
MODEL_ROOT = station_root(ROOT, selected_entry['key'])
model_ready, model_reason = available(ROOT, selected_entry['key'])
if not model_ready:
    st.warning(model_reason)
try:
    if upload is not None:
        normalized = normalize(read_csv(upload))
        data, upload_quality = clean(normalized, station)
        source_label = '使用者匯入 CSV（依縣市篩選測站）'
    else:
        data = load_history(ROOT, selected_entry['key'], include_research=include_newer)
        source_label = '環境部 AQX_P_15 歷史資料'
except (ValueError, OSError, KeyError, TypeError) as exc:
    st.error(f'無法讀取此測站資料：{exc}')


with st.sidebar:
    st.divider()
    st.caption('官方最新觀測（獨立於歷史回放）')
    if st.button('讀取官方最新觀測', width='stretch'):
        try:
            row = latest_observation(station)
            value = pd.to_numeric(row['pm25'], errors='coerce')
            if pd.isna(value) or value < 0:
                st.warning('官方最新 PM2.5 為缺值。')
            else:
                st.metric('官方最新 PM2.5', f'{value:g} μg/m³')
                st.caption(f"官方發布：{row['datetime']}")
            st.info('單筆觀測不足以預測，請匯入最近連續 24 小時資料。')
        except ValueError as exc:
            st.info(str(exc))
    st.caption('PyTorch · RNN / LSTM\n\n資料時間：Asia/Taipei')

st.markdown('<div class="eyebrow">AIR QUALITY INTELLIGENCE / 空氣品質智慧分析</div>', unsafe_allow_html=True)
if page == '首頁儀表板':
    st.markdown('<div class="hero"><h1>AirPredict AI</h1><p>PM2.5 智慧空氣品質預測平台 · 從每小時觀測，看見下一步變化。</p></div>', unsafe_allow_html=True)
else:
    st.title(page)

if not data.empty:
    latest_time = pd.Timestamp(data.datetime.max())
    now = pd.Timestamp.now(tz='Asia/Taipei').tz_localize(None)
    age = now - latest_time
    mode = '歷史資料回放' if age > pd.Timedelta(hours=3) else '近期觀測資料'
    st.info(f'{mode} ｜ {source_label} ｜ {county}／{station} ｜ 最後資料時間：{latest_time:%Y-%m-%d %H:%M}（臺灣時間）')

if page in ['首頁儀表板', 'AI 預測', '歷史資料分析'] and data.empty:
    st.warning('此測站尚無已驗證資料。請上傳對應站的 CSV，或參考 EXTENSIONS.md 執行該站資料下載。')
    st.stop()

if page == '首頁儀表板':
    valid = data.dropna(subset=['pm25'])
    columns = st.columns(4)
    current = data.iloc[-1]
    columns[0].metric('最後時段觀測 PM2.5', '缺值' if pd.isna(current.pm25) else f'{current.pm25:g} μg/m³')
    columns[1].metric('最近 24 小時有效筆數', f"{data.tail(24).pm25.notna().sum()} / 24")
    columns[2].metric('資料有效率', f'{data.pm25.notna().mean():.2%}')
    columns[3].metric('所使用模型', model_name)
    st.subheader('最近 7 天濃度變化')
    result, error = forecast(data, station, model_name)
    st.plotly_chart(line_chart(data.tail(168), result), width='stretch')
    left, right = st.columns([1, 2])
    with left:
        show_result(result, error)
    with right:
        st.markdown('#### 以真實觀測建立預測')
        st.write('模型使用過去連續 24 小時的 PM2.5，推估接下來一小時。任何輸入缺值都會阻止預測，並保留原因。')
        st.caption('比較頁面的指標固定來自已保存的訓練測試實驗，上傳資料不會改寫實驗結果。')

elif page == 'AI 預測':
    st.write(f'{county}／{station}測站：只使用此站模型與真實觀測。')
    operation = st.radio('預測模式', ['下一小時預測', '歷史回測'], horizontal=True)
    if operation == '下一小時預測':
        target_time = pd.Timestamp(data.datetime.max()) + pd.Timedelta(hours=1)
        st.write(f'下一小時目標時間：**{target_time:%Y-%m-%d %H:%M}**（臺灣時間）')
        try:
            recent, _ = context_window(data, station, target_time)
            payload = input_csv(recent)
            token = selected_entry['key'] + model_name + hashlib.sha256(payload).hexdigest()
            st.dataframe(recent[['datetime', 'station', 'pm25']], hide_index=True, width='stretch')
            st.caption('上表 24 筆即為推論原始輸入；JSON 另外保存相同逐筆紀錄與 CSV 雜湊。')
            st.download_button('下載 24 小時輸入 CSV', payload, f"{selected_entry['key']}_input24.csv", 'text/csv')
            if st.button('開始預測', type='primary', width='stretch', disabled=not model_ready):
                result, error = forecast(recent, station, model_name)
                st.session_state['audited_prediction'] = {'token': token, 'result': result, 'error': error}
            saved = st.session_state.get('audited_prediction', {})
            result = saved.get('result') if saved.get('token') == token else None
            error = saved.get('error') if saved.get('token') == token else None
            if result or error:
                show_result(result, error)
            st.plotly_chart(line_chart(recent, result), width='stretch')
            if result:
                st.download_button('下載本次預測 JSON', json.dumps(result, ensure_ascii=False, indent=2),
                                   file_name='pm25_prediction.json', mime='application/json')
        except ValueError as exc:
            st.dataframe(data.tail(24)[['datetime', 'station', 'pm25']], hide_index=True, width='stretch')
            st.warning(str(exc))
    else:
        c1, c2 = st.columns(2)
        chosen_day = c1.date_input('回測目標日期', value=data.datetime.max().date(),
                                   min_value=data.datetime.min().date(), max_value=data.datetime.max().date())
        chosen_hour = c2.selectbox('回測目標小時', list(range(24)), index=23)
        target_time = pd.Timestamp(chosen_day) + pd.Timedelta(hours=chosen_hour)
        st.caption('只擷取目標之前 24 小時；同時計算 RNN 與 LSTM。分組標示指原模型的時間歸屬。')
        if st.button('執行歷史回測', type='primary', disabled=not model_ready):
            try:
                table, recent = backtest(ROOT, selected_entry['key'], data, target_time)
                st.session_state['backtest_result'] = {'key':selected_entry['key'], 'target':str(target_time),
                                                       'table':table, 'recent':recent,
                                                       'source_hash':hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()}
            except (ValueError, OSError, RuntimeError) as exc:
                st.session_state.pop('backtest_result', None)
                st.warning(str(exc))
        saved = st.session_state.get('backtest_result')
        data_token = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
        if saved and saved['key']==selected_entry['key'] and saved['target']==str(target_time) and saved['source_hash']==data_token:
            table, recent = saved['table'], saved['recent']
            st.write('時段歸屬：**' + table['split'].iloc[0] + '**')
            if table['split'].iloc[0].startswith(('訓練','驗證')):
                st.warning('此時段參與訓練或選模，誤差不能當作獨立泛化效能。')
            if not table.verified_with_actual.all():
                st.warning('目標時間沒有有效真實觀測；只有模型預測，尚未經實際值驗證。')
            st.dataframe(table, hide_index=True, width='stretch')
            st.dataframe(recent, hide_index=True, width='stretch')
            fig = line_chart(recent)
            for row in table.itertuples():
                fig.add_trace(go.Scatter(x=[recent.datetime.iloc[-1], target_time], y=[recent.pm25.iloc[-1], row.prediction],
                                         mode='lines+markers', name=row.model, line={'dash':'dash'}))
            if table.verified_with_actual.all():
                fig.add_trace(go.Scatter(x=[target_time], y=[table.actual_pm25.iloc[0]], mode='markers', name='目標真實值', marker={'size':12,'color':'#ef4444'}))
            st.plotly_chart(fig, width='stretch')
            st.download_button('下載歷史回測 CSV', table.to_csv(index=False).encode('utf-8-sig'), 'historical_backtest.csv', 'text/csv')


elif page == '模型比較':
    metric_file = MODEL_ROOT / 'results/metrics.csv'
    if not metric_file.exists():
        st.warning('模型尚未建立：請依 EXTENSIONS.md 使用測站獨立訓練流程。')
    else:
        metadata = json.loads((MODEL_ROOT / 'models/metadata.json').read_text('utf-8'))
        st.caption(f"測站：{metadata['station']} ｜ 測試起點：{metadata['split']['test_start']} ｜ 共同測試樣本：{metadata['split']['counts']['test']:,}")
        if upload is not None:
            st.info('此頁為已保存的官方資料實驗結果；上傳 CSV 不會重新評估模型。')
        metrics = pd.read_csv(metric_file)
        st.dataframe(metrics.style.format({c: '{:.4f}' for c in ['MAE', 'MSE', 'RMSE', 'R2']}), hide_index=True, width='stretch')
        st.caption('MAE／RMSE 單位為 μg/m³；MSE 為 (μg/m³)²；R² 越接近 1 越好。測試期間沒有參與選模。')
        c1, c2 = st.columns(2)
        c1.image(str(MODEL_ROOT / 'results/model_comparison.png'), width='stretch')
        c2.image(str(MODEL_ROOT / 'results/error_distribution.png'), width='stretch')
        st.subheader('訓練與驗證 Loss')
        c1, c2 = st.columns(2)
        c1.image(str(MODEL_ROOT / 'results/training_loss.png'), width='stretch')
        c2.image(str(MODEL_ROOT / 'results/validation_loss.png'), width='stretch')
        st.subheader('實際值與模型預測')
        st.image(str(MODEL_ROOT / 'results/prediction_comparison.png'), width='stretch')
        with st.expander('較大誤差案例'):
            st.dataframe(pd.read_csv(MODEL_ROOT / 'results/largest_errors.csv'), hide_index=True, width='stretch')
        st.markdown('**Baseline** 延續最後一小時，簡單且難以被短期模型明顯超越。\n\n**RNN** 可學習短期變化，但對較長期依賴的記憶較弱。\n\n**LSTM** 透過記憶與閘門控制資訊，參數較多，效果仍需實際驗證。')
        st.download_button('下載效能表格', metric_file.read_bytes(), 'metrics.csv', 'text/csv')

        interval_folder = ROOT / 'research/intervals' / selected_entry['key']
        with st.expander('預測區間的獨立校準與評估紀錄'):
            for kind in ['rnn','lstm']:
                ipath = interval_folder / f'{kind}.json'
                if ipath.exists():
                    study = json.loads(ipath.read_text('utf-8'))
                    st.write(f"{kind.upper()}：{study['reason']}")
                    st.write(f"校準完整日 {study['calibration_complete_blocks']}；獨立評估 {study['evaluation_hours']} 小時；實測小時覆蓋率 {study['empirical_hourly_coverage']:.2%}；平均寬度 {study['mean_interval_width']:.2f} μg/m³。")
                    st.caption(study['warning'])
                else:
                    st.caption(f'{kind.upper()} 尚無獨立區間研究紀錄。')

elif page == '歷史資料分析':
    chosen = st.date_input('日期範圍', value=(data.datetime.min().date(), data.datetime.max().date()),
                           min_value=data.datetime.min().date(), max_value=data.datetime.max().date())
    if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
        selected = data[data.datetime.ge(pd.Timestamp(chosen[0])) &
                        data.datetime.lt(pd.Timestamp(chosen[1]) + pd.Timedelta(days=1))]
        if selected.empty or not selected.pm25.notna().any():
            st.warning('所選期間沒有有效觀測。')
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('平均濃度', f'{selected.pm25.mean():.2f} μg/m³')
            c2.metric('最高濃度', f'{selected.pm25.max():g} μg/m³')
            c3.metric('最低濃度', f'{selected.pm25.min():g} μg/m³')
            c4.metric('有效小時 / 全部小時', f'{selected.pm25.notna().sum():,} / {len(selected):,}')
            st.plotly_chart(line_chart(selected), width='stretch')
            st.caption('折線中斷處代表缺值，統計不包含缺值。')
            st.download_button('下載所選期間 CSV', selected.to_csv(index=False).encode('utf-8-sig'),
                               f'pm25_{station}.csv', 'text/csv')
    else:
        st.info('請完成起訖日期選擇。')

elif page == '關於專題':
    st.caption('多縣市擴充：五個縣市各自對應一站；新增站分開訓練，彰化正式模型與原報告保持不變。回測位於 AI 預測頁，區間與搜尋紀錄保存於 research/。')
    st.markdown('''### 研究動機
空氣污染濃度會隨時間變化。本專題以環境部真實監測資料，驗證循環神經網路能否對下一小時濃度提供有用的估計。

### 從觀測到預測
1. 取得各代表測站 2024～2025 年官方逐日小時資料，轉成每小時長表。
2. 保留缺值紀錄、補齊時間索引；不補零、不使用未來觀測內插。
3. 依目標時間分成 70% 訓練、15% 驗證、15% 測試，只以訓練資料計算標準化參數。
4. Sliding Window 將連續 24 小時組成 `(batch, 24, 1)`，預測第 25 小時。
5. RNN 將資訊逐時間步傳遞；LSTM 額外使用記憶單元及閘門，控制保留與遺忘。
6. 使用 Adam 與 MSE 訓練，驗證集選擇最佳權重並提前停止，最後以共同測試集比較三個模型。

### 模型限制
每個模型只有該站 PM2.5 單一特徵，未納入氣象、排放或跨站特徵。此實驗為一小時單步預測，不能直接代表多小時預報；季節與資料分布變化也會影響結果。神經網路不保證優於 Persistence Baseline。

### 資料來源
- [環境部 AQX_P_15 歷史小時值](https://data.moenv.gov.tw/dataset/detail/AQX_P_15)
- [環境部 AQX_P_432 最新空氣品質指標](https://data.moenv.gov.tw/dataset/detail/aqx_p_432)
- [PyTorch 官方文件](https://pytorch.org/docs/stable/index.html)

所有結果均為 **AI 模型預測，非官方空氣品質預報**。歷史資料回放的預測時間緊接在最後一筆觀測之後，不會被當作今天的預報。
''')

st.divider()
st.caption('AirPredict AI · 基於 PyTorch 與 RNN 之 PM2.5 空氣品質預測系統 · 大學期中專題')
