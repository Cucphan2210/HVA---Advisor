import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from vnstock import Vnstock
from statsmodels.tsa.arima.model import ARIMA
import feedparser
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* Background */
    body {
        background-color: #0E1117;
    }

    /* Title */
    h1 {
        color: #ffffff;
        font-weight: 700;
    }

    /* Card style */
    .block-container {
        padding-top: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827, #0E1117);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    /* Metric box */
    .css-1xarl3l {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
    }

    /* Text input */
    input {
        border-radius: 10px !important;
    }

    /* Section title */
    h2, h3 {
        color: #00c6ff;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeInDown 1s ease-in-out;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 16px;
    margin-top: -10px;
    animation: fadeIn 2s ease-in-out;
}

@keyframes fadeInDown {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>

<div class="title">HVA ROBO ADVISOR AI</div>
<div class="subtitle">
Smart Investment • AI Powered • Real-time Analytics
</div>
""", unsafe_allow_html=True)

# ================= KYC =================
st.sidebar.header("👤 Hồ sơ khách hàng")
client_name = st.sidebar.text_input("Tên khách hàng")
age = st.sidebar.slider("Tuổi", 18, 70, 25)
risk_profile = st.sidebar.selectbox("Khẩu vị rủi ro", ["Thấp", "Trung bình", "Cao"], key="risk_profile")
capital = st.sidebar.number_input("Vốn đầu tư", value=100000000, key="capital")

# ===== KPI DASHBOARD =====
col1, col2, col3, col4 = st.columns(4)
col1.metric(" Vốn", f"{capital:,.0f} VND")
col2.metric(" Rủi ro", risk_profile)
col3.metric(" Tuổi", age)
col4.metric(" AI Score", "87/100")

# ================= INPUT =================
mode = st.radio("Chế độ", ["Phân tích 1 mã", "Danh mục đầu tư", "Phân tích cổ phiếu OTC","Phân tích danh mục OTC tự động"])
symbols_input = st.text_input("Nhập mã cổ phiếu (VD: FPT,HPG,VCB)")

# ================= DATA =================
def get_data(symbol):
    """Lấy dữ liệu cổ phiếu"""
    try:
        stock = Vnstock().stock(symbol=symbol, source='KBS')
        df = stock.quote.history(start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        
        if df is None or len(df) == 0:
            return None
        
        df = df[['time','close','volume']]
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.dropna()
        
        if len(df) < 30:
            return None
        
        df['Return'] = df['close'].pct_change()
        df['MA20'] = df['close'].rolling(window=20, min_periods=10).mean()
        df['MA50'] = df['close'].rolling(window=50, min_periods=25).mean()
        
        return df.dropna()
    except Exception as e:
        return None

# ================= FORECAST - FIXED với Monte Carlo =================
def forecast_monte_carlo(df, n_steps=30, n_simulations=200):
    """
    Dự báo giá bằng Monte Carlo Simulation
    - Random walk dựa trên historical returns
    - Có trend từ ARIMA
    - Kết quả có biến động thực tế, không phải đường thẳng
    """
    try:
        if df is None or len(df) < 30:
            return None, None, None, None
        
        prices = df['close'].values
        returns = df['Return'].dropna().values
        
        if len(prices) < 20 or len(returns) < 20:
            return None, None, None, None
        
        # Tính các tham số thống kê từ dữ liệu lịch sử
        mean_return = np.mean(returns[-60:]) if len(returns) >= 60 else np.mean(returns)
        std_return = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns)
        
        # Dự báo trend chính (xu hướng dài hạn) từ ARIMA hoặc Linear Regression
        try:
            model = ARIMA(prices, order=(2,1,2))
            model_fit = model.fit()
            trend_forecast = model_fit.forecast(steps=n_steps)
        except:
            # Fallback: Linear regression cho trend
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x[-min(60, len(prices)):], prices[-min(60, len(prices)):], 1)
            trend_forecast = [intercept + slope * (len(prices) + i) for i in range(n_steps)]
        
        last_price = prices[-1]
        
        # Monte Carlo Simulation với Random Walk + Trend
        all_simulations = []
        
        for _ in range(n_simulations):
            sim_path = [last_price]
            for step in range(n_steps):
                # Random shock từ phân phối chuẩn với mean và std từ lịch sử
                shock = np.random.normal(mean_return, std_return)
                # Giá mới = giá cũ * (1 + shock) + một phần nhỏ của trend
                # Trend chỉ ảnh hưởng 20% để tránh làm mất biến động
                trend_effect = (trend_forecast[step] - last_price) / n_steps * (step + 1) * 0.2
                new_price = sim_path[-1] * (1 + shock) + trend_effect
                # Đảm bảo giá không âm
                new_price = max(new_price, last_price * 0.5)
                sim_path.append(new_price)
            all_simulations.append(sim_path[1:])  # Bỏ giá đầu
        
        all_simulations = np.array(all_simulations)
        
        # Tính các percentiles
        forecast_mean = np.mean(all_simulations, axis=0)
        forecast_median = np.median(all_simulations, axis=0)
        forecast_upper = np.percentile(all_simulations, 85, axis=0)  # Kênh trên
        forecast_lower = np.percentile(all_simulations, 15, axis=0)  # Kênh dưới
        forecast_upper_95 = np.percentile(all_simulations, 97.5, axis=0)  # Kênh trên cùng
        forecast_lower_95 = np.percentile(all_simulations, 2.5, axis=0)   # Kênh dưới cùng
        
        last_date = df['time'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_steps)
        
        return future_dates, forecast_mean, forecast_median, (forecast_lower, forecast_upper), (forecast_lower_95, forecast_upper_95), all_simulations
    
    except Exception as e:
        print(f"Forecast error: {e}")
        return None, None, None, None, None, None

# ================= NEWS =================
def get_news(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+chứng+khoán&hl=vi&gl=VN&ceid=VN:vi"
    feed = feedparser.parse(url)

    news_list = []

    for entry in feed.entries[:5]:
        title = entry.title
        link = entry.link

        sentiment = "⚪ Trung tính"

        if any(x in title.lower() for x in ["tăng", "lãi", "bứt phá"]):
            sentiment = "🟢 Tích cực"
        elif any(x in title.lower() for x in ["giảm", "lỗ", "bán tháo"]):
            sentiment = "🔴 Tiêu cực"

        news_list.append((title, link, sentiment))

    return news_list if news_list else [("Không có tin tức mới", "#", "⚪ Trung tính")]

# ================= MPT =================
def optimize_mpt(returns):
    mean_returns = returns.mean()*252
    cov_matrix = returns.cov()*252

    num_assets = len(mean_returns)

    results = []
    weights_record = []

    for _ in range(5000):  # Tăng lên 5000 để có nhiều điểm hơn
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret/vol if vol != 0 else 0

        results.append((ret, vol, sharpe))
        weights_record.append(weights)

    results = np.array(results)
    best_idx = np.argmax(results[:,2])

    return results, weights_record[best_idx], mean_returns.index

# ================= FORECAST ARIMA CHO MARKET SCAN - FIXED =================
def forecast_arima_for_scan(symbol):
    """Dự báo đơn giản cho market scan - FIXED: trả về trend và vol trực tiếp"""
    try:
        stock = Vnstock().stock(symbol=symbol, source='KBS')
        df = stock.quote.history(start='2023-06-01', end=datetime.today().strftime('%Y-%m-%d'))
        
        if df is None or len(df) < 30:
            return None, None, None
        
        df = df[['time','close']]
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna()
        
        if len(df) < 30:
            return None, None, None
        
        # Tính returns
        df['Return'] = df['close'].pct_change()
        prices = df['close'].values
        
        # Dùng linear regression trên 30 ngày gần nhất để tính trend
        recent_prices = prices[-30:] if len(prices) >= 30 else prices
        x = np.arange(len(recent_prices))
        slope, intercept = np.polyfit(x, recent_prices, 1)
        
        # Dự báo 30 ngày
        last_price = prices[-1]
        forecast_30d = [intercept + slope * (len(recent_prices) + i) for i in range(30)]
        
        # Tính xu hướng
        trend_pct = (forecast_30d[-1] - last_price) / last_price if last_price > 0 else 0
        volatility = df['Return'].std()
        score = trend_pct / (volatility + 1e-6)
        
        return trend_pct, volatility, score
        
    except Exception as e:
        return None, None, None

# ================= MODE 1 =================
if mode == "Phân tích 1 mã" and st.button("🔍 Phân tích"):
    if not symbols_input.strip():
        st.error("Vui lòng nhập mã cổ phiếu")
        st.stop()
    
    sym = symbols_input.strip().upper()

    with st.spinner(f"Đang phân tích {sym}..."):
        df = get_data(sym)

    if df is None or len(df) < 20:
        st.error(f"Không đủ dữ liệu cho {sym}. Vui lòng thử mã khác (FPT, HPG, VCB, VNM, MWG...)")
        st.stop()

    current_price = df['close'].iloc[-1]
    
    st.header(f" {sym} | Giá hiện tại: {current_price * 1000:,.0f} VND")
    
    # ===== METRICS =====
    col1, col2, col3 = st.columns(3)
    col1.metric(" Giá hiện tại", f"{current_price * 1000:,.0f} VND")
    trend_text = " TĂNG" if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else " GIẢM"
    col2.metric(" Xu hướng", trend_text)
    col3.metric(" Biến động", f"{df['Return'].std()*100:.2f}%")

    # ===== SIGNAL =====
    df['Signal'] = 0
    df.loc[df['MA20'] > df['MA50'], 'Signal'] = 1
    df['Position'] = df['Signal'].diff()

    # ===== CHART =====
    st.subheader(" Biểu đồ xu hướng giá")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['close'] *1000,
        name="Giá", line=dict(width=3, color='#00c6ff')
    ))
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['MA20'] *1000,
        name="MA20", line=dict(dash='dash', width=2, color='#ffa500')
    ))
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['MA50'] *1000,
        name="MA50", line=dict(dash='dot', width=2, color='#00ff88')
    ))

    buy = df[df['Position'] == 1]
    sell = df[df['Position'] == -1]

    if len(buy) > 0:
        fig.add_trace(go.Scatter(
            x=buy['time'], y=buy['close'] *1000,
            mode='markers', name='🔵 BUY',
            marker=dict(color='#00ff88', size=12, symbol='triangle-up')
        ))
    if len(sell) > 0:
        fig.add_trace(go.Scatter(
            x=sell['time'], y=sell['close'] *1000,
            mode='markers', name='🔴 SELL',
            marker=dict(color='#ff4444', size=12, symbol='triangle-down')
        ))

    fig.update_layout(
        title=f"{sym} - Xu hướng giá với MA20/MA50",
        xaxis_title="Ngày",
        yaxis_title="Giá (VND)",
        template="plotly_dark",
        height=600,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ===== GIẢI THÍCH =====
    st.subheader(" Phân tích xu hướng")
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
        st.success(" **Xu hướng TĂNG - Golden Cross**\n\nMA20 nằm trên MA50, báo hiệu động lực tăng trưởng tích cực trong ngắn hạn.")
    else:
        st.error(" **Xu hướng GIẢM - Death Cross**\n\nMA20 nằm dưới MA50, áp lực bán đang chiếm ưu thế, cần thận trọng.")

    # ===== FORECAST - FIXED với Monte Carlo =====
    st.subheader(" Dự báo giá 30 ngày (Monte Carlo Simulation)")
    
    future_dates, forecast_mean, forecast_median, forecast_bands_70, forecast_bands_95, all_sims = forecast_monte_carlo(df)
    
    if future_dates is not None and forecast_mean is not None:
        # Biểu đồ dự báo
        fig2 = go.Figure()
        
        # Lịch sử 60 ngày
        fig2.add_trace(go.Scatter(
            x=df['time'].iloc[-60:], y=df['close'].iloc[-60:] *1000,
            name="Lịch sử (60 ngày)", line=dict(color='#888888', width=2)
        ))
        
        # Vẽ một số đường simulation để thấy biến động (chọn 30 đường ngẫu nhiên)
        if all_sims is not None:
            np.random.seed(42)
            random_indices = np.random.choice(len(all_sims), min(30, len(all_sims)), replace=False)
            for idx in random_indices:
                fig2.add_trace(go.Scatter(
                    x=future_dates, y=all_sims[idx],
                    name=None, line=dict(width=0.5, color='rgba(255,255,255,0.08)'),
                    showlegend=False, hoverinfo='skip'
                ))
        
        # Kênh tin cậy 95% (rộng)
        if forecast_bands_95 is not None:
            lower_95, upper_95 = forecast_bands_95
            fig2.add_trace(go.Scatter(
                x=future_dates, y=upper_95,
                name="Kênh 95%", line=dict(color='rgba(0,255,136,0.3)', width=1, dash='dash'),
                fill=None
            ))
            fig2.add_trace(go.Scatter(
                x=future_dates, y=lower_95,
                name=None, line=dict(color='rgba(255,68,68,0.3)', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(0,198,255,0.05)',
                showlegend=False
            ))
        
        # Kênh tin cậy 70% (hẹp hơn)
        if forecast_bands_70 is not None:
            lower_70, upper_70 = forecast_bands_70
            fig2.add_trace(go.Scatter(
                x=future_dates, y=upper_70,
                name="Kênh 70%", line=dict(color='rgba(0,255,136,0.5)', width=1.5, dash='dash'),
                fill=None
            ))
            fig2.add_trace(go.Scatter(
                x=future_dates, y=lower_70,
                name=None, line=dict(color='rgba(255,68,68,0.5)', width=1.5, dash='dash'),
                fill='tonexty', fillcolor='rgba(0,198,255,0.1)',
                showlegend=False
            ))
        
        # Đường dự báo trung bình
        fig2.add_trace(go.Scatter(
            x=future_dates, y=forecast_mean *1000,
            name=" Dự báo trung bình", line=dict(color='#00c6ff', width=3),
            mode='lines+markers', marker=dict(size=4, color='#00c6ff')
        ))
        
        # Đường median
        fig2.add_trace(go.Scatter(
            x=future_dates, y=forecast_median *1000,
            name="Dự báo trung vị", line=dict(color='#ffa500', width=2, dash='dot'),
            mode='lines'
        ))
        
        fig2.update_layout(
            title="Dự báo giá 30 ngày với Monte Carlo (có biến động thực tế)",
            xaxis_title="Ngày", yaxis_title="Giá (VND)",
            template="plotly_dark", height=550, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Tính toán thống kê dự báo
        trend = "TĂNG" if forecast_mean[-1] > current_price else "GIẢM"
        change_pct = ((forecast_mean[-1] - current_price) / current_price) * 100
        max_price = np.max(forecast_mean)
        min_price = np.min(forecast_mean)
        volatility_forecast = np.std(np.diff(forecast_mean) / forecast_mean[:-1]) * np.sqrt(252) if len(forecast_mean) > 1 else 0
        
        # Hiển thị thông tin chi tiết
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Xu hướng", f"{trend}", delta=f"{change_pct:+.1f}%")
        col2.metric("Giá cuối kỳ", f"{forecast_mean[-1]*1000:,.0f} VND")
        col3.metric("Biên độ", f"{min_price:,.0f} - {max_price:,.0f}")
        col4.metric("Volatility", f"{volatility_forecast*100:.1f}%")
        
        # Bảng dự báo chi tiết
        with st.expander("📋 Xem bảng dự báo chi tiết 30 ngày"):
            forecast_df = pd.DataFrame({
                "Ngày": future_dates.strftime('%Y-%m-%d'),
                "Dự báo TB (VND)": (forecast_mean *1000).round(0).astype(int),
                "Trung vị (VND)": (forecast_median *1000).round(0).astype(int),
                "Kênh dưới 70%": (forecast_bands_70[0] *1000).round(0).astype(int) if forecast_bands_70 else None,
                "Kênh trên 70%": (forecast_bands_70[1] *1000).round(0).astype(int) if forecast_bands_70 else None,
            })
            st.dataframe(forecast_df, use_container_width=True)
        
        if trend == "TĂNG":
            st.success(f" **Xu hướng dự báo: {trend}** (+{change_pct:.1f}% sau 30 ngày) - Dự báo có biến động lên xuống trong kỳ")
        else:
            st.error(f" **Xu hướng dự báo: {trend}** ({change_pct:.1f}% sau 30 ngày) - Dự báo có biến động lên xuống trong kỳ")
    else:
        st.warning(" Không thể thực hiện dự báo do dữ liệu không đủ.")
        trend = "TĂNG" if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else "GIẢM"

    # ===== NEWS =====
    st.subheader("📰 Tin tức mới nhất")
    for title, link, sentiment in get_news(sym):
        st.markdown(f"- {sentiment} [{title}]({link})")
    # ===== CONCLUSION =====
    st.subheader(" Kết luận & Khuyến nghị")
    if trend == "TĂNG":
        st.success(f"""
    Đánh giá tổng thể: TÍCH CỰC
    - Dữ liệu kỹ thuật cho thấy xu hướng giá đang vận động theo chiều hướng tăng
    - Đường trung bình MA20 duy trì phía trên MA50 → động lực tăng vẫn được giữ
    - Mô hình dự báo Monte Carlo cho thấy giá có xu hướng tăng nhưng có biến động lên xuống trong 30 ngày tới
    ** Quan điểm đầu tư:** Có thể xem xét giải ngân từng phần, đặt stop-loss hợp lý để quản trị rủi ro.
    """)
    else:
        st.error(f"""
    Đánh giá tổng thể: TIÊU CỰC
    - Xu hướng kỹ thuật đang suy yếu khi MA20 nằm dưới MA50
    - Giá có dấu hiệu mất động lực tăng và chịu áp lực điều chỉnh
    - Mô hình dự báo cho thấy khả năng giá tiếp tục giảm nhưng có thể có nhịp hồi phục ngắn
    ** Quan điểm đầu tư:** Chưa phù hợp để mở vị thế mới, nên quan sát thêm và chờ tín hiệu đảo chiều.
    """)

# ================= MODE 2 =================
if mode == "Danh mục đầu tư" and st.button("🔍 Phân tích danh mục"):
    if not symbols_input.strip():
        st.error("Vui lòng nhập mã cổ phiếu (VD: FPT,HPG,VCB)")
        st.stop()
    
    symbols = [s.strip().upper() for s in symbols_input.split(",")]
    
    if len(symbols) < 2:
        st.error("Cần ít nhất 2 cổ phiếu để phân tích danh mục")
        st.stop()

    with st.spinner("Đang phân tích danh mục..."):
        data = {}
        returns = pd.DataFrame()

        for sym in symbols:
            df = get_data(sym)
            if df is not None and len(df) > 20:
                df['Signal'] = 0
                df.loc[df['MA20'] > df['MA50'], 'Signal'] = 1
                df['Position'] = df['Signal'].diff()
                data[sym] = df
                returns[sym] = df['Return']

    if len(data) < 2:
        st.error("Không đủ dữ liệu cho 2 cổ phiếu trở lên. Vui lòng thử mã khác.")
        st.stop()

    # ===== CHART SO SÁNH GIÁ =====
    st.subheader(" Biểu đồ giá và xu hướng từng cổ phiếu")
    
    num_cols = min(3, len(data))
    cols = st.columns(num_cols)

    for i, (sym, df) in enumerate(data.items()):
        col_idx = i % num_cols
        
        with cols[col_idx]:
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 15px;
                text-align: center;
                border-left: 4px solid #00c6ff;
            '>
                <h3 style='margin:0; color: #00c6ff;'>{sym}</h3>
                <p style='margin:5px 0 0 0; color: #9ca3af; font-size: 12px;'>
                    Giá: {df['close'].iloc[-1]*1000:,.0f} VND
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['close'] *1000,
                name="Giá", line=dict(width=3, color='#00c6ff')
            ))
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['MA20'] *1000,
                name="MA20", line=dict(dash='dash', width=2, color='#ffa500')
            ))
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['MA50'] *1000,
                name="MA50", line=dict(dash='dot', width=2, color='#00ff88')
            ))
            
            buy = df[df['Position'] == 1]
            sell = df[df['Position'] == -1]
            
            if len(buy) > 0:
                fig.add_trace(go.Scatter(
                    x=buy['time'], y=buy['close'],
                    mode='markers', name='🔵 BUY',
                    marker=dict(color='#00ff88', size=12, symbol='triangle-up')
                ))
            if len(sell) > 0:
                fig.add_trace(go.Scatter(
                    x=sell['time'], y=sell['close'],
                    mode='markers', name='🔴 SELL',
                    marker=dict(color='#ff4444', size=12, symbol='triangle-down')
                ))
            
            fig.update_layout(
                title=dict(text=f"{sym} - Xu hướng giá", font=dict(size=14), x=0.5),
                xaxis_title="Ngày", yaxis_title="Giá (VND)",
                template="plotly_dark", height=450, hovermode='x unified',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            ma20 = df['MA20'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            
            if ma20 > ma50:
                st.success(f"""
                **Xu hướng TĂNG - Golden Cross**
                - MA20 ({ma20:,.0f}) > MA50 ({ma50:,.0f})
                - Động lực ngắn hạn mạnh
                - Có thể tiếp tục nắm giữ hoặc tăng tỷ trọng
                """)
            else:
                st.error(f"""
                **Xu hướng GIẢM - Death Cross**
                - MA20 ({ma20:,.0f}) < MA50 ({ma50:,.0f})
                - Áp lực giảm chiếm ưu thế
                - Nên thận trọng, giảm tỷ trọng
                """)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Biến động", f"{df['Return'].std()*100:.2f}%")
            with c2:
                change_5d = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
                st.metric("5 ngày", f"{change_5d:+.1f}%")

    # ===== FORECAST COMPARE - FIXED với Monte Carlo =====
    st.subheader(" Dự báo giá 30 ngày (Monte Carlo Simulation)")
    
    forecast_cols = st.columns(num_cols)
    forecast_results = {}
    trend_summary = {}

    for i, (sym, df) in enumerate(data.items()):
        col_idx = i % num_cols
        future_dates, forecast_mean, forecast_median, forecast_bands_70, forecast_bands_95, all_sims = forecast_monte_carlo(df, n_simulations=100)
        
        with forecast_cols[col_idx]:
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                text-align: center;
            '>
                <h4 style='margin:0; color: #00c6ff;'>{sym}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if future_dates is not None and forecast_mean is not None:
                forecast_results[sym] = pd.Series(forecast_mean, index=future_dates)
                trend = "TĂNG" if forecast_mean[-1] > forecast_mean[0] else "GIẢM"
                trend_summary[sym] = trend
                
                final_price = forecast_mean[-1]
                current_price = df['close'].iloc[-1]
                change_pct = ((final_price - current_price) / current_price) * 100

                fig = go.Figure()
                
                # Historical
                fig.add_trace(go.Scatter(
                    x=df['time'].iloc[-60:], y=df['close'].iloc[-60:] *1000,
                    name="Lịch sử", line=dict(color='#888888', width=2)
                ))
                
                # Kênh 70%
                if forecast_bands_70 is not None:
                    lower_70, upper_70 = forecast_bands_70
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=upper_70,
                        name="Kênh 70%", line=dict(color='rgba(0,255,136,0.4)', width=1, dash='dash'),
                        fill=None
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=lower_70,
                        name=None, line=dict(color='rgba(255,68,68,0.4)', width=1, dash='dash'),
                        fill='tonexty', fillcolor='rgba(0,198,255,0.08)',
                        showlegend=False
                    ))
                
                # Dự báo trung bình
                fig.add_trace(go.Scatter(
                    x=future_dates, y=forecast_mean,
                    name="Dự báo", line=dict(color='#00c6ff', width=2.5),
                    mode='lines+markers', marker=dict(size=3)
                ))
                
                fig.update_layout(
                    title=f"Dự báo 30 ngày - {trend} {change_pct:+.1f}%",
                    template="plotly_dark", height=400, hovermode='x unified',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Hiển thị thông tin
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.metric("Xu hướng", trend, delta=f"{change_pct:+.1f}%")
                with col_f2:
                    st.metric("Giá cuối", f"{final_price *1000:,.0f} VND")
                
                if trend == "TĂNG":
                    st.success(f" Dự báo TĂNG {change_pct:+.1f}%")
                else:
                    st.error(f" Dự báo GIẢM {change_pct:+.1f}%")
            else:
                st.warning(f" Không đủ dữ liệu dự báo cho {sym}")
                trend_summary[sym] = "Không xác định"

    # ===== KẾT LUẬN FORECAST =====
    st.markdown("### Đánh giá xu hướng danh mục dựa trên dự báo Monte Carlo")
    
    col_up, col_down = st.columns(2)
    with col_up:
        st.markdown("#### Mã có xu hướng TĂNG")
        up_stocks = [sym for sym, trend in trend_summary.items() if trend == "TĂNG"]
        if up_stocks:
            for sym in up_stocks:
                st.write(f"🔹 **{sym}** - Kỳ vọng tăng trưởng nhưng có biến động")
        else:
            st.write("⚪ Không có mã nào")
    
    with col_down:
        st.markdown("#### Mã có xu hướng GIẢM")
        down_stocks = [sym for sym, trend in trend_summary.items() if trend == "GIẢM"]
        if down_stocks:
            for sym in down_stocks:
                st.write(f"🔻 **{sym}** - Cần giảm tỷ trọng, có thể có nhịp hồi")
        else:
            st.write("⚪ Không có mã nào")

    st.info("""
    **Chiến lược đề xuất:**
    -  **Tăng tỷ trọng** các mã có xu hướng TĂNG
    -  **Giảm tỷ trọng** các mã có xu hướng GIẢM
    -  **Dự báo có biến động** - cần đặt stop-loss hợp lý
    -  **Tái cân bằng** danh mục hàng tháng
    """)

    # ===== MPT & RISK ANALYSIS =====
    if returns.shape[1] >= 2 and returns.dropna().shape[0] > 0:
        results, best_weights, labels = optimize_mpt(returns.dropna())
        
        st.subheader(" Phân tích rủi ro danh mục")
        
        market_return = returns.mean(axis=1)
        beta_list = []
        for sym in labels:
            cov = np.cov(returns[sym].dropna(), market_return.dropna())[0][1]
            var = np.var(market_return.dropna())
            beta = cov / var if var != 0 else 0
            beta_list.append(beta)
        
        for i, sym in enumerate(labels):
            st.write(f"**{sym}** | Beta: {beta_list[i]:.2f}")
        
        portfolio_returns = returns.dot(best_weights)
        VaR_95 = np.percentile(portfolio_returns.dropna(), 5)
        st.warning(f" **VaR 95%**: {VaR_95:.2%} → Có 5% khả năng lỗ vượt mức này trong 1 ngày")
        
        # Monte Carlo cho danh mục
        st.subheader(" Mô phỏng Monte Carlo danh mục (30 ngày)")
        simulations = 50
        days = 30
        sim_results = []
        for _ in range(simulations):
            sim_path = [1]
            for d in range(days):
                rand_return = np.random.normal(portfolio_returns.mean(), portfolio_returns.std())
                sim_path.append(sim_path[-1] * (1 + rand_return))
            sim_results.append(sim_path)
        
        fig_mc = go.Figure()
        for sim in sim_results:
            fig_mc.add_trace(go.Scatter(
                y=sim, mode='lines', line=dict(width=1, color='rgba(0,198,255,0.3)'), showlegend=False
            ))
        mean_path = np.mean(sim_results, axis=0)
        fig_mc.add_trace(go.Scatter(
            y=mean_path, mode='lines', line=dict(width=3, color='#ffa500'), name='Trung bình'
        ))
        fig_mc.update_layout(
            title="Mô phỏng biến động danh mục trong 30 ngày",
            xaxis_title="Ngày", yaxis_title="Giá trị (chuẩn hóa)",
            template="plotly_dark", height=500
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # ===== EFFICIENT FRONTIER - FIXED =====
        st.subheader(" Efficient Frontier - Đường biên hiệu quả")
        
        # Lọc bỏ các điểm có volatility hoặc return quá bất thường
        valid_indices = (results[:, 1] > 0) & (results[:, 0] > -1) & (results[:, 1] < results[:, 1].max() * 2)
        filtered_results = results[valid_indices]
        
        if len(filtered_results) > 0:
            fig = go.Figure()
            
            # Vẽ tất cả các danh mục
            fig.add_trace(go.Scatter(
                x=filtered_results[:, 1],
                y=filtered_results[:, 0],
                mode='markers',
                marker=dict(
                    size=5,
                    color=filtered_results[:, 2],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name='Danh mục ngẫu nhiên',
                hovertemplate='Rủi ro: %{x:.2%}<br>Lợi nhuận: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>'
            ))
            
            # Tìm các điểm đặc biệt
            best_idx = np.argmax(filtered_results[:, 2])
            min_vol_idx = np.argmin(filtered_results[:, 1])
            max_return_idx = np.argmax(filtered_results[:, 0])
            
            # Điểm tối ưu (Sharpe max)
            fig.add_trace(go.Scatter(
                x=[filtered_results[best_idx, 1]],
                y=[filtered_results[best_idx, 0]],
                mode='markers',
                marker=dict(color='red', size=18, symbol='star'),
                name=f'🌟 Tối ưu (Sharpe: {filtered_results[best_idx, 2]:.2f})'
            ))
            
            # Điểm rủi ro thấp nhất
            fig.add_trace(go.Scatter(
                x=[filtered_results[min_vol_idx, 1]],
                y=[filtered_results[min_vol_idx, 0]],
                mode='markers',
                marker=dict(color='yellow', size=12, symbol='circle'),
                name=f' Rủi ro thấp nhất (Vol: {filtered_results[min_vol_idx, 1]:.2%})'
            ))
            
            # Điểm lợi nhuận cao nhất
            fig.add_trace(go.Scatter(
                x=[filtered_results[max_return_idx, 1]],
                y=[filtered_results[max_return_idx, 0]],
                mode='markers',
                marker=dict(color='purple', size=12, symbol='diamond'),
                name=f' Lợi nhuận cao nhất (Return: {filtered_results[max_return_idx, 0]:.2%})'
            ))
            
            fig.update_layout(
                title="Biên hiệu quả - Tối ưu hóa danh mục (5000 danh mục ngẫu nhiên)",
                xaxis_title="Rủi ro (Volatility hàng năm)",
                yaxis_title="Lợi nhuận kỳ vọng hàng năm",
                template="plotly_dark",
                height=550,
                hovermode='closest',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Giải thích
            st.info(f"""
            **Giải thích biểu đồ:**
            - **{len(filtered_results)} danh mục ngẫu nhiên** được tạo từ {len(labels)} cổ phiếu
            - **🌟 Điểm đỏ**: Danh mục tối ưu nhất (Max Sharpe Ratio = {filtered_results[best_idx, 2]:.2f})
            - ** Điểm vàng**: Danh mục có rủi ro thấp nhất (Vol = {filtered_results[min_vol_idx, 1]:.2%})
            - ** Điểm tím**: Danh mục có lợi nhuận kỳ vọng cao nhất (Return = {filtered_results[max_return_idx, 0]:.2%})
            """)
        else:
            st.warning("Không đủ dữ liệu hợp lệ để vẽ Efficient Frontier")
        
                # ===== PORTFOLIO ALLOCATION - PHÂN BỔ DỰA TRÊN XU HƯỚNG DỰ BÁO =====
        st.subheader(" Danh mục đầu tư đề xuất (dựa trên xu hướng dự báo)")
        
        # Lấy xu hướng dự báo từ forecast_results
        stock_trends = {}
        for sym in labels:
            if sym in forecast_results:
                preds = forecast_results[sym]
                if preds is not None and len(preds) > 0:
                    # Xu hướng dự báo: TĂNG nếu giá cuối > giá đầu
                    if preds.iloc[-1] > preds.iloc[0]:
                        trend = "TĂNG"
                        strength = (preds.iloc[-1] - preds.iloc[0]) / preds.iloc[0]  # % tăng dự kiến
                    else:
                        trend = "GIẢM"
                        strength = (preds.iloc[0] - preds.iloc[-1]) / preds.iloc[0]  # % giảm dự kiến
                else:
                    trend = "KHÔNG XÁC ĐỊNH"
                    strength = 0
            else:
                # Nếu không có dự báo, dùng MA20/MA50
                df_stock = data.get(sym)
                if df_stock is not None:
                    if df_stock['MA20'].iloc[-1] > df_stock['MA50'].iloc[-1]:
                        trend = "TĂNG"
                        strength = 0.05
                    else:
                        trend = "GIẢM"
                        strength = 0.05
                else:
                    trend = "KHÔNG XÁC ĐỊNH"
                    strength = 0
            
            stock_trends[sym] = {"trend": trend, "strength": strength}
        
        # Hiển thị xu hướng từng cổ phiếu
        st.markdown("#### Xu hướng dự báo từng cổ phiếu:")
        for sym, info in stock_trends.items():
            if info["trend"] == "TĂNG":
                st.write(f"🔹 **{sym}**:  {info['trend']} (tăng {info['strength']*100:.1f}%)")
            elif info["trend"] == "GIẢM":
                st.write(f"🔹 **{sym}**:  {info['trend']} (giảm {info['strength']*100:.1f}%)")
            else:
                st.write(f"🔹 **{sym}**:  {info['trend']}")
        
        # ===== TÍNH TOÁN PHÂN BỔ DỰA TRÊN XU HƯỚNG =====
        # Cổ phiếu TĂNG được ưu tiên, cổ phiếu GIẢM bị giảm tỷ trọng
        base_weight = 1 / len(labels)  # Tỷ trọng cơ bản
        
        adjusted_weights = []
        for sym in labels:
            info = stock_trends[sym]
            
            if info["trend"] == "TĂNG":
                # Cổ phiếu tăng: tăng tỷ trọng (hệ số 1.3 đến 1.8 tùy theo mức độ tăng)
                multiplier = min(1.8, 1.3 + info["strength"] * 5)
                weight = base_weight * multiplier
            elif info["trend"] == "GIẢM":
                # Cổ phiếu giảm: giảm tỷ trọng (hệ số 0.3 đến 0.7)
                multiplier = max(0.3, 0.7 - info["strength"] * 5)
                weight = base_weight * multiplier
            else:
                # Không xác định: giữ nguyên
                weight = base_weight
            
            adjusted_weights.append(weight)
        
        # Chuẩn hóa tổng = 1
        adjusted_weights = np.array(adjusted_weights)
        total_weight = adjusted_weights.sum()
        if total_weight > 0:
            adjusted_weights = adjusted_weights / total_weight
        else:
            adjusted_weights = np.ones(len(labels)) / len(labels)
        
        # ===== HIỂN THỊ PHÂN BỔ ĐỀ XUẤT =====
        st.markdown("#### Phân bổ đề xuất theo xu hướng:")
        
        # Tạo dataframe hiển thị
        allocation_data = []
        for i, sym in enumerate(labels):
            allocation_data.append({
                "Cổ phiếu": sym,
                "Xu hướng": stock_trends[sym]["trend"],
                "Tỷ trọng": f"{adjusted_weights[i]*100:.1f}%",
                "Giá trị (VND)": f"{capital * adjusted_weights[i]:,.0f}"
            })
        
        st.dataframe(pd.DataFrame(allocation_data), use_container_width=True)
        
        # Giải thích lý do phân bổ
        st.info("""
        **Cách thức phân bổ:**
        - **Cổ phiếu dự báo TĂNG** → Tăng tỷ trọng (ưu tiên nắm giữ)
        - **Cổ phiếu dự báo GIẢM** → Giảm tỷ trọng (hạn chế rủi ro)
        - **Điều chỉnh theo khẩu vị rủi ro** của khách hàng
        """)
        
        # ===== ĐIỀU CHỈNH THEO KHẨU VỊ RỦI RO =====
        if risk_profile == "Thấp":
            # Khẩu vị thấp: giảm tỷ trọng cổ phiếu giảm nhiều hơn
            for i, sym in enumerate(labels):
                if stock_trends[sym]["trend"] == "GIẢM":
                    adjusted_weights[i] *= 0.5
            # Chuẩn hóa lại
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            st.success("**Đã điều chỉnh theo khẩu vị THẤP** (giảm tỷ trọng cổ phiếu giảm)")
            
        elif risk_profile == "Cao":
            # Khẩu vị cao: tăng tỷ trọng cổ phiếu tăng nhiều hơn
            for i, sym in enumerate(labels):
                if stock_trends[sym]["trend"] == "TĂNG":
                    adjusted_weights[i] *= 1.2
            # Chuẩn hóa lại
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            st.warning("**Đã điều chỉnh theo khẩu vị CAO** (tăng tỷ trọng cổ phiếu tăng)")
        else:
            st.info("⚖️ **Giữ nguyên phân bổ theo khẩu vị TRUNG BÌNH**")
        
        # Hiển thị lại sau khi điều chỉnh
        st.markdown("#### Phân bổ cuối cùng sau điều chỉnh:")
        for i, sym in enumerate(labels):
            st.write(f"**{sym}**: {adjusted_weights[i]*100:.1f}% → {capital * adjusted_weights[i]:,.0f} VND")
        
        # ===== PIE CHART - FIXED =====
        st.subheader("Biểu đồ phân bổ danh mục")
        
        # Chuẩn bị dữ liệu cho pie chart
        pie_data = []
        for i, sym in enumerate(labels):
            if adjusted_weights[i] > 0.01:  # Chỉ hiển thị nếu > 1%
                pie_data.append({
                    'symbol': sym,
                    'weight': adjusted_weights[i],
                    'value': capital * adjusted_weights[i]
                })
        
        # Sắp xếp theo weight giảm dần
        pie_data = sorted(pie_data, key=lambda x: x['weight'], reverse=True)
        
        if pie_data:
            pie_labels = [item['symbol'] for item in pie_data]
            pie_values = [item['weight'] for item in pie_data]
            pie_amounts = [item['value'] for item in pie_data]
            
            # Màu sắc
            colors = ['#00c6ff', '#ffa500', '#00ff88', '#ff4444', '#9b59b6', '#e74c3c', '#2ecc71', '#f1c40f']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f"{sym} ({amount:,.0f} VND)" for sym, amount in zip(pie_labels, pie_amounts)],
                values=pie_values,
                hole=0.4,
                textinfo='label+percent',
                textposition='auto',
                marker=dict(colors=colors[:len(pie_labels)]),
                pull=[0.05 if i == 0 else 0 for i in range(len(pie_labels))],
                hoverinfo='label+percent+value',
                hovertemplate='%{label}<br>Tỷ trọng: %{percent:.1f}%%<br>Giá trị: %{value:,.0f} VND<extra></extra>'
            )])
            
            fig_pie.update_layout(
                title=f"Phân bổ danh mục - Khẩu vị {risk_profile}",
                template="plotly_dark",
                height=500,
                annotations=[dict(
                    text=f"Tổng vốn<br>{capital:,.0f} VND",
                    x=0.5, y=0.5, font_size=14, showarrow=False,
                    font=dict(color='white', size=14, weight='bold')
                )],
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Hiển thị bảng chi tiết
            st.markdown("#### Chi tiết phân bổ vốn:")
            for item in pie_data:
                st.write(f"**{item['symbol']}**: {item['weight']*100:.1f}% → {item['value']:,.0f} VND")
        else:
            st.warning("Không có dữ liệu để hiển thị biểu đồ phân bổ")
        
        # ===== AI FUND ADVISOR - Đề xuất tái cơ cấu danh mục =====
        st.subheader("Đề xuất tái cơ cấu danh mục")
        
        portfolio_review = []
        
        for i, sym in enumerate(labels):
            if sym in forecast_results:
                preds = forecast_results[sym]
                trend = (preds.iloc[-1] - preds.iloc[0]) / preds.iloc[0] if len(preds) > 0 else 0
            else:
                trend = 0
            
            weight = adjusted_weights[i]
            vol = returns[sym].std() if sym in returns.columns else 0.1
            
            # Scoring kiểu quỹ
            score = trend / (vol + 1e-6)
            
            if score > 1:
                status = "🏆 CORE HOLD - Cổ phiếu trụ cột, nên giữ tỷ trọng cao"
            elif score > 0.5:
                status = "✅ HOLD - Tiếp tục nắm giữ"
            elif score > 0:
                status = "⚠️ REDUCE - Giảm tỷ trọng"
            else:
                status = "❌ REMOVE - Loại khỏi danh mục"
            
            portfolio_review.append((sym, trend, vol, weight, score, status))
        
        # Hiển thị đánh giá từng mã
        st.markdown("### Đánh giá từng mã")
        
        remove_list = []
        reduce_list = []
        
        for sym, trend, vol, weight, score, status in portfolio_review:
            with st.expander(f"🔹 **{sym}** - {status.split('-')[0]}"):
                st.write(f"""
                - **Tỷ trọng hiện tại:** {weight*100:.1f}%
                - **Xu hướng dự báo:** {trend:+.2%}
                - **Rủi ro (Volatility):** {vol:.2%}
                - **Hiệu quả (Sharpe-like):** {score:.2f}
                - **Khuyến nghị:** {status}
                """)
            
            if "REMOVE" in status:
                remove_list.append(sym)
            elif "REDUCE" in status:
                reduce_list.append(sym)
        
        # Nhận định tổng thể
        st.markdown("### Nhận định tổng thể")
        
        if not remove_list and not reduce_list:
            st.success("""
            **Danh mục đạt trạng thái tối ưu**
            - Các tài sản đều có hiệu quả tốt
            - Cân bằng giữa rủi ro và lợi nhuận
            - **Chiến lược:** HOLD & REBALANCE nhẹ khi cần
            """)
        else:
            st.warning(f"""
            **⚠️ Danh mục cần tái cơ cấu**
            - **Nên loại bỏ:** {", ".join(remove_list) if remove_list else "Không có"}
            - **Nên giảm tỷ trọng:** {", ".join(reduce_list) if reduce_list else "Không có"}
            - **Mục tiêu:** Tối ưu lại hiệu suất Sharpe toàn danh mục
            """)
        
                # ===== MARKET SCAN - GỢI Ý CỔ PHIẾU THAY THẾ =====
        st.markdown("### 🔍 Gợi ý cổ phiếu thay thế")
        
        # Danh sách cổ phiếu quét
        scan_list = ["FPT", "HPG", "VCB", "VNM", "MWG", "SSI", "TCB", "MBB", "ACB", "CTG", "GAS", "PLX", "VIC", "MSN", "REE", "BVH"]
        
        # Phân tích từng cổ phiếu
        stock_analysis = []
        
        with st.spinner("Đang phân tích cổ phiếu tiềm năng..."):
            for sym in scan_list:
                # Bỏ qua cổ phiếu đã có trong danh mục
                if sym in data.keys():
                    continue
                
                df_test = get_data(sym)
                if df_test is None or len(df_test) < 30:
                    continue
                
                # Tính các chỉ số
                current_price = df_test['close'].iloc[-1]
                ma20 = df_test['MA20'].iloc[-1]
                ma50 = df_test['MA50'].iloc[-1]
                volatility = df_test['Return'].std() * 100
                returns = df_test['Return'].dropna()
                
                # Tính ROI 20 ngày gần nhất
                roi_20d = (df_test['close'].iloc[-1] / df_test['close'].iloc[-20] - 1) * 100 if len(df_test) >= 20 else 0
                
                # Tính chỉ số ổn định (Sharpe-like)
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                # Xác định xu hướng
                if ma20 > ma50:
                    trend = "TĂNG"
                    trend_strength = (ma20 / ma50 - 1) * 100
                    trend_color = "🟢"
                else:
                    trend = "GIẢM"
                    trend_strength = (ma50 / ma20 - 1) * 100
                    trend_color = "🔴"
                
                # Tính điểm ổn định (càng cao càng tốt)
                stability_score = sharpe_ratio * 10
                
                # Tính điểm tăng trưởng
                growth_score = roi_20d if roi_20d > 0 else roi_20d * 0.5
                
                # Tổng điểm (ưu tiên ổn định + tăng trưởng)
                total_score = stability_score + growth_score
                
                stock_analysis.append({
                    "symbol": sym,
                    "price": current_price,
                    "trend": trend,
                    "trend_strength": abs(trend_strength),
                    "trend_color": trend_color,
                    "volatility": volatility,
                    "roi_20d": roi_20d,
                    "sharpe_ratio": sharpe_ratio,
                    "stability_score": stability_score,
                    "growth_score": growth_score,
                    "total_score": total_score,
                    "ma20": ma20,
                    "ma50": ma50
                })
        
        # Sắp xếp theo tổng điểm giảm dần (ưu tiên ổn định và tăng trưởng)
        stock_analysis = sorted(stock_analysis, key=lambda x: x["total_score"], reverse=True)
        
        if stock_analysis:
            st.markdown("#### Top cổ phiếu tiềm năng (tăng trưởng ổn định):")
            
            # Lấy top 8 cổ phiếu
            top_stocks = stock_analysis[:8]
            
            # Hiển thị dạng lưới 4 cột
            for i in range(0, len(top_stocks), 4):
                cols = st.columns(4)
                row_stocks = top_stocks[i:i+4]
                
                for j, stock in enumerate(row_stocks):
                    with cols[j]:
                        # Màu sắc theo xu hướng
                        border_color = "#00ff88" if stock["trend"] == "TĂNG" else "#ff4444"
                        trend_icon = "" if stock["trend"] == "TĂNG" else ""
                        
                        st.markdown(f"""
                        <div style='| Giá hiện tại | {stock['price'] *1000:,.0f} VND |
                            background: linear-gradient(135deg, #1a1a2e 0%, #0f0f23 100%);
                            padding: 15px 8px;
                            border-radius: 12px;
                            text-align: center;
                            border: 2px solid {border_color};
                            margin: 5px;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        '>
                            <h3 style='color: #00c6ff; margin: 0; font-size: 24px; font-weight: bold;'>{stock['symbol']}</h3>
                            <p style='color: {border_color}; margin: 8px 0; font-size: 18px; font-weight: bold;'>{trend_icon} {stock['trend']} {stock['trend_strength']:.1f}%</p>
                            <p style='color: #ffffff; margin: 5px 0; font-size: 13px;'>{stock['price'] *1000:,.0f} VND</p>
                            <p style='color: #ffa500; margin: 5px 0; font-size: 11px;'> {stock['volatility']:.1f}%</p>
                            <p style='color: #00ff88; margin: 5px 0; font-size: 11px;'> 20 ngày: {stock['roi_20d']:+.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Nút xem chi tiết
                        if st.button(f"🔍 Xem {stock['symbol']}", key=f"view_{stock['symbol']}_{i}_{j}", use_container_width=True):
                            st.session_state['selected_stock_detail'] = stock
                            st.rerun()
            
            # Hiển thị chi tiết khi chọn một cổ phiếu
            if 'selected_stock_detail' in st.session_state:
                stock = st.session_state['selected_stock_detail']
                st.markdown("---")
                st.markdown(f"""
                ### Phân tích chi tiết: **{stock['symbol']}**
                
                | Chỉ số | Giá trị | Đánh giá |
        |--------|---------|----------|
                | Giá hiện tại | {stock['price']:,.0f} VND | - |
                | MA20 | {stock['ma20']:,.0f} VND | Đường trung bình 20 ngày |
                | MA50 | {stock['ma50']:,.0f} VND | Đường trung bình 50 ngày |
                | Xu hướng | {stock['trend']} {stock['trend_strength']:.1f}% | {stock['trend_color']} |
                | Biến động | {stock['volatility']:.2f}% | {'Thấp ' if stock['volatility'] < 30 else 'Cao '} |
                | ROI 20 ngày | {stock['roi_20d']:+.2f}% | {'Tốt ' if stock['roi_20d'] > 0 else 'Xấu '} |
                | Sharpe Ratio | {stock['sharpe_ratio']:.2f} | {'Tốt ' if stock['sharpe_ratio'] > 1 else 'Trung bình' if stock['sharpe_ratio'] > 0 else 'Kém '} |
                
                ** Vì sao nên chọn?**
                - {'MA20 nằm TRÊN MA50 → xu hướng TĂNG' if stock['trend'] == 'TĂNG' else 'MA20 nằm DƯỚI MA50 → cần theo dõi thêm'}
                - Biến động {stock['volatility']:.1f}% - {'ổn định, ít rủi ro' if stock['volatility'] < 30 else 'khá cao, cần cân nhắc'}
                - Tăng trưởng {stock['roi_20d']:+.1f}% trong 20 ngày qua
                - Điểm ổn định: {stock['stability_score']:.1f}/10
                """)
                
                # Tìm cổ phiếu đang giảm trong danh mục để thay thế
                weak_stocks = []
                for sym in data.keys():
                    if sym in forecast_results:
                        preds = forecast_results[sym]
                        if preds is not None and len(preds) > 0:
                            change = (preds.iloc[-1] - preds.iloc[0]) / preds.iloc[0]
                            if change < 0:  # Dự báo giảm
                                weak_stocks.append((sym, change))
                
                if weak_stocks:
                    weak_stocks = sorted(weak_stocks, key=lambda x: x[1])  # Sắp xếp theo mức giảm nhiều nhất
                    st.markdown("#### 🔄 Đề xuất thay thế:")
                    st.write(f"Cổ phiếu đang có xu hướng GIẢM trong danh mục: **{', '.join([w[0] for w in weak_stocks[:3]])}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Chọn cổ phiếu muốn thay thế
                        replace_target = st.selectbox(
                            "Chọn cổ phiếu muốn thay thế:",
                            options=[w[0] for w in weak_stocks],
                            key="replace_target"
                        )
                    
                    with col2:
                        if st.button(f"🔄 Thay thế {replace_target} bằng {stock['symbol']}", use_container_width=True):
                            # Tạo danh mục mới
                            new_portfolio = []
                            for sym in data.keys():
                                if sym == replace_target:
                                    new_portfolio.append(stock['symbol'])
                                else:
                                    new_portfolio.append(sym)
                            
                            st.session_state['new_portfolio'] = new_portfolio
                            st.success(f" Đã thay thế **{replace_target}** bằng **{stock['symbol']}**")
                            st.info(f"💡 Danh mục mới: {', '.join(new_portfolio)}")
                            
                            # Nút xác nhận
                            if st.button("🔄 Xác nhận và phân tích lại", use_container_width=True):
                                st.session_state['confirmed_portfolio'] = new_portfolio
                                st.rerun()
                else:
                    st.info(" Danh mục hiện tại đang có xu hướng tốt, không cần thay thế!")
                
                # Nút đóng
                if st.button(" Đóng", key="close_detail"):
                    del st.session_state['selected_stock_detail']
                    st.rerun()
            
            # Thông báo số lượng
            st.caption(f" Tìm thấy {len(stock_analysis)} cổ phiếu tiềm năng")
            
        else:
            st.info("Không tìm thấy cổ phiếu phù hợp")
        
                # ===== AUTO REALLOCATION - TÁI PHÂN BỔ VỐN =====
        st.markdown("### 🔄 Gợi ý tái phân bổ vốn")
        
        # Hiển thị danh mục hiện tại
        st.markdown(" Danh mục hiện tại:")
        current_data = []
        for i, sym in enumerate(labels):
            # Lấy xu hướng dự báo của từng cổ phiếu
            trend_info = ""
            if sym in forecast_results:
                preds = forecast_results[sym]
                if preds is not None and len(preds) > 0:
                    if preds.iloc[-1] > preds.iloc[0]:
                        trend_info = " TĂNG"
                    else:
                        trend_info = " GIẢM"
            current_data.append({"Cổ phiếu": sym, "Tỷ trọng": f"{adjusted_weights[i]*100:.1f}%", "Giá trị": f"{capital * adjusted_weights[i]:,.0f} VND", "Xu hướng": trend_info})
        
        st.dataframe(pd.DataFrame(current_data), use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 3 CHIẾN LƯỢC PHÂN BỔ THAM KHẢO:")
        
        # ===== CHIẾN LƯỢC 1: AN TOÀN =====
        st.markdown("### CHIẾN LƯỢC 1: AN TOÀN (Bảo toàn vốn)")
        
        # Tính điểm an toàn cho từng cổ phiếu
        safety_scores = []
        for sym in labels:
            score = 1
            if sym in forecast_results:
                preds = forecast_results[sym]
                if preds is not None and len(preds) > 0:
                    if preds.iloc[-1] > preds.iloc[0]:
                        score *= 1.5  # Tăng điểm nếu dự báo tăng
                    else:
                        score *= 0.5  # Giảm điểm nếu dự báo giảm
            
            # Lấy volatility từ dữ liệu - ĐÃ SỬA LỖI
            try:
                if sym in returns.columns:
                    vol = returns[sym].std()
                elif isinstance(returns, pd.Series):
                    vol = returns.std()
                else:
                    vol = 0.03
            except:
                vol = 0.03
            
            if vol < 0.02:  # Biến động thấp
                score *= 1.3
            elif vol > 0.04:  # Biến động cao
                score *= 0.7
            
            safety_scores.append(max(0.3, min(2.0, score)))
        
        # Chuẩn hóa thành tỷ trọng
        total_score = sum(safety_scores)
        if total_score > 0:
            safe_weights = [s / total_score for s in safety_scores]
        else:
            safe_weights = [1/len(labels)] * len(labels)
        
        # Hiển thị bảng an toàn
        safe_data = []
        for i, sym in enumerate(labels):
            safe_data.append({
                "Cổ phiếu": sym,
                "Tỷ trọng": f"{safe_weights[i]*100:.1f}%",
                "Giá trị (VND)": f"{capital * safe_weights[i]:,.0f}",
                "Lý do": "🟢 Ổn định" if safety_scores[i] > 1 else "🟡 Bình thường" if safety_scores[i] > 0.7 else "🔴 Rủi ro cao"
            })
        
        st.dataframe(pd.DataFrame(safe_data), use_container_width=True)
        
        # Biểu đồ tròn an toàn
        fig_safe = go.Figure(data=[go.Pie(
            labels=labels,
            values=safe_weights,
            hole=0.4,
            marker=dict(colors=['#00ff88', '#00c6ff', '#ffa500', '#ff4444', '#888888'][:len(labels)])
        )])
        fig_safe.update_layout(height=400, template="plotly_dark", title="Phân bổ theo chiến lược AN TOÀN")
        st.plotly_chart(fig_safe, use_container_width=True)
        
        st.caption(" **Ưu tiên:** Cổ phiếu có xu hướng TĂNG và biến động THẤP")
        
        st.markdown("---")
        
        # ===== CHIẾN LƯỢC 2: CÂN BẰNG =====
        st.markdown("### CHIẾN LƯỢC 2: CÂN BẰNG (Phân bổ đều)")
        
        equal_weights = [1/len(labels)] * len(labels)
        
        # Hiển thị bảng cân bằng
        balance_data = []
        for i, sym in enumerate(labels):
            balance_data.append({
                "Cổ phiếu": sym,
                "Tỷ trọng": f"{equal_weights[i]*100:.1f}%",
                "Giá trị (VND)": f"{capital * equal_weights[i]:,.0f}",
                "Lý do": "⚖️ Phân bổ đều"
            })
        
        st.dataframe(pd.DataFrame(balance_data), use_container_width=True)
        
        # Biểu đồ tròn cân bằng
        fig_balance = go.Figure(data=[go.Pie(
            labels=labels,
            values=equal_weights,
            hole=0.4,
            marker=dict(colors=['#00c6ff', '#00ff88', '#ffa500', '#ff4444', '#888888'][:len(labels)])
        )])
        fig_balance.update_layout(height=400, template="plotly_dark", title="Phân bổ theo chiến lược CÂN BẰNG")
        st.plotly_chart(fig_balance, use_container_width=True)
        
        st.caption("**Ưu tiên:** Đa dạng hóa, rủi ro trung bình")
        
        st.markdown("---")
        
        # ===== CHIẾN LƯỢC 3: TĂNG TRƯỞNG =====
        st.markdown("### CHIẾN LƯỢC 3: TĂNG TRƯỞNG (Mạo hiểm)")
        
        # Tính điểm tăng trưởng cho từng cổ phiếu
        growth_scores = []
        growth_reasons = []
        for sym in labels:
            score = 1
            reason = ""
            
            if sym in forecast_results:
                preds = forecast_results[sym]
                if preds is not None and len(preds) > 0:
                    if preds.iloc[-1] > preds.iloc[0]:
                        growth_rate = (preds.iloc[-1] - preds.iloc[0]) / preds.iloc[0]
                        score = 1 + growth_rate * 10  # Tăng điểm theo % tăng dự báo
                        reason = f" Dự báo TĂNG {growth_rate*100:.1f}%"
                    else:
                        score = 0.3
                        reason = " Dự báo GIẢM"
                else:
                    reason = " Chưa có dự báo"
            else:
                reason = " Chưa có dữ liệu"
            
            # Lấy ROI 20 ngày
            if sym in data:
                df_stock = data[sym]
                if df_stock is not None and len(df_stock) >= 20:
                    roi_20d = (df_stock['close'].iloc[-1] / df_stock['close'].iloc[-20] - 1)
                    if roi_20d > 0:
                        score *= (1 + roi_20d * 5)
                        reason += f" | ROI 20 ngày: {roi_20d*100:+.1f}%"
            
            growth_scores.append(max(0.2, min(3.0, score)))
            growth_reasons.append(reason)
        
        # Chuẩn hóa thành tỷ trọng
        total_score = sum(growth_scores)
        if total_score > 0:
            growth_weights = [s / total_score for s in growth_scores]
        else:
            growth_weights = [1/len(labels)] * len(labels)
        
        # Hiển thị bảng tăng trưởng
        growth_data = []
        for i, sym in enumerate(labels):
            growth_data.append({
                "Cổ phiếu": sym,
                "Tỷ trọng": f"{growth_weights[i]*100:.1f}%",
                "Giá trị (VND)": f"{capital * growth_weights[i]:,.0f}",
                "Lý do": growth_reasons[i][:50]
            })
        
        st.dataframe(pd.DataFrame(growth_data), use_container_width=True)
        
        # Biểu đồ tròn tăng trưởng
        fig_growth = go.Figure(data=[go.Pie(
            labels=labels,
            values=growth_weights,
            hole=0.4,
            marker=dict(colors=['#ff4444', '#ffa500', '#ffcc00', '#00ff88', '#00c6ff'][:len(labels)])
        )])
        fig_growth.update_layout(height=400, template="plotly_dark", title="Phân bổ theo chiến lược TĂNG TRƯỞNG")
        st.plotly_chart(fig_growth, use_container_width=True)
        
        st.caption("**Ưu tiên:** Cổ phiếu có dự báo TĂNG và ROI 20 ngày CAO")
        
        st.markdown("---")
        
        # ===== SO SÁNH 3 CHIẾN LƯỢC =====
        st.markdown("### SO SÁNH 3 CHIẾN LƯỢC")
        
        comparison_data = []
        for i, sym in enumerate(labels):
            comparison_data.append({
                "Cổ phiếu": sym,
                "Hiện tại": f"{adjusted_weights[i]*100:.1f}%",
                "An toàn": f"{safe_weights[i]*100:.1f}%",
                "Cân bằng": f"{equal_weights[i]*100:.1f}%",
                "Tăng trưởng": f"{growth_weights[i]*100:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Tổng kết
        st.markdown("""
        <div style='background: #1f2937; padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h4 style='color: #00c6ff; margin: 0 0 10px 0;'>HƯỚNG DẪN CHỌN CHIẾN LƯỢC:</h4>
            <ul style='margin: 0; padding-left: 20px;'>
                <li><span style='color: #00ff88;'>AN TOÀN</span> - Dành cho nhà đầu tư muốn bảo toàn vốn, ưu tiên cổ phiếu ổn định</li>
                <li><span style='color: #00c6ff;'>CÂN BẰNG</span> - Dành cho nhà đầu tư trung dung, phân bổ đều các mã</li>
                <li><span style='color: #ffa500;'>TĂNG TRƯỞNG</span> - Dành cho nhà đầu tư chấp nhận rủi ro cao, kỳ vọng lợi nhuận lớn</li>
            </ul>
            <p style='margin: 15px 0 0 0; color: #9ca3af; font-size: 12px;'>Bạn có thể copy tỷ trọng từ chiến lược mong muốn để áp dụng vào danh mục thực tế.</p>
        </div>
        """, unsafe_allow_html=True)
# ================= MODE 3: OTC STOCK ANALYSIS - HIỆN ĐẠI =================
elif mode == "Phân tích cổ phiếu OTC" and st.button("🔍 Phân tích OTC PRO"):
    
    # ================= 1. INPUT =================
    with st.container():
        
        col1, col2 = st.columns(2)
        with col1:
            sym = st.text_input("🔖 Mã cổ phiếu OTC", value=symbols_input.strip().upper() if symbols_input.strip() else "", key="otc_sym", 
                               placeholder="VD: CAV, NS2, VCW...")
            if not sym:
                st.warning("⚠️ Vui lòng nhập mã cổ phiếu")
                st.stop()
        with col2:
            price_input = st.number_input("💰 Giá OTC (VND/cổ phiếu)", min_value=0, value=10000, step=1000, key="otc_price")
        
        col3, col4 = st.columns(2)
        with col3:
            shares_outstanding = st.number_input(" Số CP lưu hành (triệu)", min_value=0, value=100, step=10, key="otc_shares")
        with col4:
            avg_volume = st.number_input(" Volume TB (CP/ngày)", min_value=0, value=1000, step=500, key="otc_volume")
        
        col5, col6 = st.columns(2)
        with col5:
            industry = st.selectbox(" Ngành hoạt động", ["Khác", "Ngân hàng", "Bất động sản", "Sản xuất", "Công nghệ", "Tiêu dùng", "Năng lượng"], key="otc_industry")
    
    if not sym:
        st.error("⚠️ Vui lòng nhập mã cổ phiếu")
        st.stop()
    
    sym = sym.upper()
    
    # Mapping P/E theo ngành
    industry_pe_mapping = {
        "Ngân hàng": 10, "Bất động sản": 8, "Sản xuất": 12,
        "Công nghệ": 18, "Tiêu dùng": 15, "Năng lượng": 11, "Khác": 10
    }
    industry_pe = industry_pe_mapping.get(industry, 10)
    industry_pb = 1.5
    industry_ps = 1.0
    
    # ================= 2. ĐỌC DỮ LIỆU =================
    try:
        import os
        possible_paths = ["OTC_data_clean.csv", "./OTC_data_clean.csv", "../OTC_data_clean.csv", "data/OTC_data_clean.csv"]
        
        otc_df = None
        for path in possible_paths:
            if os.path.exists(path):
                otc_df = pd.read_csv(path)
                break

        if otc_df is None:
            uploaded_file = st.file_uploader("", type=['csv'], key="otc_upload", label_visibility="collapsed")
            if uploaded_file is not None:
                otc_df = pd.read_csv(uploaded_file)
            else:
                st.stop()
        
        otc_df = otc_df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = ['Revenue', 'GrossProfit', 'NetProfit', 'OperatingProfit', 
                        'TotalAssets', 'CurrentAssets', 'Debt', 'ShortTermDebt', 
                        'Equity', 'EPS', 'BVPS', 'PE', 'ROS', 'ROEA', 'ROAA']
        for col in numeric_cols:
            if col in otc_df.columns:
                otc_df[col] = pd.to_numeric(otc_df[col], errors='coerce')
    except Exception as e:
        st.error("Lỗi đọc dữ liệu")
        st.stop()
    
    stock_data = otc_df[otc_df['Symbol'] == sym].copy()
    
    if stock_data.empty:
        st.error(f"Không tìm thấy dữ liệu cho mã OTC: {sym}")
        available_symbols = otc_df['Symbol'].unique().tolist()
        st.info(f"Các mã có sẵn: {', '.join(available_symbols[:30])}")
        st.stop()
    
    # Lấy tên công ty
    company_name = sym
    if 'CompanyName' in otc_df.columns:
        company_row = otc_df[otc_df['Symbol'] == sym]
        if not company_row.empty:
            company_name = company_row['CompanyName'].iloc[0]
    
    # Header doanh nghiệp
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); padding: 30px; border-radius: 20px; margin-bottom: 25px; text-align: center;'>
        <div style='font-size: 48px; font-weight: bold; color: #00c6ff;'>{sym}</div>
        <div style='font-size: 20px; color: #ffffff; margin-top: 10px;'>{company_name}</div>
        <div style='font-size: 14px; color: #9ca3af; margin-top: 8px;'>Phân tích cổ phiếu OTC | Hệ thống hỗ trợ tư vấn viên HVA</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bảng dữ liệu
    with st.expander("📋 Báo cáo tài chính chi tiết", expanded=False):
        display_df = stock_data.copy()
        for col in ['Revenue', 'GrossProfit', 'NetProfit', 'OperatingProfit', 
                    'TotalAssets', 'CurrentAssets', 'Debt', 'ShortTermDebt', 'Equity']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        st.dataframe(display_df, use_container_width=True)
    
        # ================= BIỂU ĐỒ TÀI CHÍNH =================
    st.markdown("### PHÂN TÍCH CÁC XU HƯỚNG TÀI CHÍNH CỦA DOANH NGHIỆP")
    
    # Chuẩn bị dữ liệu
    rev_data = stock_data[stock_data['Revenue'].notna()]
    profit_data = stock_data[stock_data['NetProfit'].notna()]
    roe_data = stock_data[stock_data['ROEA'].notna()]
    debt_data = stock_data[stock_data['Debt'].notna() & stock_data['Equity'].notna()].copy()
    
    # ===== BIỂU ĐỒ 1: DOANH THU & LỢI NHUẬN =====
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 12px; border-radius: 15px; margin-bottom: 15px; text-align: center;'>
        <h4 style='color: #00c6ff; margin: 0;'>💰 DOANH THU & LỢI NHUẬN</h4>
        <p style='color: #9ca3af; font-size: 12px; margin: 5px 0 0 0;'>Đơn vị: Tỷ VND</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig1 = go.Figure()
    if not rev_data.empty:
        # FIX: Data là triệu, muốn hiển thị tỷ -> chia 1000
        fig1.add_trace(go.Scatter(
            x=rev_data['Year'], y=rev_data['Revenue'] / 1000,
            name="Doanh thu", line=dict(color='#00c6ff', width=4),
            mode='lines+markers', marker=dict(size=14, symbol='circle', color='#00c6ff', line=dict(width=2, color='white')),
            fill='tozeroy', fillcolor='rgba(0,198,255,0.15)'
        ))
    if not profit_data.empty:
        # FIX: Data là triệu, muốn hiển thị tỷ -> chia 1000
        fig1.add_trace(go.Scatter(
            x=profit_data['Year'], y=profit_data['NetProfit'] / 1000,
            name="Lợi nhuận", line=dict(color='#00ff88', width=4),
            mode='lines+markers', marker=dict(size=14, symbol='diamond', color='#00ff88', line=dict(width=2, color='white')),
            fill='tozeroy', fillcolor='rgba(0,255,136,0.15)'
        ))
    
    fig1.update_layout(
        template="plotly_dark", 
        height=420,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="Năm", font=dict(color='#9ca3af'))),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="Tỷ VND", font=dict(color='#9ca3af')))
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # ===== BIỂU ĐỒ 2: ROE VÀ D/E  =====
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 12px; border-radius: 15px; margin-bottom: 10px; text-align: center;'>
            <h4 style='color: #ffa500; margin: 0;'> ROE</h4>
            <p style='color: #9ca3af; font-size: 12px; margin: 5px 0 0 0;'>Tỷ suất lợi nhuận trên vốn</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig2 = go.Figure()
        if not roe_data.empty:
            colors = ['#00ff88' if x >= 20 else '#ffa500' if x >= 10 else '#ff4444' for x in roe_data['ROEA']]
            fig2.add_trace(go.Bar(
                x=roe_data['Year'], y=roe_data['ROEA'], 
                marker_color=colors,
                marker=dict(line=dict(width=1, color='white')),
                text=roe_data['ROEA'].round(1), 
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hovertemplate='Năm: %{x}<br>ROE: %{y:.1f}%<extra></extra>'
            ))
            fig2.add_hline(y=20, line_dash="dash", line_color="#00ff88", line_width=2,
                          annotation_text="🏆 Tốt (>20%)", annotation_position="bottom right",
                          annotation_font=dict(size=10, color="#00ff88"))
            fig2.add_hline(y=10, line_dash="dash", line_color="#ffa500", line_width=2,
                          annotation_text="⚠️ TB (>10%)", annotation_position="top right",
                          annotation_font=dict(size=10, color="#ffa500"))
        
        fig2.update_layout(
            template="plotly_dark", 
            height=380,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="Năm", font=dict(color='#9ca3af'))),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="ROE (%)", font=dict(color='#9ca3af')), 
                       range=[0, max(roe_data['ROEA'].max() + 10, 30)] if not roe_data.empty else [0, 30])
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 12px; border-radius: 15px; margin-bottom: 10px; text-align: center;'>
            <h4 style='color: #ff4444; margin: 0;'> TỶ LỆ NỢ/VỐN (D/E)</h4>
            <p style='color: #9ca3af; font-size: 12px; margin: 5px 0 0 0;'>Càng thấp càng tốt</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig3 = go.Figure()
        if not debt_data.empty:
            debt_data['D/E'] = debt_data['Debt'] / debt_data['Equity']
            colors = ['#00ff88' if x <= 1 else '#ffa500' if x <= 2 else '#ff4444' for x in debt_data['D/E']]
            fig3.add_trace(go.Bar(
                x=debt_data['Year'], y=debt_data['D/E'], 
                marker_color=colors,
                marker=dict(line=dict(width=1, color='white')),
                text=debt_data['D/E'].round(2), 
                textposition='outside',
                textfont=dict(size=12, color='white'),
                hovertemplate='Năm: %{x}<br>D/E: %{y:.2f}<extra></extra>'
            ))
            fig3.add_hline(y=1, line_dash="dash", line_color="#00ff88", line_width=2,
                          annotation_text="✅ An toàn", annotation_position="bottom right",
                          annotation_font=dict(size=10, color="#00ff88"))
            fig3.add_hline(y=2, line_dash="dash", line_color="#ffa500", line_width=2,
                          annotation_text="⚠️ Cảnh báo", annotation_position="top right",
                          annotation_font=dict(size=10, color="#ffa500"))
        
        fig3.update_layout(
            template="plotly_dark", 
            height=380,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="Năm", font=dict(color='#9ca3af'))),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="D/E (Lần)", font=dict(color='#9ca3af')))
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # ===== BIỂU ĐỒ 3: TĂNG TRƯỞNG HÀNG NĂM =====
    growth_data = []
    for i in range(1, len(stock_data)):
        prev_rev = stock_data.iloc[i-1]['Revenue']
        curr_rev = stock_data.iloc[i]['Revenue']
        prev_profit = stock_data.iloc[i-1]['NetProfit']
        curr_profit = stock_data.iloc[i]['NetProfit']
        
        rev_growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else 0
        profit_growth = ((curr_profit - prev_profit) / prev_profit * 100) if prev_profit > 0 else 0
        
        growth_data.append({
            'Năm': f"{stock_data.iloc[i-1]['Year']}→{stock_data.iloc[i]['Year']}",
            'Doanh thu': rev_growth,
            'Lợi nhuận': profit_growth
        })
    
    growth_df = pd.DataFrame(growth_data)
    
    if not growth_df.empty:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 12px; border-radius: 15px; margin: 15px 0 10px 0; text-align: center;'>
            <h4 style='color: #9b59b6; margin: 0;'> TĂNG TRƯỞNG HÀNG NĂM</h4>
            <p style='color: #9ca3af; font-size: 12px; margin: 5px 0 0 0;'>So sánh với năm trước (%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Bar(
            x=growth_df['Năm'], y=growth_df['Doanh thu'],
            name="Doanh thu",
            marker_color='#00c6ff',
            text=growth_df['Doanh thu'].round(1),
            textposition='outside',
            textfont=dict(size=11),
            hovertemplate='%{x}<br>Tăng trưởng doanh thu: %{y:.1f}%<extra></extra>'
        ))
        
        fig4.add_trace(go.Bar(
            x=growth_df['Năm'], y=growth_df['Lợi nhuận'],
            name="Lợi nhuận",
            marker_color='#00ff88',
            text=growth_df['Lợi nhuận'].round(1),
            textposition='outside',
            textfont=dict(size=11),
            hovertemplate='%{x}<br>Tăng trưởng lợi nhuận: %{y:.1f}%<extra></extra>'
        ))
        
        fig4.add_hline(y=0, line_dash="solid", line_color="#888", line_width=1)
        
        fig4.update_layout(
            template="plotly_dark",
            height=400,
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="Kỳ", font=dict(color='#9ca3af'))),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="Tăng trưởng (%)", font=dict(color='#9ca3af')))
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    # Lấy dữ liệu mới nhất
    latest = stock_data.iloc[-1]
    prev = stock_data.iloc[-2] if len(stock_data) >= 2 else latest
    
    # FIX: Data là triệu -> chia 1000 để ra tỷ VND
    revenue = latest.get('Revenue', 0) / 1000
    profit = latest.get('NetProfit', 0) / 1000
    prev_revenue = prev.get('Revenue', revenue * 1000) / 1000
    prev_profit = prev.get('NetProfit', profit * 1000) / 1000
    
    revenue_growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
    profit_growth = ((profit - prev_profit) / prev_profit * 100) if prev_profit > 0 else 0
    roe = latest.get('ROEA', 0)
    debt = latest.get('Debt', 0)
    equity = latest.get('Equity', 1)
    debt_ratio = debt / equity if equity > 0 else 999
    eps = latest.get('EPS', 0)
    bvps = latest.get('BVPS', 0)
    years_of_data = len(stock_data)
    profit_margin = profit / revenue if revenue > 0 else 0
    
    # ================= 3. KPI THẺ HIỆN ĐẠI =================
    st.markdown("### CHỈ SỐ TÀI CHÍNH")
    
    kpi_html = f"""
    <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 30px;'>
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; text-align: center; border-left: 4px solid #00c6ff;'>
            <div style='font-size: 12px; color: #9ca3af;'>DOANH THU</div>
            <div style='font-size: 22px; font-weight: bold; color: #00c6ff;'>{revenue:.1f} tỷ</div>
            <div style='font-size: 12px; color: {"#00ff88" if revenue_growth >= 0 else "#ff4444"};'>{revenue_growth:+.1f}%</div>
        </div>
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; text-align: center; border-left: 4px solid #00ff88;'>
            <div style='font-size: 12px; color: #9ca3af;'>LỢI NHUẬN</div>
            <div style='font-size: 22px; font-weight: bold; color: #00ff88;'>{profit:.1f} tỷ</div>
            <div style='font-size: 12px; color: {"#00ff88" if profit_growth >= 0 else "#ff4444"};'>{profit_growth:+.1f}%</div>
        </div>
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; text-align: center; border-left: 4px solid #ffa500;'>
            <div style='font-size: 12px; color: #9ca3af;'>ROE</div>
            <div style='font-size: 22px; font-weight: bold; color: #ffa500;'>{roe:.1f}%</div>
            <div style='font-size: 12px; color: #9ca3af;'>Tỷ suất lợi nhuận</div>
        </div>
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; text-align: center; border-left: 4px solid #ff4444;'>
            <div style='font-size: 12px; color: #9ca3af;'>D/E</div>
            <div style='font-size: 22px; font-weight: bold; color: {"#00ff88" if debt_ratio < 1 else "#ffa500" if debt_ratio < 2 else "#ff4444"};'>{debt_ratio:.2f}</div>
            <div style='font-size: 12px; color: #9ca3af;'>Nợ/Vốn chủ sở hữu</div>
        </div>
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; text-align: center; border-left: 4px solid #9b59b6;'>
            <div style='font-size: 12px; color: #9ca3af;'>DỮ LIỆU</div>
            <div style='font-size: 22px; font-weight: bold; color: #9b59b6;'>{years_of_data} năm</div>
            <div style='font-size: 12px; color: #9ca3af;'>Lịch sử tài chính</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)
    
    # ================= 4. ĐỊNH GIÁ & THANH KHOẢN =================
    col1, col2 = st.columns(2)
    
    with col1:
        market_cap = (price_input * shares_outstanding * 1e6) / 1e9
        pe_ratio = price_input / eps if eps > 0 else 0
        fair_value_pe = eps * industry_pe if eps > 0 else 0
        upside_pe = ((fair_value_pe - price_input) / price_input * 100) if price_input > 0 and fair_value_pe > 0 else 0
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px;'>
            <h4 style='color: #00c6ff; margin-bottom: 15px;'> ĐỊNH GIÁ</h4>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                <span>Market Cap</span>
                <span style='font-weight: bold;'>{market_cap:,.0f} tỷ VND</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                <span>P/E</span>
                <span style='font-weight: bold;'>{pe_ratio:.1f}x</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                <span>Fair Value (P/E)</span>
                <span style='font-weight: bold; color: {"#00ff88" if upside_pe > 0 else "#ff4444"};'>{fair_value_pe:,.0f} VND</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span>Upside</span>
                <span style='font-weight: bold; color: {"#00ff88" if upside_pe > 0 else "#ff4444"};'>{upside_pe:+.1f}%</span>
            </div>
            <div style='margin-top: 15px; padding-top: 10px; border-top: 1px solid #333;'>
                <span style='color: #9ca3af;'>Ngành: {industry}</span>
                <span style='float: right; color: #9ca3af;'>P/E tham khảo: {industry_pe}x</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if avg_volume >= 10000:
            liquidity_score = 15
            liquidity_level = "TỐT"
            liquidity_color = "#00ff88"
        elif avg_volume >= 5000:
            liquidity_score = 10
            liquidity_level = "TRUNG BÌNH"
            liquidity_color = "#ffa500"
        elif avg_volume >= 1000:
            liquidity_score = 5
            liquidity_level = "THẤP"
            liquidity_color = "#ff6666"
        else:
            liquidity_score = 0
            liquidity_level = "RẤT THẤP"
            liquidity_color = "#ff4444"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px;'>
            <h4 style='color: #00c6ff; margin-bottom: 15px;'> THANH KHOẢN</h4>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                <span>Volume TB</span>
                <span style='font-weight: bold;'>{avg_volume:,.0f} CP/ngày</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                <span>Mức độ</span>
                <span style='font-weight: bold; color: {liquidity_color};'>{liquidity_level}</span>
            </div>
            <div style='margin-top: 15px;'>
                <div style='background: #333; border-radius: 10px; height: 8px;'>
                    <div style='background: {liquidity_color}; width: {liquidity_score/15*100}%; height: 8px; border-radius: 10px;'></div>
                </div>
                <div style='text-align: center; margin-top: 8px; color: #9ca3af; font-size: 12px;'>Điểm thanh khoản: {liquidity_score}/15</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ================= 5. CHẤM ĐIỂM DẠNG ĐỒNG HỒ =================
    st.markdown("### HỆ THỐNG CHẤM ĐIỂM CỔ PHIẾU")
    
    # Tính điểm
    score_fundamental = 0
    if roe > 20: score_fundamental += 7
    elif roe > 15: score_fundamental += 5
    elif roe > 10: score_fundamental += 3
    else: score_fundamental += 1
    
    if revenue > 1000: score_fundamental += 6
    elif revenue > 300: score_fundamental += 4
    else: score_fundamental += 1
    
    if profit > 100: score_fundamental += 6
    elif profit > 30: score_fundamental += 4
    else: score_fundamental += 1
    
    if debt_ratio < 0.5: score_fundamental += 6
    elif debt_ratio < 1: score_fundamental += 4
    else: score_fundamental += 1
    score_fundamental = min(25, score_fundamental)
    
    score_growth = 0
    if revenue_growth > 15: score_growth += 10
    elif revenue_growth > 5: score_growth += 6
    else: score_growth += 2
    
    if profit_growth > 15: score_growth += 10
    elif profit_growth > 5: score_growth += 6
    else: score_growth += 2
    score_growth = min(20, score_growth)
    
    score_safety = 25
    if profit_margin < 0.05: score_safety -= 8
    if profit_margin < 0.02: score_safety -= 5
    if debt_ratio > 1: score_safety -= 6
    if debt_ratio > 2: score_safety -= 4
    if profit_growth < -10: score_safety -= 5
    elif profit_growth < 0: score_safety -= 3
    score_safety = max(0, min(25, score_safety))
    
    score_valuation = 0
    if pe_ratio > 0:
        if pe_ratio < 8: score_valuation += 8
        elif pe_ratio < 12: score_valuation += 6
        elif pe_ratio < 18: score_valuation += 4
        elif pe_ratio < 25: score_valuation += 2
        else: score_valuation += 1
    else: score_valuation += 3
    
    if upside_pe > 50: score_valuation += 7
    elif upside_pe > 30: score_valuation += 5
    elif upside_pe > 10: score_valuation += 3
    elif upside_pe > 0: score_valuation += 1
    score_valuation = min(15, score_valuation)
    
    score_liquidity = liquidity_score
    total_score_raw = score_fundamental + score_growth + score_safety + score_valuation + score_liquidity
    
    # Hiển thị điểm dạng vòng tròn
    score_percent = total_score_raw
    if score_percent >= 80: score_grade = "A"
    elif score_percent >= 65: score_grade = "B"
    elif score_percent >= 50: score_grade = "C"
    elif score_percent >= 35: score_grade = "D"
    else: score_grade = "F"
    
    score_color = "#00ff88" if score_percent >= 70 else ("#ffa500" if score_percent >= 50 else "#ff4444")
    
    col_score1, col_score2 = st.columns([1, 2])
    
    with col_score1:
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='position: relative; width: 150px; height: 150px; margin: 0 auto;'>
                <svg width='150' height='150' viewBox='0 0 150 150'>
                    <circle cx='75' cy='75' r='65' fill='none' stroke='#333' stroke-width='12'/>
                    <circle cx='75' cy='75' r='65' fill='none' stroke='{score_color}' stroke-width='12' 
                            stroke-dasharray='{score_percent * 4.08} 408' stroke-dashoffset='0' transform='rotate(-90 75 75)'/>
                </svg>
                <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;'>
                    <div style='font-size: 32px; font-weight: bold; color: {score_color};'>{score_percent:.0f}</div>
                    <div style='font-size: 14px; color: #9ca3af;'>/ 100</div>
                </div>
            </div>
            <div style='margin-top: 10px; font-size: 24px; font-weight: bold; color: {score_color};'>Hạng {score_grade}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_score2:
        st.markdown(f"""
        <div style='background: #1a1a2e; padding: 20px; border-radius: 15px;'>
            <div style='margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span> Fundamental</span>
                    <span>{score_fundamental}/25</span>
                </div>
                <div style='background: #333; border-radius: 5px; height: 6px;'><div style='background: #00c6ff; width: {score_fundamental/25*100}%; height: 6px; border-radius: 5px;'></div></div>
            </div>
            <div style='margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span> Growth</span>
                    <span>{score_growth}/20</span>
                </div>
                <div style='background: #333; border-radius: 5px; height: 6px;'><div style='background: #00ff88; width: {score_growth/20*100}%; height: 6px; border-radius: 5px;'></div></div>
            </div>
            <div style='margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span> Safety</span>
                    <span>{score_safety}/25</span>
                </div>
                <div style='background: #333; border-radius: 5px; height: 6px;'><div style='background: #ffa500; width: {score_safety/25*100}%; height: 6px; border-radius: 5px;'></div></div>
            </div>
            <div style='margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span> Valuation</span>
                    <span>{score_valuation}/15</span>
                </div>
                <div style='background: #333; border-radius: 5px; height: 6px;'><div style='background: #9b59b6; width: {score_valuation/15*100}%; height: 6px; border-radius: 5px;'></div></div>
            </div>
            <div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span> Liquidity</span>
                    <span>{score_liquidity}/15</span>
                </div>
                <div style='background: #333; border-radius: 5px; height: 6px;'><div style='background: #ff4444; width: {score_liquidity/15*100}%; height: 6px; border-radius: 5px;'></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        # ================= 6. BỘ LỌC RỦI RO =================
    st.markdown("### 🚨 BỘ LỌC RỦI RO")
    
    kill_switch_activated = False
    kill_reasons = []
    
    if liquidity_score == 0:
        kill_switch_activated = True
        kill_reasons.append(" Thanh khoản = 0")
    if debt_ratio > 3:
        kill_switch_activated = True
        kill_reasons.append(f" Nợ quá cao (D/E = {debt_ratio:.2f})")
    if profit < 0:
        kill_switch_activated = True
        kill_reasons.append(f" Doanh nghiệp lỗ ({profit:.1f} tỷ)")
    if profit < 5:
        kill_switch_activated = True
        kill_reasons.append(f" Lợi nhuận quá nhỏ ({profit:.1f} tỷ)")
    if revenue < 50:
        kill_switch_activated = True
        kill_reasons.append(f" Doanh thu quá nhỏ ({revenue:.1f} tỷ)")
    
    final_score = total_score_raw
    
    if kill_switch_activated:
        final_score = min(final_score, 45)
        
        st.markdown("#### ⚠️ CÁC YẾU TỐ CẢNH BÁO")
        
        # Tạo 2 cột
        col_w1, col_w2 = st.columns(2)
        
        # Tạo HTML cho từng cảnh báo
        warning_html_left = ""
        warning_html_right = ""
        
        half = len(kill_reasons) // 2 + len(kill_reasons) % 2
        
        for i, reason in enumerate(kill_reasons):
            if i < half:
                warning_html_left += f"""
                <div style='background: rgba(255,68,68,0.15); border-left: 3px solid #ff4444; padding: 10px; margin-bottom: 10px; border-radius: 8px;'>
                    <span style='color: #ff4444; font-size: 14px;'>{reason}</span>
                </div>
                """
            else:
                warning_html_right += f"""
                <div style='background: rgba(255,68,68,0.15); border-left: 3px solid #ff4444; padding: 10px; margin-bottom: 10px; border-radius: 8px;'>
                    <span style='color: #ff4444; font-size: 14px;'>{reason}</span>
                </div>
                """
        
        with col_w1:
            st.markdown(warning_html_left, unsafe_allow_html=True)
        with col_w2:
            st.markdown(warning_html_right, unsafe_allow_html=True)
        
        st.info("📌 **Lưu ý:** Cổ phiếu có dấu hiệu rủi ro cao, cần thận trọng khi tư vấn")
    else:
        st.success("✅ Không phát hiện yếu tố loại trừ - Cổ phiếu đủ điều kiện phân tích")
    
    # ================= 7. DỰ BÁO 3 KỊCH BẢN =================
    st.markdown("### DỰ BÁO TÀI CHÍNH")
    
    recent_growths = []
    for i in range(1, min(4, len(stock_data))):
        if i < len(stock_data):
            curr = stock_data.iloc[-i]['Revenue'] if pd.notna(stock_data.iloc[-i]['Revenue']) else 0
            prev_year = stock_data.iloc[-i-1]['Revenue'] if -i-1 >= -len(stock_data) and pd.notna(stock_data.iloc[-i-1]['Revenue']) else 0
            if prev_year > 0:
                recent_growths.append((curr - prev_year) / prev_year * 100)
    
    avg_growth = np.mean(recent_growths) if recent_growths else revenue_growth
    growth_std = np.std(recent_growths) if len(recent_growths) > 1 else abs(avg_growth) * 0.5
    
    base_growth = min(max(avg_growth, -10), 25)
    bear_growth = max(-15, base_growth - growth_std)
    bull_growth = min(40, base_growth + growth_std)
    
    # Tạo 3 cột - SỬA LỖI: dùng st.columns(3) trước
    forecast_cols = st.columns(3)
    
    with forecast_cols[0]:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #ff4444;'>
            <div style='font-size: 24px; margin-bottom: 10px;'></div>
            <div style='font-size: 18px; font-weight: bold; color: #ff4444;'>KỊCH BẢN XẤU</div>
            <div style='font-size: 28px; font-weight: bold; margin: 10px 0;'>{bear_growth:+.1f}%</div>
            <div style='font-size: 12px; color: #9ca3af;'>Doanh thu dự kiến</div>
            <div style='font-size: 16px; font-weight: bold;'>{(revenue * (1 + bear_growth/100)):.0f} tỷ</div>
            <div style='font-size: 12px; color: #9ca3af; margin-top: 8px;'>Lợi nhuận: {(profit * (1 + bear_growth/100)):.0f} tỷ</div>
        </div>
        """, unsafe_allow_html=True)
    
    with forecast_cols[1]:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px; text-align: center; border: 2px solid #00c6ff;'>
            <div style='font-size: 24px; margin-bottom: 10px;'></div>
            <div style='font-size: 18px; font-weight: bold; color: #00c6ff;'>KỊCH BẢN CƠ SỞ</div>
            <div style='font-size: 28px; font-weight: bold; margin: 10px 0;'>{base_growth:+.1f}%</div>
            <div style='font-size: 12px; color: #9ca3af;'>Doanh thu dự kiến</div>
            <div style='font-size: 16px; font-weight: bold;'>{(revenue * (1 + base_growth/100)):.0f} tỷ</div>
            <div style='font-size: 12px; color: #9ca3af; margin-top: 8px;'>Lợi nhuận: {(profit * (1 + base_growth/100)):.0f} tỷ</div>
        </div>
        """, unsafe_allow_html=True)
    
    with forecast_cols[2]:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #00ff88;'>
            <div style='font-size: 24px; margin-bottom: 10px;'></div>
            <div style='font-size: 18px; font-weight: bold; color: #00ff88;'>KỊCH BẢN TỐT</div>
            <div style='font-size: 28px; font-weight: bold; margin: 10px 0;'>{bull_growth:+.1f}%</div>
            <div style='font-size: 12px; color: #9ca3af;'>Doanh thu dự kiến</div>
            <div style='font-size: 16px; font-weight: bold;'>{(revenue * (1 + bull_growth/100)):.0f} tỷ</div>
            <div style='font-size: 12px; color: #9ca3af; margin-top: 8px;'>Lợi nhuận: {(profit * (1 + bull_growth/100)):.0f} tỷ</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ================= 8. TIN TỨC =================
    st.markdown("### 📰 TIN TỨC LIÊN QUAN")
    
    def get_news_otc(symbol, company_name):
        import feedparser
        import time
        news_list = []
        search_terms = [symbol]
        if company_name and company_name != symbol:
            short_name = company_name.split()[0] if ' ' in company_name else company_name
            search_terms.append(short_name)
        
        for term in search_terms[:2]:
            url = f"https://news.google.com/rss/search?q={term}+cổ+phiếu&hl=vi&gl=VN&ceid=VN:vi"
            try:
                feed = feedparser.parse(url)
                if feed and feed.entries:
                    for entry in feed.entries[:5]:
                        title = entry.title if hasattr(entry, 'title') else ''
                        link = entry.link if hasattr(entry, 'link') else '#'
                        sentiment = "⚪"
                        sentiment_score = 0
                        title_lower = title.lower()
                        if any(kw in title_lower for kw in ["tăng", "lãi", "bứt phá", "kỷ lục"]):
                            sentiment = "🟢"
                            sentiment_score = 1
                        elif any(kw in title_lower for kw in ["giảm", "lỗ", "bán tháo"]):
                            sentiment = "🔴"
                            sentiment_score = -1
                        if not any(n['title'][:50] == title[:50] for n in news_list):
                            news_list.append({'title': title, 'link': link, 'sentiment': sentiment, 'sentiment_score': sentiment_score})
                        if len(news_list) >= 6:
                            break
                time.sleep(0.3)
            except:
                continue
        unique_news = {}
        for news in news_list:
            key = news['title'][:60]
            if key not in unique_news:
                unique_news[key] = news
        return list(unique_news.values())[:5]
    
    otc_news = get_news_otc(sym, company_name)
    
    if otc_news:
        sentiment_sum = sum([n['sentiment_score'] for n in otc_news])
        if sentiment_sum >= 2:
            sentiment_adjustment = 5
        elif sentiment_sum >= 1:
            sentiment_adjustment = 3
        elif sentiment_sum <= -2:
            sentiment_adjustment = -5
        elif sentiment_sum <= -1:
            sentiment_adjustment = -3
        else:
            sentiment_adjustment = 0
        
        st.info(f" Tìm thấy {len(otc_news)} bài báo | Điều chỉnh điểm: {sentiment_adjustment:+.0f}")
        for news in otc_news:
            st.markdown(f"- {news['sentiment']} [{news['title'][:80]}]({news['link']})")
        if len(otc_news) < 3:
            st.caption("💡 OTC ít được báo chí đưa tin, điều này là bình thường")
    else:
        st.info("📭 Không tìm thấy tin tức (OTC ít được báo chí đưa tin)")
        sentiment_adjustment = 0
    
    # ================= TÍNH ĐIỂM CUỐI CÙNG =================
    final_score_adjusted = final_score + sentiment_adjustment
    final_score_adjusted = max(0, min(100, final_score_adjusted))
    
    # ================= HIỂN THỊ ĐIỂM SỐ CUỐI CÙNG =================
    st.markdown("###  ĐIỂM SỐ CUỐI CÙNG")
    
    # Xác định màu sắc và nhãn theo điểm
    if final_score_adjusted >= 70:
        score_color = "#00ff88"
        score_label = "🟢 KHUYẾN NGHỊ CAO"
        score_badge = "🏆 TỐT"
    elif final_score_adjusted >= 50:
        score_color = "#ffa500"
        score_label = "🟡 CẦN CÂN NHẮC"
        score_badge = "⚠️ TRUNG BÌNH"
    else:
        score_color = "#ff4444"
        score_label = "🔴 RỦI RO CAO"
        score_badge = "❌ YẾU"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #1a1a2e 0%, #0f0f23 100%); border-radius: 20px; margin: 20px 0; border: 1px solid {score_color};'>
        <div style='display: inline-block; background: {score_color}; padding: 5px 15px; border-radius: 20px; margin-bottom: 15px;'>
            <span style='color: #000; font-weight: bold; font-size: 14px;'>{score_badge}</span>
        </div>
        <div style='font-size: 72px; font-weight: bold; color: {score_color};'>{final_score_adjusted:.1f}</div>
        <div style='font-size: 18px; font-weight: bold; color: {score_color}; margin: 10px 0;'>{score_label}</div>
        <div style='display: flex; justify-content: center; gap: 20px; margin-top: 15px;'>
            <div style='background: rgba(255,255,255,0.05); padding: 8px 15px; border-radius: 10px;'>
                <span style='color: #9ca3af; font-size: 12px;'>Điểm gốc</span>
                <div style='color: #00c6ff; font-size: 18px; font-weight: bold;'>{final_score:.1f}</div>
            </div>
            <div style='background: rgba(255,255,255,0.05); padding: 8px 15px; border-radius: 10px;'>
                <span style='color: #9ca3af; font-size: 12px;'>Điều chỉnh</span>
                <div style='color: {"#00ff88" if sentiment_adjustment >= 0 else "#ff4444"}; font-size: 18px; font-weight: bold;'>{sentiment_adjustment:+.0f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ================= TỶ TRỌNG ĐỀ XUẤT =================
    st.markdown("### TỶ TRỌNG ĐẦU TƯ ĐỀ XUẤT")
    
    if final_score_adjusted >= 70:
        suggested_weight = 15 if risk_profile == "Cao" else (10 if risk_profile == "Trung bình" else 5)
    elif final_score_adjusted >= 50:
        suggested_weight = 10 if risk_profile == "Cao" else (5 if risk_profile == "Trung bình" else 3)
    else:
        suggested_weight = 0
    
    # Giới hạn tỷ trọng
    max_position = 20
    if liquidity_score < 5:
        max_position = 10
    if debt_ratio > 2:
        max_position = min(max_position, 5)
    suggested_weight = min(suggested_weight, max_position)
    
    # Tính số tiền cụ thể
    suggested_amount = capital * suggested_weight / 100
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 25px; border-radius: 15px; margin: 15px 0;'>
        <div style='display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap; gap: 15px;'>
            <div style='flex: 1; min-width: 120px;'>
                <div style='display: flex; align-items: center; justify-content: center; gap: 5px;'>
                    <span style='font-size: 20px;'></span>
                    <span style='font-size: 12px; color: #9ca3af;'>TỶ TRỌNG</span>
                </div>
                <div style='font-size: 32px; font-weight: bold; color: #00c6ff;'>{suggested_weight}%</div>
                <div style='background: #333; border-radius: 10px; height: 6px; margin-top: 8px;'>
                    <div style='background: #00c6ff; width: {suggested_weight}%; height: 6px; border-radius: 10px;'></div>
                </div>
            </div>
            <div style='flex: 1; min-width: 120px;'>
                <div style='display: flex; align-items: center; justify-content: center; gap: 5px;'>
                    <span style='font-size: 20px;'></span>
                    <span style='font-size: 12px; color: #9ca3af;'>SỐ TIỀN</span>
                </div>
                <div style='font-size: 28px; font-weight: bold; color: #00ff88;'>{suggested_amount:,.0f} VND</div>
                <div style='font-size: 12px; color: #9ca3af;'>/ {capital:,.0f} VND</div>
            </div>
            <div style='flex: 1; min-width: 120px;'>
                <div style='display: flex; align-items: center; justify-content: center; gap: 5px;'>
                    <span style='font-size: 20px;'></span>
                    <span style='font-size: 12px; color: #9ca3af;'>GIỚI HẠN</span>
                </div>
                <div style='font-size: 32px; font-weight: bold; color: #ffa500;'>≤{max_position}%</div>
                <div style='font-size: 12px; color: #9ca3af;'>tổng tài sản</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
        # ================= PHÂN BỔ DANH MỤC THEO KHẨU VỊ =================
    st.markdown("### PHÂN BỔ DANH MỤC ĐẦU TƯ THEO KHẨU VỊ KHÁCH HÀNG")
    
    if risk_profile == "Thấp":
        max_otc_allocation = 10
        recommended_cash = 70
        recommended_bonds = 20
        cash_name = "Tiền mặt / Chứng chỉ tiền gửi"
        bonds_name = "Trái phiếu / Cổ phiếu Bluechip"
        otc_name = f"OTC {sym}"
    elif risk_profile == "Trung bình":
        max_otc_allocation = 15
        recommended_cash = 50
        recommended_bonds = 35
        cash_name = "Tiền mặt / Chứng chỉ tiền gửi"
        bonds_name = "Cổ phiếu niêm yết"
        otc_name = f"OTC {sym}"
    else:
        max_otc_allocation = 20
        recommended_cash = 30
        recommended_bonds = 50
        cash_name = "Tiền mặt linh hoạt"
        bonds_name = "Cổ phiếu tăng trưởng"
        otc_name = f"OTC {sym} (Rủi ro cao)"
    
    # Điều chỉnh theo điểm OTC
    if final_score_adjusted >= 70:
        adjusted_otc = min(max_otc_allocation, 15)
    elif final_score_adjusted >= 55:
        adjusted_otc = min(max_otc_allocation, 12)
    elif final_score_adjusted >= 45:
        adjusted_otc = max(5, min(max_otc_allocation, 8))
    else:
        adjusted_otc = 0
    
    # Điều chỉnh lại tỷ trọng
    if adjusted_otc > 0:
        remaining = 100 - adjusted_otc
        adjusted_cash = remaining * recommended_cash / (recommended_cash + recommended_bonds)
        adjusted_bonds = remaining - adjusted_cash
    else:
        adjusted_cash = 70
        adjusted_bonds = 30
        adjusted_otc = 0
    
    # Tính số tiền cụ thể
    cash_amount = capital * adjusted_cash / 100
    bonds_amount = capital * adjusted_bonds / 100
    otc_amount = capital * adjusted_otc / 100
    
    # Tạo 2 cột: 1 cho Pie Chart, 1 cho bảng chi tiết
    col_pie, col_table = st.columns([1, 1])
    
    with col_pie:
        # Vẽ Pie Chart - ĐÃ SỬA LEGEND
        fig_pie = go.Figure(data=[go.Pie(
            labels=[
                f"{cash_name}",
                f"{bonds_name}",
                f"{otc_name}"
            ],
            values=[adjusted_cash, adjusted_bonds, adjusted_otc],
            hole=0.4,
            marker=dict(colors=['#00c6ff', '#00ff88', '#ffa500']),
            textinfo='percent',
            textposition='auto',
            textfont=dict(size=12, color='white'),
            hovertemplate='%{label}<br>Tỷ trọng: %{percent:.1f}%<br>Giá trị: %{value:,.0f} VND<extra></extra>',
            showlegend=True
        )])
        
        fig_pie.update_layout(
            title=dict(
                text=f" PHÂN BỔ {capital:,.0f} VND THEO KHẨU VỊ {risk_profile.upper()}",
                font=dict(size=14, color='#00c6ff'),
                x=0.5
            ),
            template="plotly_dark",
            height=450,
            # Điều chỉnh legend ra ngoài bên phải, không bị dính
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=11, color='#9ca3af'),
                bgcolor='rgba(0,0,0,0)'
            ),
            # Thêm margin để legend không bị cắt
            margin=dict(l=20, r=150, t=50, b=20),
            annotations=[dict(
                text=f"Tổng vốn<br>{capital:,.0f} VND",
                x=0.5, y=0.5, font_size=14, showarrow=False,
                font=dict(color='white', size=14, weight='bold')
            )],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_table:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px; height: 100%;'>
            <h4 style='color: #00c6ff; margin-bottom: 20px; text-align: center;'> CHI TIẾT PHÂN BỔ</h4>
            <div style='margin-bottom: 15px;'>
                <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333;'>
                    <span style='color: #00c6ff;'>🏦 {cash_name}</span>
                    <div style='text-align: right;'>
                        <div><strong>{adjusted_cash:.1f}%</strong></div>
                        <div style='color: #00ff88; font-size: 14px;'>{cash_amount:,.0f} VND</div>
                    </div>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333;'>
                    <span style='color: #00ff88;'>📊 {bonds_name}</span>
                    <div style='text-align: right;'>
                        <div><strong>{adjusted_bonds:.1f}%</strong></div>
                        <div style='color: #00ff88; font-size: 14px;'>{bonds_amount:,.0f} VND</div>
                    </div>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #ffa500;'>
                    <span style='color: #ffa500;'>⚡ {otc_name}</span>
                    <div style='text-align: right;'>
                        <div><strong>{adjusted_otc:.1f}%</strong></div>
                        <div style='color: #ffa500; font-size: 14px;'>{otc_amount:,.0f} VND</div>
                    </div>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 12px 0; margin-top: 10px; background: #0f0f23; border-radius: 8px;'>
                    <span style='font-weight: bold;'> TỔNG CỘNG</span>
                    <div style='text-align: right;'>
                        <div><strong>100%</strong></div>
                        <div style='color: #00c6ff; font-size: 16px; font-weight: bold;'>{capital:,.0f} VND</div>
                    </div>
                </div>
            </div>
            <div style='margin-top: 15px; padding: 10px; background: #0f0f23; border-radius: 8px; text-align: center;'>
                <p style='color: #9ca3af; font-size: 12px; margin: 0;'>
                    ⚠️ OTC không nên vượt quá <strong style='color: #ffa500;'>{max_otc_allocation}%</strong> tổng tài sản theo khẩu vị {risk_profile}
                </p>
                <p style='color: #9ca3af; font-size: 11px; margin: 5px 0 0 0;'>
                    📌 Số tiền OTC đề xuất: <strong style='color: #00ff88;'>{adjusted_otc:.0f}%</strong> = <strong style='color: #00ff88;'>{otc_amount:,.0f} VND</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Hiển thị thanh tiến trình phân bổ
    st.markdown("### BIỂU ĐỒ PHÂN BỔ THEO TỶ LỆ")
    
    col_bar1, col_bar2, col_bar3 = st.columns(3)
    
    with col_bar1:
        st.progress(adjusted_cash/100, text=f"🏦 {cash_name}: {adjusted_cash:.0f}%")
    with col_bar2:
        st.progress(adjusted_bonds/100, text=f"📊 {bonds_name}: {adjusted_bonds:.0f}%")
    with col_bar3:
        st.progress(adjusted_otc/100, text=f"⚡ {otc_name}: {adjusted_otc:.0f}%")
    
    st.caption(f"⚠️ OTC không nên vượt quá {max_otc_allocation}% tổng tài sản theo khẩu vị {risk_profile}")
    st.markdown("---")
    
    # ================= NHẬN ĐỊNH TƯ VẤN VIÊN =================
    st.markdown("---")
    st.markdown("### ✍️ NHẬN ĐỊNH CỦA TƯ VẤN VIÊN")
    
    if 'advisor_note' not in st.session_state:
        st.session_state.advisor_note = ""
    if 'override_option' not in st.session_state:
        st.session_state.override_option = "Theo gợi ý hệ thống"
    if 'confirmed' not in st.session_state:
        st.session_state.confirmed = False
    
    col_n1, col_n2 = st.columns([2, 1])
    
    with col_n1:
        advisor_note = st.text_area(
            "📝 Ghi chú / Nhận xét:",
            value=st.session_state.advisor_note,
            height=100,
            placeholder="Ví dụ: Khách hàng có thể chấp nhận rủi ro cao hơn do tuổi trẻ..."
        )
        st.session_state.advisor_note = advisor_note
        
        override_option = st.selectbox(
            "⚙️ Điều chỉnh khuyến nghị:",
            ["Theo gợi ý hệ thống", "Tăng 1 bậc", "Giảm 1 bậc", "Chờ thêm thông tin"]
        )
        st.session_state.override_option = override_option
        
        confirmed = st.checkbox(" Tôi đã xem xét và hiểu rủi ro của cổ phiếu này")
        st.session_state.confirmed = confirmed
        
        if st.button("💾 Lưu nhận định", key="save_advisor_note"):
            if confirmed:
                st.success("✅ Đã lưu nhận định!")
            else:
                st.warning("⚠️ Vui lòng xác nhận đã xem xét rủi ro")
    
    with col_n2:
        st.markdown("""
        <div style='background: #1f2937; padding: 15px; border-radius: 10px;'>
            <p style='color: #00c6ff; font-weight: bold;'>📋 Trách nhiệm TVV:</p>
            <ul style='color: #9ca3af; font-size: 12px; padding-left: 20px;'>
                <li>Xác minh thông tin DN</li>
                <li>Đánh giá phù hợp KH</li>
                <li>Giải thích rủi ro OTC</li>
                <li>Chịu trách nhiệm cuối cùng</li>
            </ul>
            <hr>
            <p style='color: #ffa500; font-size: 11px;'>⚠️ Hệ thống chỉ hỗ trợ phân tích</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ================= XUẤT BÁO CÁO & FEEDBACK =================
    col_x1, col_x2 = st.columns(2)
    
    with col_x1:
        st.markdown("### 📄 XUẤT BÁO CÁO")
        
        report_content = f"""
        ===================================
        HVA ROBO ADVISOR AI - BÁO CÁO TƯ VẤN
        ===================================
        Ngày: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        
        👤 KHÁCH HÀNG
        ------------------------
        Tên: {client_name if client_name else "Khách hàng"}
        Tuổi: {age}
        Khẩu vị: {risk_profile}
        Vốn: {capital:,.0f} VND
        
        🔖 CỔ PHIẾU
        ------------------------
        Mã: {sym}
        Tên: {company_name}
        Giá: {price_input:,.0f} VND
        Ngành: {industry}
        Thanh khoản: {avg_volume:,.0f} CP/ngày
        
        CÁC CHỈ SỐ
        ------------------------
        Doanh thu: {revenue:.1f} tỷ ({revenue_growth:+.1f}%)
        Lợi nhuận: {profit:.1f} tỷ ({profit_growth:+.1f}%)
        ROE: {roe:.1f}%
        D/E: {debt_ratio:.2f}
        P/E: {pe_ratio:.1f}x
        
        ĐIỂM OTC: {final_score_adjusted:.1f}/100
        ------------------------
        Fundamental: {score_fundamental}/25
        Growth: {score_growth}/20
        Safety: {score_safety}/25
        Valuation: {score_valuation}/15
        Liquidity: {score_liquidity}/15
        
        KHUYẾN NGHỊ
        ------------------------
        Tỷ trọng: {suggested_weight}% ({capital * suggested_weight / 100:,.0f} VND)
        """
        
        if advisor_note:
            report_content += f"\n\n📝 TVV NHẬN XÉT: {advisor_note}\nĐiều chỉnh: {override_option}"
        
        st.download_button(
            label="📥 Tải báo cáo (TXT)",
            data=report_content,
            file_name=f"OTC_{sym}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_x2:
        st.markdown("### 📝 FEEDBACK")
        
        actual_result = st.selectbox(
            "Kết quả sau 3-6 tháng:",
            ["Chưa có", "Lãi >15%", "Lãi 10-15%", "Lãi 5-10%", "Hòa vốn", "Lỗ 5-10%", "Lỗ >10%"]
        )
        
        if st.button(" Ghi nhận phản hồi", use_container_width=True):
            if actual_result != "Chưa có":
                import json
                fb_file = "feedback_log.json"
                fb_data = []
                if os.path.exists(fb_file):
                    with open(fb_file, 'r', encoding='utf-8') as f:
                        try:
                            fb_data = json.load(f)
                        except:
                            pass
                fb_data.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': sym,
                    'score': final_score_adjusted,
                    'result': actual_result
                })
                with open(fb_file, 'w', encoding='utf-8') as f:
                    json.dump(fb_data, f, ensure_ascii=False, indent=2)
                st.success(" Đã ghi nhận!")
            else:
                st.info(" Hãy quay lại sau 3-6 tháng")
    
    # ================= LỊCH SỬ PHẢN HỒI =================
    fb_file = "feedback_log.json"
    if os.path.exists(fb_file):
        with open(fb_file, 'r', encoding='utf-8') as f:
            try:
                fb_data = json.load(f)
                if len(fb_data) > 0:
                    with st.expander("📊 Lịch sử phản hồi"):
                        for fb in fb_data[-5:]:
                            st.write(f"- {fb['timestamp']} | {fb['symbol']} | Kết quả: {fb['result']}")
            except:
                pass
    
    # ================= CẢNH BÁO CUỐI =================
    st.markdown("---")
    st.warning("""
    ⚠️ **CẢNH BÁO RỦI RO OTC**
    
    - Dữ liệu tài chính KHÔNG được kiểm toán độc lập
    - Thanh khoản thấp, khó bán khi cần
    - Chỉ dành cho nhà đầu tư chấp nhận rủi ro cao
    - **Không nên phân bổ quá 10-20% tổng vốn vào OTC**
    
    *Đây là công cụ hỗ trợ phân tích, không phải lời khuyên đầu tư chính thức*
    """)
# ================= MODE 4: OTC PORTFOLIO ROBO-ADVISOR (PRO MAX ULTIMATE) =================
elif mode == "Phân tích danh mục OTC tự động":
    
    # ================= LẤY DỮ LIỆU TỪ SESSION STATE =================
    client_name = st.session_state.get('client_name', 'Khách hàng')
    age = st.session_state.get('age', 'N/A')
    risk_profile = st.session_state['risk_profile']
    capital = st.session_state.get('capital', 100000000)
    
    # ================= 1. THU THẬP DỮ LIỆU =================
    with st.spinner(" Đang kết nối và thu thập dữ liệu OTC..."):
        try:
            import os
            possible_paths = ["OTC_data_clean.csv", "./OTC_data_clean.csv", "../OTC_data_clean.csv", "data/OTC_data_clean.csv"]
            
            otc_df = None
            for path in possible_paths:
                if os.path.exists(path):
                    otc_df = pd.read_csv(path)
                    break

            if otc_df is None:
                uploaded_file = st.file_uploader(" Tải file dữ liệu OTC", type=['csv'], key="otc_portfolio_upload")
                if uploaded_file is not None:
                    otc_df = pd.read_csv(uploaded_file)
                else:
                    st.warning("⚠️ Vui lòng tải file dữ liệu OTC để tiếp tục")
                    st.stop()
            
            # Dữ liệu CSV đang ở đơn vị TRIỆU đồng
            st.info(" *Dữ liệu CSV: Doanh thu và lợi nhuận đang ở đơn vị TRIỆU đồng*")
            
            # Chuẩn hóa dữ liệu
            otc_df = otc_df.replace([np.inf, -np.inf], np.nan)
            numeric_cols = ['Revenue', 'GrossProfit', 'NetProfit', 'OperatingProfit', 
                            'TotalAssets', 'CurrentAssets', 'Debt', 'ShortTermDebt', 
                            'Equity', 'EPS', 'BVPS', 'PE', 'ROS', 'ROEA', 'ROAA']
            for col in numeric_cols:
                if col in otc_df.columns:
                    otc_df[col] = pd.to_numeric(otc_df[col], errors='coerce')
            
            # ================= LIQUIDITY ƯỚC LƯỢNG =================
            if 'Volume' not in otc_df.columns:
                otc_df['Volume'] = otc_df['Revenue'].apply(lambda x: np.log1p(x / 1000) * 5000 if pd.notna(x) and x > 0 else 1000)
                otc_df['Volume'] = otc_df['Volume'].clip(500, 30000).astype(int)
                st.caption(" *Chỉ số thanh khoản được ước lượng từ quy mô doanh thu (logarithmic scale)*")
            
        except Exception as e:
            st.error(f"❌ Lỗi đọc dữ liệu: {e}")
            st.stop()
    
    # ================= 2. THAM SỐ THAM KHẢO =================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; margin: 15px 0;'>
        <h3 style='color: #00c6ff; margin: 0;'> CẤU HÌNH PHÂN TÍCH</h3>
        <p style='color: #9ca3af; margin: 5px 0 0 0;'>Soft filter: Chỉ loại cổ phiếu có doanh thu ≤ 0 hoặc vốn chủ ≤ 0</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_param1, col_param2, col_param3 = st.columns(3)
    
    with col_param1:
        with st.container():
            st.markdown("""
            <div style='background: #1f2937; padding: 12px; border-radius: 10px; text-align: center;'>
                <p style='color: #00c6ff; font-weight: bold; margin: 0;'> BỘ LỌC CƠ BẢN</p>
                <hr style='margin: 8px 0;'>
                <p style='margin: 0;'>Doanh thu: <span style='color: #00ff88;'>> 0 (triệu)</span></p>
                <p style='margin: 0;'>Vốn chủ: <span style='color: #00ff88;'>> 0 (triệu)</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_param2:
        with st.container():
            st.markdown("""
            <div style='background: #1f2937; padding: 12px; border-radius: 10px; text-align: center;'>
                <p style='color: #00c6ff; font-weight: bold; margin: 0;'> THAM SỐ MPT</p>
                <hr style='margin: 8px 0;'>
                <p style='margin: 0;'>Expected Return: <span style='color: #00ff88;'>5% - 20%</span></p>
                <p style='margin: 0;'>Risk-free rate: <span style='color: #00ff88;'>3%</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_param3:
        min_years = st.number_input(" Số năm dữ liệu tối thiểu", min_value=1, value=3, step=1, key="otc_min_years")
        max_otc_allocation = st.slider(" Tỷ trọng OTC tối đa", 0, 30, 20, step=5, key="otc_max_allocation")
    
    # ================= 3. MÔ HÌNH CHẤM ĐIỂM CỔ PHIẾU =================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; margin: 15px 0;'>
        <h3 style='color: #00c6ff; margin: 0;'> CHẤM ĐIỂM CỔ PHIẾU OTC</h3>
        <p style='color: #9ca3af; margin: 5px 0 0 0;'>Tiêu chí: ROE | Tăng trưởng | An toàn tài chính | Quy mô | Thanh khoản</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ================= THÊM GIẢI THÍCH BƯỚC 1 =================
    with st.expander(" Xem chi tiết cách chấm điểm"):
        st.markdown("""
        <style>
        .scoring-table th, .scoring-table td { padding: 8px; text-align: left; }
        </style>
        <table class='scoring-table' style='width:100%; border-collapse: collapse;'>
            <tr style='background: #00c6ff; color: #000;'>
                <th>Tiêu chí</th><th>Thang điểm</th><th>Cách tính</th>
            </tr>
            <tr style='background: #1a1a2e;'><td><b>ROE</b></th><td>0-25 điểm</td><td>≥30%:25đ, ≥20%:20đ, ≥15%:15đ, ≥10%:12đ, ≥5%:8đ, >0%:5đ, âm:2đ</td></tr>
            <tr style='background: #0f0f23;'><td><b>Tăng trưởng CAGR</b></th><td>0-25 điểm</td><td>≥25%:25đ, ≥15%:20đ, ≥10%:15đ, ≥5%:10đ, >0%:5đ, ≤0%:1đ</td></tr>
            <tr style='background: #1a1a2e;'><td><b>An toàn D/E</b></th><td>0-20 điểm</td><td>≤0.5:20đ, ≤1.0:15đ, ≤1.5:10đ, ≤2.0:6đ, ≤3.0:3đ, >3.0:1đ</td></tr>
            <tr style='background: #0f0f23;'><td><b>Quy mô doanh thu</b></th><td>0-15 điểm</td><td>≥1000 tỷ:15đ, ≥500 tỷ:12đ, ≥200 tỷ:9đ, ≥100 tỷ:6đ, ≥50 tỷ:4đ, ≥10 tỷ:2đ</td></tr>
            <tr style='background: #1a1a2e;'><td><b>Thanh khoản</b></th><td>0-15 điểm</td><td>≥20.000:15đ, ≥10.000:12đ, ≥5.000:9đ, ≥2.000:6đ, ≥1.000:3đ</td></tr>
        </table>
        <p style='margin-top: 10px; color: #00ff88;'> Điểm càng cao → Cổ phiếu càng tốt → Được ưu tiên chọn vào danh mục</p>
        """, unsafe_allow_html=True)
    
    with st.spinner("🔍 Đang phân tích và chấm điểm toàn bộ cổ phiếu OTC..."):
        
        all_scores = []
        
        for sym in otc_df['Symbol'].unique():
            stock_data = otc_df[otc_df['Symbol'] == sym].copy()
            
            if len(stock_data) < min_years:
                continue
            
            # Lấy dữ liệu mới nhất
            latest = stock_data.iloc[-1]
            first = stock_data.iloc[0]
            
            # Đơn vị: TRIỆU đồng
            revenue = latest.get('Revenue', 0)
            profit = latest.get('NetProfit', 0)
            roe = latest.get('ROEA', 0)
            debt = latest.get('Debt', 0)
            equity = latest.get('Equity', 1)
            debt_ratio = debt / equity if equity > 0 else 999
            volume = latest.get('Volume', 1000)
            
            # ================= SOFT FILTER =================
            if revenue <= 0:
                continue
            if equity <= 0:
                continue
            if roe < -100 or roe > 100:
                continue
            
            # ===== TÍNH ĐIỂM =====
            score = 0
            
            # 1. ROE (0-25)
            if roe >= 30: score += 25
            elif roe >= 20: score += 20
            elif roe >= 15: score += 15
            elif roe >= 10: score += 12
            elif roe >= 5: score += 8
            elif roe > 0: score += 5
            else: score += 2
            
            # 2. Tăng trưởng CAGR (0-25)
            first_rev = first.get('Revenue', revenue)
            last_rev = revenue
            years = len(stock_data) - 1
            cagr = 0
            if first_rev > 0 and years > 0:
                cagr = (last_rev / first_rev) ** (1/years) - 1
            
            if cagr >= 0.25: score += 25
            elif cagr >= 0.15: score += 20
            elif cagr >= 0.10: score += 15
            elif cagr >= 0.05: score += 10
            elif cagr > 0: score += 5
            else: score += 1
            
            # 3. D/E (0-20)
            if debt_ratio <= 0.5: score += 20
            elif debt_ratio <= 1.0: score += 15
            elif debt_ratio <= 1.5: score += 10
            elif debt_ratio <= 2.0: score += 6
            elif debt_ratio <= 3.0: score += 3
            else: score += 1
            
            # 4. Quy mô doanh thu (0-15) - đơn vị triệu
            if revenue >= 1000000: score += 15
            elif revenue >= 500000: score += 12
            elif revenue >= 200000: score += 9
            elif revenue >= 100000: score += 6
            elif revenue >= 50000: score += 4
            elif revenue >= 10000: score += 2
            else: score += 1
            
            # 5. Thanh khoản (0-15)
            if volume >= 20000: score += 15
            elif volume >= 10000: score += 12
            elif volume >= 5000: score += 9
            elif volume >= 2000: score += 6
            elif volume >= 1000: score += 3
            else: score += 1
            
            score = min(100, score)
            
            company_name = sym
            if 'CompanyName' in otc_df.columns:
                company_row = otc_df[otc_df['Symbol'] == sym]
                if not company_row.empty:
                    company_name = company_row['CompanyName'].iloc[0]
            
            all_scores.append({
                "Symbol": sym,
                "CompanyName": company_name[:50] if len(company_name) > 50 else company_name,
                "Revenue": revenue,
                "Profit": profit,
                "ROE": roe,
                "D/E": debt_ratio,
                "Volume": volume,
                "CAGR": cagr,
                "Years": len(stock_data),
                "Score": score
            })
    
    if not all_scores:
        st.warning("⚠️ Không có cổ phiếu OTC nào đáp ứng bộ lọc cơ bản.")
        st.stop()
    
    score_df = pd.DataFrame(all_scores)
    score_df = score_df.sort_values('Score', ascending=False).reset_index(drop=True)
    
    # ================= HIỂN THỊ KẾT QUẢ BƯỚC 1 =================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; margin: 15px 0;'>
        <h3 style='color: #00c6ff; margin: 0;'> KẾT QUẢ CHẤM ĐIỂM</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric(" Tổng số CP", len(all_scores), delta=None)
    with col_stat2:
        st.metric(" Điểm cao nhất", f"{score_df['Score'].max():.0f}", delta=None)
    with col_stat3:
        st.metric(" Điểm TB", f"{score_df['Score'].mean():.1f}", delta=None)
    with col_stat4:
        st.metric(" Điểm thấp nhất", f"{score_df['Score'].min():.0f}", delta=None)
    
    st.markdown("#### 🏆 TOP 10 CỔ PHIẾU ĐIỂM CAO NHẤT")
    top10_display = score_df.head(10)[['Symbol', 'CompanyName', 'Revenue', 'Profit', 'ROE', 'D/E', 'Volume', 'CAGR', 'Score']].copy()
    top10_display['Revenue'] = top10_display['Revenue'].apply(lambda x: f"{x/1000:,.0f}" if x >= 1000 else f"{x:,.0f}")
    top10_display['Profit'] = top10_display['Profit'].apply(lambda x: f"{x/1000:,.1f}" if abs(x) >= 1000 else f"{x:,.0f}")
    top10_display['ROE'] = top10_display['ROE'].apply(lambda x: f"{x:.1f}%")
    top10_display['D/E'] = top10_display['D/E'].apply(lambda x: f"{x:.2f}")
    top10_display['Volume'] = top10_display['Volume'].apply(lambda x: f"{x:,.0f}")
    top10_display['CAGR'] = top10_display['CAGR'].apply(lambda x: f"{x*100:.1f}%")
    top10_display['Score'] = top10_display['Score'].apply(lambda x: f"{x:.0f}")
    
    st.dataframe(
        top10_display,
        column_config={
            "Symbol": "Mã CK",
            "CompanyName": "Tên công ty",
            "Revenue": "DT (tỷ)",
            "Profit": "LN (tỷ)",
            "ROE": "ROE",
            "D/E": "D/E",
            "Volume": "Liquidity*",
            "CAGR": "Tăng trưởng",
            "Score": "Điểm"
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Phân tích kết quả Bước 1
    avg_score = score_df['Score'].mean()
    high_score_count = len(score_df[score_df['Score'] >= 70])
    medium_score_count = len(score_df[(score_df['Score'] >= 50) & (score_df['Score'] < 70)])
    low_score_count = len(score_df[score_df['Score'] < 50])
    
    if avg_score >= 60:
        st.success(f" **Đánh giá TỐT:** Điểm TB {avg_score:.1f}/100 | Tốt: {high_score_count} | TB: {medium_score_count} | Yếu: {low_score_count}")
    elif avg_score >= 45:
        st.info(f" **Đánh giá TRUNG BÌNH:** Điểm TB {avg_score:.1f}/100 | Tốt: {high_score_count} | TB: {medium_score_count} | Yếu: {low_score_count}")
    else:
        st.warning(f" **Đánh giá YẾU:** Điểm TB {avg_score:.1f}/100 | Tốt: {high_score_count} | TB: {medium_score_count} | Yếu: {low_score_count}")
    
    # ================= 4. LỰA CHỌN TOP N =================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; margin: 15px 0;'>
        <h3 style='color: #00c6ff; margin: 0;'> LỰA CHỌN TOP N CỔ PHIẾU OTC </h3>
        <p style='color: #9ca3af; margin: 5px 0 0 0;'> Chiến lược Robo-Advisor: Luôn chọn top N cổ phiếu tốt nhất</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_hist, col_info = st.columns([2, 1])
    
    with col_hist:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=score_df['Score'],
            nbinsx=20,
            marker_color='#00c6ff',
            marker_line_color='white',
            marker_line_width=1,
            opacity=0.85,
            hovertemplate='Điểm: %{x}<br>Số lượng: %{y}<extra></extra>'
        ))
        
        # Thêm đường trung bình
        avg_score_val = score_df['Score'].mean()
        fig_hist.add_vline(
            x=avg_score_val, 
            line_dash="dash", 
            line_color="#ffa500", 
            line_width=2,
            annotation_text=f"TB: {avg_score_val:.1f}",
            annotation_position="top",
            annotation_font=dict(color="#ffa500", size=11)
        )
        
        fig_hist.update_layout(
            title=dict(
                text=" PHÂN PHỐI ĐIỂM SỐ OTC",
                font=dict(size=16, color='#00c6ff', weight='bold'),
                x=0.5
            ),
            xaxis_title="Điểm số",
            yaxis_title="Số lượng cổ phiếu",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af'),
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.1)',
                title_font=dict(size=12, color='#00c6ff')
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255,255,255,0.1)',
                title_font=dict(size=12, color='#00c6ff')
            ),
            bargap=0.05
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_info:
        st.metric(" Tổng số CP", len(score_df))
        st.metric(" Điểm cao nhất", f"{score_df['Score'].max():.0f}")
        st.metric(" Điểm TB", f"{score_df['Score'].mean():.1f}")
        st.metric(" Điểm thấp nhất", f"{score_df['Score'].min():.0f}")
    
    max_top_n = min(15, len(score_df))
    top_n = st.slider(" Số lượng cổ phiếu chọn vào danh mục (top N)", 3, max_top_n, 8, step=1)
    
    selected_stocks = score_df.head(top_n).copy()
    selected_stocks = selected_stocks.reset_index(drop=True)
    
    st.success(f" Đã chọn **{top_n}** cổ phiếu điểm cao nhất (Top {top_n}/{len(score_df)})")
    
    # Hiển thị bảng
    display_cols = ['Symbol', 'CompanyName', 'Revenue', 'Profit', 'ROE', 'D/E', 'Volume', 'CAGR', 'Score']
    display_df = selected_stocks[display_cols].copy()
    
    display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f"{x/1000:,.0f}" if x >= 1000 else f"{x:,.0f}")
    display_df['Profit'] = display_df['Profit'].apply(lambda x: f"{x/1000:,.1f}" if abs(x) >= 1000 else f"{x:,.0f}")
    display_df['ROE'] = display_df['ROE'].apply(lambda x: f"{x:.1f}%")
    display_df['D/E'] = display_df['D/E'].apply(lambda x: f"{x:.2f}")
    display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
    display_df['CAGR'] = display_df['CAGR'].apply(lambda x: f"{x*100:.1f}%")
    display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.0f}")
    
    st.dataframe(
        display_df,
        column_config={
            "Symbol": "Mã CK",
            "CompanyName": "Tên công ty",
            "Revenue": "DT (tỷ)",
            "Profit": "LN (tỷ)",
            "ROE": "ROE",
            "D/E": "D/E",
            "Volume": "Liquidity*",
            "CAGR": "Tăng trưởng",
            "Score": "Điểm"
        },
        use_container_width=True,
        hide_index=True
    )
    
    # ================= 5. TỐI ƯU HÓA DANH MỤC MPT =================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; margin: 15px 0;'>
        <h3 style='color: #00c6ff; margin: 0;'> TỐI ƯU HÓA DANH MỤC (MPT)</h3>
        <p style='color: #9ca3af; margin: 5px 0 0 0;'>Risk-free rate = 3% | Ràng buộc: ≤35%/mã | Không dùng correlation</p>
    </div>
    """, unsafe_allow_html=True)
    
    num_assets = len(selected_stocks)
    
    # ================= EXPECTED RETURN REALISTIC (5-20%) + LIQUIDITY PENALTY =================
    expected_returns = []
    for _, row in selected_stocks.iterrows():
        raw_return = (
            (row['ROE'] / 100) * 0.4 +
            row['CAGR'] * 0.3 +
            (row['Score'] / 100) * 0.3
        )
        
        # Penalty thanh khoản: volume càng thấp, penalty càng cao
        liquidity_penalty = 1 - (1 / np.log1p(row['Volume'] / 1000))
        liquidity_penalty = np.clip(liquidity_penalty, 0.6, 1.0)
        
        expected_return = raw_return * 0.8 * liquidity_penalty
        expected_return = np.clip(expected_return, 0.05, 0.20)
        expected_returns.append(expected_return)
    
    selected_stocks['ExpectedReturn'] = expected_returns
    
    # ================= RISK MODEL CÓ LIQUIDITY PENALTY & SIZE PENALTY =================
    risks = []
    for _, row in selected_stocks.iterrows():
        liquidity_risk = (1 / np.log1p(row['Volume'] / 1000)) * 0.2
        liquidity_risk = np.clip(liquidity_risk, 0.05, 0.25)
        
        size_risk = max(0, 0.1 - (row['Revenue'] / 10000000))
        size_risk = np.clip(size_risk, 0, 0.1)
        
        raw_risk = (
            0.12
            + row['D/E'] * 0.08
            - row['ROE'] * 0.001
            - row['CAGR'] * 0.08
            + liquidity_risk
            + size_risk
        )
        risk = np.clip(raw_risk, 0.12, 0.55)
        risks.append(risk)
    
    selected_stocks['Risk'] = risks
    
    # Không dùng correlation
    variances = np.array(risks) ** 2
    
    # ===== MONTE CARLO SIMULATION =====
    n_portfolios = 5000
    risk_free_rate = 0.03
    
    results = []
    weights_record = []
    
    for i in range(n_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / weights.sum()
        weights = np.minimum(weights, 0.35)
        if np.sum(weights) == 0:
            continue
        weights = weights / weights.sum()
        
        port_return = np.dot(weights, expected_returns)
        port_risk = np.sqrt(np.sum((weights ** 2) * variances))
        sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0
        
        results.append([port_return, port_risk, sharpe])
        weights_record.append(weights)
    
    results = np.array(results)
    
    if len(results) == 0:
        st.warning("⚠️ Không tìm được danh mục hợp lệ, sử dụng phân bổ đều")
        optimal_weights = np.ones(num_assets) / num_assets
        optimal_return = np.dot(optimal_weights, expected_returns)
        optimal_risk = np.sqrt(np.sum((optimal_weights ** 2) * variances))
        optimal_sharpe = (optimal_return - risk_free_rate) / optimal_risk if optimal_risk > 0 else 0
        
        min_risk_weights = optimal_weights
        min_risk_return = optimal_return
        min_risk_risk = optimal_risk
    else:
        optimal_idx = np.argmax(results[:, 2])
        optimal_weights = weights_record[optimal_idx]
        optimal_return = results[optimal_idx, 0]
        optimal_risk = results[optimal_idx, 1]
        optimal_sharpe = results[optimal_idx, 2]
        
        min_risk_idx = np.argmin(results[:, 1])
        min_risk_weights = weights_record[min_risk_idx]
        min_risk_return = results[min_risk_idx, 0]
        min_risk_risk = results[min_risk_idx, 1]
    
    # ===== EFFICIENT FRONTIER =====
    if len(results) > 0:
        fig_ef = go.Figure()
        
        fig_ef.add_trace(go.Scatter(
            x=results[:, 1], y=results[:, 0],
            mode='markers',
            marker=dict(
                size=5,
                color=results[:, 2],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Danh mục ngẫu nhiên',
            hovertemplate='Rủi ro: %{x:.2%}<br>Lợi nhuận: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>'
        ))
        
        fig_ef.add_trace(go.Scatter(
            x=[optimal_risk], y=[optimal_return],
            mode='markers',
            marker=dict(color='red', size=18, symbol='star'),
            name=f'🌟 Tối ưu (Sharpe: {optimal_sharpe:.2f})'
        ))
        
        fig_ef.add_trace(go.Scatter(
            x=[min_risk_risk], y=[min_risk_return],
            mode='markers',
            marker=dict(color='yellow', size=12, symbol='circle'),
            name=f'🛡️ Rủi ro thấp nhất'
        ))
        
        fig_ef.update_layout(
            title="Efficient Frontier - Tối ưu hóa danh mục OTC",
            xaxis_title="Rủi ro (Volatility hàng năm)",
            yaxis_title="Lợi nhuận kỳ vọng hàng năm",
            template="plotly_dark",
            height=500,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                x=1.02,
                y=0.95,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(0,0,0,0.5)'
            ),
            margin=dict(r=60)
        )
        st.plotly_chart(fig_ef, use_container_width=True)
    
    # ================= THÊM KẾT LUẬN CHUYÊN VIÊN CHO BƯỚC 3 =================
    st.markdown("#### 📝 NHẬN ĐỊNH CHUYÊN VIÊN - PHÂN TÍCH MPT")
    
    if optimal_sharpe >= 1.5:
        st.success(f"""
        ** DANH MỤC HIỆU QUẢ CAO**
        - Sharpe Ratio: {optimal_sharpe:.2f} | LN: {optimal_return*100:.1f}%/năm | RR: {optimal_risk*100:.1f}%/năm
        """)
    elif optimal_sharpe >= 0.8:
        st.info(f"""
        ** DANH MỤC HIỆU QUẢ TRUNG BÌNH**
        - Sharpe Ratio: {optimal_sharpe:.2f} | LN: {optimal_return*100:.1f}%/năm | RR: {optimal_risk*100:.1f}%/năm
        """)
    else:
        st.warning(f"""
        **⚠️ DANH MỤC HIỆU QUẢ THẤP**
        - Sharpe Ratio: {optimal_sharpe:.2f} | LN: {optimal_return*100:.1f}%/năm | RR: {optimal_risk*100:.1f}%/năm
        """)
    
    # Hiển thị bảng chi tiết rủi ro - lợi nhuận từng cổ phiếu
    st.markdown("####  CHI TIẾT RỦI RO - LỢI NHUẬN TỪNG CỔ PHIẾU")
    risk_return_df = selected_stocks[['Symbol', 'ExpectedReturn', 'Risk', 'Score']].copy()
    risk_return_df['ExpectedReturn'] = risk_return_df['ExpectedReturn'].apply(lambda x: f"{x*100:.1f}%")
    risk_return_df['Risk'] = risk_return_df['Risk'].apply(lambda x: f"{x*100:.1f}%")
    risk_return_df.columns = ['Mã CK', 'LN kỳ vọng', 'Rủi ro', 'Điểm']
    st.dataframe(risk_return_df, use_container_width=True, hide_index=True)
    
    # ================= 6. PHÂN BỔ TỶ TRỌNG THEO KHẨU VỊ RỦI RO =================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; margin: 15px 0;'>
        <h3 style='color: #00c6ff; margin: 0;'> PHÂN BỔ THEO KHẨU VỊ RỦI RO</h3>
        <p style='color: #9ca3af; margin: 5px 0 0 0;'>Lựa chọn chiến lược phù hợp với khách hàng</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"Khẩu vị rủi ro hiện tại của khách hàng: **{risk_profile}**")
    
    with st.expander(" Giải thích các chiến lược phân bổ theo khẩu vị rủi ro"):
        st.markdown(f"""
        **Phân bổ danh mục dựa trên khẩu vị rủi ro của khách hàng: {risk_profile}
        
        -  Tối ưu Sharpe: Cân bằng lợi nhuận và rủi ro
        -  Rủi ro thấp nhất: Ưu tiên bảo toàn vốn
        -  Theo điểm số: Phân bổ theo điểm chấm
        """)
    
    if risk_profile == "Thấp":
        default_strategy = " Rủi ro thấp nhất (An toàn)"
    else:
        default_strategy = " Tối ưu Sharpe (Cân bằng)"
    
    allocation_strategy = st.radio(
        "Chiến lược phân bổ:",
        [" Tối ưu Sharpe (Cân bằng)", " Rủi ro thấp nhất (An toàn)", " Đều theo điểm số (Đơn giản)"],
        horizontal=True
    )
    
    if allocation_strategy == " Tối ưu Sharpe (Cân bằng)":
        final_weights = optimal_weights
        strategy_name = "Tối ưu Sharpe"
    elif allocation_strategy == " Rủi ro thấp nhất (An toàn)":
        final_weights = min_risk_weights
        strategy_name = "Rủi ro thấp nhất"
    else:
        score_weights = selected_stocks['Score'].values
        final_weights = score_weights / score_weights.sum()
        strategy_name = "Theo điểm số"
    
    final_weights = final_weights * max_otc_allocation / 100
    selected_stocks['Weight'] = final_weights
    selected_stocks['Amount'] = capital * selected_stocks['Weight']
    
    st.markdown(f"#### Chiến lược: {strategy_name}")
    
    # Biểu đồ tròn OTC
    pie_data = [(row['Symbol'], row['Weight']*100, row['Amount']) for _, row in selected_stocks.iterrows() if row['Weight'] > 0.005]
    if pie_data:
        fig_pie = go.Figure(data=[go.Pie(labels=[f"{s} ({w:.1f}%)" for s,w,_ in pie_data], values=[w for _,w,_ in pie_data], hole=0.4)])
        fig_pie.update_layout(title=f"Phân bổ OTC - {max_otc_allocation}% tổng vốn", template="plotly_dark", height=450)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Bảng chi tiết
    table_data = []
    for _, row in selected_stocks.iterrows():
        table_data.append({
            "Mã CK": row['Symbol'],
            "Điểm": f"{row['Score']:.0f}",
            "ROE": f"{row['ROE']:.1f}%",
            "LN KV": f"{row['ExpectedReturn']*100:.1f}%",
            "Risk": f"{row['Risk']*100:.1f}%",
            "Tỷ trọng": f"{row['Weight']*100:.1f}%",
            "Số tiền": f"{row['Amount']:,.0f} VND"
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    
    # ================= 7. TỔNG HỢP DANH MỤC =================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; margin: 15px 0;'>
        <h3 style='color: #00c6ff; margin: 0;'> DANH MỤC TỔNG THỂ</h3>
        <p style='color: #9ca3af; margin: 5px 0 0 0;'>Phân bổ theo khẩu vị {risk_profile}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if risk_profile == "Thấp":
        cash_weight = 70
        bonds_weight = 30 - max_otc_allocation
    elif risk_profile == "Trung bình":
        cash_weight = 50
        bonds_weight = 50 - max_otc_allocation
    else:
        cash_weight = 30
        bonds_weight = 70 - max_otc_allocation
    
    bonds_weight = max(0, bonds_weight)
    if bonds_weight < 0:
        cash_weight = cash_weight + bonds_weight
        bonds_weight = 0
    
    capital = st.session_state['capital']
    cash_amount = capital * cash_weight / 100
    bonds_amount = capital * bonds_weight / 100
    otc_amount = capital * max_otc_allocation / 100
    
    fig_total = go.Figure(data=[go.Pie(
        labels=[
            f"💰 Tiền mặt ({cash_weight:.0f}%)",
            f"📊 CK niêm yết ({bonds_weight:.0f}%)",
            f"⚡ OTC ({max_otc_allocation:.0f}%)"
        ],
        values=[cash_weight, bonds_weight, max_otc_allocation],
        hole=0.4,
        marker=dict(colors=['#00c6ff', '#00ff88', '#ffa500']),
        textinfo='percent'
    )])
    fig_total.update_layout(
        title=f"Danh mục tổng thể - Khẩu vị {risk_profile.upper()}",
        template="plotly_dark",
        height=450,
        annotations=[dict(
            text=f"Tổng vốn<br>{capital:,.0f} VND",
            x=0.5, y=0.5, font_size=14, showarrow=False,
            font=dict(color='white', size=14, weight='bold')
        )]
    )
    st.plotly_chart(fig_total, use_container_width=True)
    
    # Kết luận tổng hợp
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px; margin: 15px 0;'>
        <h4 style='color: #00c6ff; margin: 0 0 15px 0;'> KẾT LUẬN</h4>
        <table style='width: 100%;'>
            <tr style='background: #0f0f23;'><td> Tiền mặt</th><td style='text-align: right;'>{cash_weight:.1f}%</td><td style='text-align: right; color: #00ff88;'>{cash_amount:,.0f} VND</td></tr>
            <tr style='background: #1a1a2e;'><td> CK niêm yết</th><td style='text-align: right;'>{bonds_weight:.1f}%</td><td style='text-align: right; color: #00ff88;'>{bonds_amount:,.0f} VND</td></tr>
            <tr style='background: #0f0f23;'><td> OTC</th><td style='text-align: right;'>{max_otc_allocation:.1f}%</td><td style='text-align: right; color: #ffa500;'>{otc_amount:,.0f} VND</td></tr>
            <tr style='border-top: 2px solid #00c6ff; background: #1a1a2e; font-weight: bold;'><td> TỔNG CỘNG</th><td style='text-align: right;'><b>100%</b></td><td style='text-align: right; color: #00c6ff;'><b>{capital:,.0f} VND</b></td></tr>
        </table>
        <p style='margin-top: 15px; color: #9ca3af;'> LN OTC: <b>{optimal_return*100:.1f}%/năm</b> | Rủi ro: <b>{optimal_risk*100:.1f}%/năm</b> | Sharpe: <b>{optimal_sharpe:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # ================= 8. XUẤT BÁO CÁO =================
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 15px; margin: 15px 0;'>
        <h3 style='color: #00c6ff; margin: 0;'> XUẤT KẾT QUẢ</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        report_lines = [
            "=" * 60,
            "HVA ROBO-ADVISOR - BÁO CÁO DANH MỤC OTC",
            "=" * 60,
            f"Ngày: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            f"Khách hàng: {client_name} | Tuổi: {age} | Khẩu vị: {risk_profile}",
            f"Vốn: {capital:,.0f} VND",
            "-" * 60,
            "DANH MỤC OTC ĐỀ XUẤT",
        ]
        for _, row in selected_stocks.iterrows():
            report_lines.append(f"{row['Symbol']} | Điểm: {row['Score']:.0f} | LN KV: {row['ExpectedReturn']*100:.1f}% | Tỷ trọng: {row['Weight']*100:.1f}% | {row['Amount']:,.0f} VND")
        report_lines.extend([
            "-" * 60,
            f"TỔNG OTC: {max_otc_allocation:.1f}% ({otc_amount:,.0f} VND)",
            f"Lợi nhuận kỳ vọng: {optimal_return*100:.2f}%",
            f"Rủi ro: {optimal_risk*100:.2f}%",
            f"Sharpe Ratio: {optimal_sharpe:.2f}",
            "=" * 60
        ])
        st.download_button(" Tải báo cáo TXT", "\n".join(report_lines), f"OTC_Report_{datetime.now().strftime('%Y%m%d')}.txt", use_container_width=True)
    
    with col_export2:
        export_df = selected_stocks[['Symbol', 'CompanyName', 'Score', 'ROE', 'ExpectedReturn', 'Risk', 'Weight', 'Amount']].copy()
        export_df['Weight'] = export_df['Weight'] * 100
        export_df['ExpectedReturn'] = export_df['ExpectedReturn'] * 100
        export_df['Risk'] = export_df['Risk'] * 100
        export_df.columns = ['Mã CK', 'Tên công ty', 'Điểm', 'ROE (%)', 'LN kỳ vọng (%)', 'Rủi ro (%)', 'Tỷ trọng (%)', 'Số tiền (VND)']
        st.download_button(" Tải danh mục CSV", export_df.to_csv(index=False, encoding='utf-8-sig'), f"OTC_Portfolio_{datetime.now().strftime('%Y%m%d')}.csv", use_container_width=True)
    
    # ================= 9. CẢNH BÁO =================
    st.markdown("---")
    st.warning("""
    ⚠️ **CẢNH BÁO RỦI RO OTC**
    - Dữ liệu tài chính KHÔNG được kiểm toán độc lập
    - Thanh khoản thấp, khó bán khi cần
    - Chỉ dành cho nhà đầu tư chấp nhận rủi ro cao
    - **Không nên phân bổ quá 20% tổng vốn vào OTC**
    """)
    
    st.caption(f" **TỔNG KẾT** |  Đã phân tích: {len(all_scores)} CP |  Chọn top {top_n} |  OTC: {otc_amount:,.0f} VND ({max_otc_allocation}%) |  Sharpe: {optimal_sharpe:.2f}")

# ================= BẢN QUYỀN =================
st.markdown("""
    <hr style='margin: 30px 0 10px 0; border-color: #333;'>
    <div style='text-align: center; padding: 15px 0;'>
        <p style='color: #6c7293; font-size: 13px; margin: 0;'>
            © 2026 <strong style='color: #00c6ff;'> PHAN THỊ KIM CÚC </strong> - HVA Robo Advisor. 
            Mọi bản quyền được bảo lưu.
        </p>
        <p style='color: #4a4f6e; font-size: 11px; margin: 5px 0 0 0;'>
            📊 Dữ liệu từ KBS, OTC tổng hợp | ⚠️ Công cụ hỗ trợ phân tích và tư vấn đầu tư
        </p>
    </div>
    """, unsafe_allow_html=True)
