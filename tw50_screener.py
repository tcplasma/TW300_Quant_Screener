"""
╔══════════════════════════════════════════════════════════════════════╗
║           台股前50大權值股 — 量化因子選股系統                          ║
║           TW50 Multi-Layer Quantitative Stock Screener               ║
║                                                                      ║
║  作者：自動化量化分析引擎                                               ║
║  策略：價值 × 品質 × 動能 三層篩選                                     ║
╚══════════════════════════════════════════════════════════════════════╝

【投資哲學】
  好公司 (Quality) × 好價格 (Value) × 好時機 (Momentum) = 超額報酬

【指標金融意義速查】
  P/E (本益比)    → 你付出多少錢買 1 元盈餘；越低越便宜
  P/B (本淨比)    → 你付出多少錢買 1 元帳面資產；< 1 代表市值低於清算價值
  殖利率           → 每年現金回報率；越高代表持有成本越低
  ROE             → 公司用股東的錢賺錢的效率；> 15% 為優質
  營業利益率        → 本業真實獲利能力；剔除業外雜訊
  MA20 穿越       → 短期趨勢轉折訊號（月線）
  量能激增比        → > 1.5 倍代表有資金積極介入
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
import time
import sys
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# 【模組 1】台灣 50 成分股清單
# 每季度更新一次，或參考：https://www.etf.com.tw/etf/0050
# ═══════════════════════════════════════════════════════════════════════

# 台股代號在 yfinance 需加上 .TW 後綴
TW50_COMPONENTS = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2454.TW": "聯發科",
    "2308.TW": "台達電",
    "2881.TW": "富邦金",
    "2882.TW": "國泰金",
    "2412.TW": "中華電",
    "2303.TW": "聯電",
    "3711.TW": "日月光投控",
    "2002.TW": "中鋼",
    "1301.TW": "台塑",
    "1303.TW": "南亞",
    "2886.TW": "兆豐金",
    "2891.TW": "中信金",
    "1326.TW": "台化",
    "2884.TW": "玉山金",
    "2892.TW": "第一金",
    "5880.TW": "合庫金",
    "2883.TW": "開發金",
    "2885.TW": "元大金",
    "2890.TW": "永豐金",
    "2887.TW": "台新金",
    "6505.TW": "台塑化",
    "3008.TW": "大立光",
    "2207.TW": "和泰車",
    "2880.TW": "華南金",
    "1216.TW": "統一",
    "2382.TW": "廣達",
    "4938.TW": "和碩",
    "2395.TW": "研華",
    "2379.TW": "瑞昱",
    "2345.TW": "智邦",
    "2357.TW": "華碩",
    "2327.TW": "國巨",
    "3045.TW": "台灣大",
    "4904.TW": "遠傳",
    "2912.TW": "統一超",
    "1101.TW": "台泥",
    "2354.TW": "鴻準",
    "2377.TW": "微星",
    "2408.TW": "南亞科",
    "2301.TW": "光寶科",
    "6669.TW": "緯穎",
    "3034.TW": "聯詠",
    "2887.TW": "台新金",
    "2603.TW": "長榮",
    "2609.TW": "陽明",
    "2615.TW": "萬海",
    "1590.TW": "亞德客",
    "8046.TW": "南電",
}

# ═══════════════════════════════════════════════════════════════════════
# 【模組 2】數據抓取引擎（含錯誤處理）
# ═══════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker: str, name: str, period_years: int = 5) -> dict | None:
    """
    抓取單一股票的完整數據，並計算所有量化因子。
    
    Args:
        ticker: Yahoo Finance 代號（如 "2330.TW"）
        name: 股票中文名稱
        period_years: 歷史數據年數（用於計算殖利率均值）
    
    Returns:
        包含所有因子的字典，若抓取失敗則回傳 None
    """
    try:
        print(f"  ⟳ 抓取 {ticker} ({name})...", end="", flush=True)
        
        # 下載 K 線數據（近 period_years 年）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * period_years)
        
        hist = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True
        )
        
        # 安全檢查：若無歷史數據（停牌、下市）則跳過
        if hist is None or len(hist) < 60:
            print(" ✗ 數據不足，跳過")
            return None
        
        # 取得即時基本面數據
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # ── 價格技術指標計算 ──────────────────────────────────────────
        close = hist['Close'].squeeze()  # 確保為 Series
        volume = hist['Volume'].squeeze()
        
        current_price = float(close.iloc[-1])
        
        # 移動平均線（MA）
        ma5   = float(close.rolling(5).mean().iloc[-1])
        ma20  = float(close.rolling(20).mean().iloc[-1])
        ma60  = float(close.rolling(60).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
        
        # 前一日的 MA20（用於判斷是否「剛穿越」）
        prev_price = float(close.iloc[-2])
        prev_ma20  = float(close.rolling(20).mean().iloc[-2])
        
        # 量能激增比：今日成交量 / 5日平均量
        # 金融意義：> 1.5 倍代表有異常資金介入，是趨勢轉折的重要確認訊號
        vol_today   = float(volume.iloc[-1])
        vol_5d_avg  = float(volume.rolling(5).mean().iloc[-1])
        volume_ratio = vol_today / vol_5d_avg if vol_5d_avg > 0 else 1.0
        
        # 判斷「黃金交叉穿越」：昨日在 MA20 之下，今日突破到 MA20 之上
        # 這是短期趨勢由空轉多的關鍵技術訊號
        crossed_ma20 = (prev_price < prev_ma20) and (current_price > ma20)
        
        # ── 價值面因子 ────────────────────────────────────────────────
        pe_ratio = info.get('trailingPE', np.nan)      # 本益比（過去12個月）
        pb_ratio = info.get('priceToBook', np.nan)     # 本淨比（市值/帳面價值）
        
        # 近五年平均殖利率（用歷史股利資料）
        dividends = stock.dividends
        if len(dividends) > 0:
            # 計算最近 5 年的年度殖利率
            recent_divs = dividends[dividends.index >= start_date.tz_localize(dividends.index.tz)]
            annual_div = recent_divs.resample('YE').sum()
            # 殖利率 = 年度股利 / 期末股價
            hist_year_end = close.resample('YE').last()
            common_idx = annual_div.index.intersection(hist_year_end.index)
            if len(common_idx) > 0:
                yearly_yields = (annual_div[common_idx] / hist_year_end[common_idx]) * 100
                avg_yield = float(yearly_yields.mean())
            else:
                avg_yield = info.get('dividendYield', np.nan)
                avg_yield = avg_yield * 100 if avg_yield and not np.isnan(avg_yield) else np.nan
        else:
            avg_yield = info.get('dividendYield', np.nan)
            avg_yield = avg_yield * 100 if avg_yield and not np.isnan(avg_yield) else np.nan
        
        # ── 品質面因子 ────────────────────────────────────────────────
        # ROE (股東權益報酬率) = 淨利 / 股東權益
        # 金融意義：衡量公司用股東資金賺錢的效率，巴菲特最重視此指標
        roe = info.get('returnOnEquity', np.nan)
        roe = roe * 100 if roe and not np.isnan(roe) else np.nan
        
        # 營業利益率 = 營業利益 / 營收
        # 金融意義：反映本業真實競爭力，排除業外收益的干擾
        op_margin = info.get('operatingMargins', np.nan)
        op_margin = op_margin * 100 if op_margin and not np.isnan(op_margin) else np.nan
        
        # ── 安全邊際計算 ──────────────────────────────────────────────
        # 「超跌黃金」定義：股價低於年線 10% 以上，但 ROE 仍 > 15%
        # 代表公司基本面健康，只是被市場情緒過度懲罰
        below_ma200_pct = np.nan
        if not np.isnan(ma200):
            below_ma200_pct = ((current_price - ma200) / ma200) * 100  # 負值代表低於年線
        
        print(" ✓")
        
        return {
            "代號":        ticker.replace(".TW", ""),
            "名稱":        name,
            "現價":        round(current_price, 1),
            # ── 價值層 ──
            "本益比(PE)":  round(pe_ratio, 1) if not np.isnan(pe_ratio) else np.nan,
            "本淨比(PB)":  round(pb_ratio, 2) if not np.isnan(pb_ratio) else np.nan,
            "平均殖利率%": round(avg_yield, 2) if avg_yield and not np.isnan(avg_yield) else np.nan,
            # ── 品質層 ──
            "ROE%":        round(roe, 1) if not np.isnan(roe) else np.nan,
            "營業利益率%": round(op_margin, 1) if not np.isnan(op_margin) else np.nan,
            # ── 動能層 ──
            "MA5":         round(ma5, 1),
            "MA20":        round(ma20, 1),
            "MA60":        round(ma60, 1),
            "MA200":       round(ma200, 1) if not np.isnan(ma200) else np.nan,
            "量能倍數":    round(volume_ratio, 2),
            "穿越MA20":    crossed_ma20,
            # ── 安全邊際 ──
            "偏離年線%":   round(below_ma200_pct, 1) if not np.isnan(below_ma200_pct) else np.nan,
        }
    
    except Exception as e:
        print(f" ✗ 錯誤：{str(e)[:50]}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# 【模組 3】多維度評分引擎
# ═══════════════════════════════════════════════════════════════════════

def calculate_composite_score(row: pd.Series) -> float:
    """
    計算「綜合得分」，整合三個層次的訊號強度。
    
    評分邏輯（總分 100 分）：
      價值層  (40分)：P/E 和 P/B 越低分數越高
      品質層  (35分)：ROE 和營業利益率越高分數越高
      動能層  (25分)：穿越均線 + 量能激增加分
    """
    score = 0.0
    
    # ── 價值層評分（40 分）──────────────────────────────────────────
    pe = row.get("本益比(PE)", np.nan)
    pb = row.get("本淨比(PB)", np.nan)
    
    if not np.isnan(pe) and pe > 0:
        # P/E < 10 → 20 分；10-15 → 15 分；15-20 → 8 分；> 20 → 0 分
        if pe < 10:       score += 20
        elif pe < 15:     score += 15
        elif pe < 20:     score += 8
        
    if not np.isnan(pb) and pb > 0:
        # P/B < 1 → 20 分；1-1.5 → 15 分；1.5-2 → 8 分；> 2 → 0 分
        if pb < 1.0:      score += 20
        elif pb < 1.5:    score += 15
        elif pb < 2.0:    score += 8
    
    # ── 品質層評分（35 分）──────────────────────────────────────────
    roe = row.get("ROE%", np.nan)
    op_margin = row.get("營業利益率%", np.nan)
    
    if not np.isnan(roe):
        # ROE > 20% → 20 分；15-20% → 15 分；10-15% → 8 分；< 10% → 0 分
        if roe > 20:      score += 20
        elif roe > 15:    score += 15
        elif roe > 10:    score += 8
    
    if not np.isnan(op_margin):
        # 營業利益率 > 20% → 15 分；10-20% → 10 分；5-10% → 5 分
        if op_margin > 20:   score += 15
        elif op_margin > 10: score += 10
        elif op_margin > 5:  score += 5
    
    # ── 動能層評分（25 分）──────────────────────────────────────────
    vol_ratio  = row.get("量能倍數", 1.0)
    crossed    = row.get("穿越MA20", False)
    below_ma200 = row.get("偏離年線%", np.nan)
    
    # 穿越月線 + 量能確認：最強烈的趨勢轉多訊號
    if crossed:           score += 15
    if vol_ratio > 2.0:   score += 10
    elif vol_ratio > 1.5: score += 7
    elif vol_ratio > 1.2: score += 4
    
    return round(score, 1)


def classify_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    根據三大策略為每檔股票貼上信號標籤。
    """
    df = df.copy()
    labels = []
    
    for _, row in df.iterrows():
        tags = []
        pe   = row.get("本益比(PE)", np.nan)
        pb   = row.get("本淨比(PB)", np.nan)
        roe  = row.get("ROE%", np.nan)
        vol  = row.get("量能倍數", 1.0)
        crossed  = row.get("穿越MA20", False)
        below200 = row.get("偏離年線%", np.nan)
        
        # 策略一：被低估的寶藏（Value Trap 排除：要求 ROE > 8%）
        if (not np.isnan(pe) and pe < 15 and pe > 0 and
            not np.isnan(pb) and pb < 1.5 and
            not np.isnan(roe) and roe > 8):
            tags.append("💎 被低估")
        
        # 策略二：趨勢轉強（動能轉折）
        if crossed and vol > 1.2:
            tags.append("🚀 趨勢轉多")
        
        # 策略三：超跌黃金（安全邊際）
        if (not np.isnan(below200) and below200 < -10 and
            not np.isnan(roe) and roe > 15):
            tags.append("⭐ 超跌黃金")
        
        labels.append(" | ".join(tags) if tags else "—")
    
    df["信號標籤"] = labels
    return df


# ═══════════════════════════════════════════════════════════════════════
# 【模組 4】視覺化引擎
# ═══════════════════════════════════════════════════════════════════════

def plot_risk_reward_matrix(df: pd.DataFrame):
    """
    繪製「風險/回報矩陣圖」
    X 軸：P/B（越低越安全，代表資產保護越強）
    Y 軸：ROE（越高越優質，代表獲利能力越強）
    氣泡大小：量能倍數（越大代表資金關注度越高）
    顏色：信號標籤類型
    """
    # 過濾掉缺值
    plot_df = df.dropna(subset=["本淨比(PB)", "ROE%"]).copy()
    
    if len(plot_df) == 0:
        print("⚠️  無足夠數據繪圖")
        return
    
    # ── 圖表設定 ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor='#0D1117')
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor('#161B22')
    
    # 背景象限分割線（視覺引導）
    ax.axvline(x=1.5, color='#30363D', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.axhline(y=15,  color='#30363D', linewidth=1.5, linestyle='--', alpha=0.8)
    
    # 象限標籤
    ax.text(0.3, 30, "🏆 最佳區間\n低估值 × 高獲利",
            fontsize=9, color='#3FB950', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1117', alpha=0.5))
    ax.text(3.5, 30, "✨ 高品質成長\n高估值 × 高獲利",
            fontsize=9, color='#58A6FF', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1117', alpha=0.5))
    ax.text(0.3, 2, "⚠️ 價值陷阱區\n低估值 × 低獲利",
            fontsize=9, color='#F85149', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1117', alpha=0.5))
    ax.text(3.5, 2, "🚫 迴避區\n高估值 × 低獲利",
            fontsize=9, color='#8B949E', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1117', alpha=0.5))
    
    # 顏色對應邏輯
    def get_color_and_marker(row):
        label = row.get("信號標籤", "—")
        if "被低估" in str(label) and "趨勢轉多" in str(label):
            return '#FFD700', '*', 200  # 金色星形 = 雙重信號最強
        elif "超跌黃金" in str(label):
            return '#FF8C00', 'D', 120  # 橙色菱形
        elif "被低估" in str(label):
            return '#3FB950', 'o', 100  # 綠色
        elif "趨勢轉多" in str(label):
            return '#58A6FF', '^', 100  # 藍色三角
        else:
            return '#8B949E', 'o', 60   # 灰色 = 一般
    
    # 繪製各點
    for _, row in plot_df.iterrows():
        pb  = row["本淨比(PB)"]
        roe = row["ROE%"]
        vol = min(row.get("量能倍數", 1.0), 4.0)  # 限制最大泡泡大小
        color, marker, base_size = get_color_and_marker(row)
        size = base_size * vol
        
        # 主體散點
        ax.scatter(pb, roe, c=color, marker=marker, s=size,
                   alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)
        
        # 股票名稱標籤
        ax.annotate(
            f"{row['代號']}\n{row['名稱']}",
            (pb, roe),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=7,
            color='#C9D1D9',
            fontfamily='sans-serif'
        )
    
    # ── 圖例 ────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color='#FFD700', label='💎🚀 低估值 + 趨勢轉多（最強訊號）'),
        mpatches.Patch(color='#FF8C00', label='⭐ 超跌黃金（安全邊際）'),
        mpatches.Patch(color='#3FB950', label='💎 低估值（價值面）'),
        mpatches.Patch(color='#58A6FF', label='🚀 趨勢轉多（動能面）'),
        mpatches.Patch(color='#8B949E', label='— 觀察中'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              facecolor='#21262D', edgecolor='#30363D',
              labelcolor='#C9D1D9', fontsize=9)
    
    # ── 軸線與標題 ──────────────────────────────────────────────────
    ax.set_xlabel("P/B 本淨比（越低 = 估值越安全）", color='#8B949E', fontsize=11)
    ax.set_ylabel("ROE 股東權益報酬率 %（越高 = 品質越佳）", color='#8B949E', fontsize=11)
    ax.set_title(
        f"台股前50大權值股  風險/回報矩陣\n"
        f"氣泡大小 = 量能激增倍數 | 分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        color='#F0F6FC', fontsize=14, pad=15
    )
    
    ax.tick_params(colors='#8B949E')
    ax.spines['bottom'].set_color('#30363D')
    ax.spines['left'].set_color('#30363D')
    ax.spines['top'].set_color('#30363D')
    ax.spines['right'].set_color('#30363D')
    
    # X 軸範圍稍微拉寬以容納標籤
    x_max = min(plot_df["本淨比(PB)"].max() * 1.3, 8)
    y_max = min(plot_df["ROE%"].max() * 1.2, 60)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    
    output_path = "tw50_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0D1117', edgecolor='none')
    print(f"\n📊 矩陣圖已儲存：{output_path}")
    plt.show()


def print_summary_table(df: pd.DataFrame):
    """
    美化列印摘要表格，突出顯示各類信號。
    """
    display_cols = [
        "代號", "名稱", "現價", "本益比(PE)", "本淨比(PB)",
        "平均殖利率%", "ROE%", "營業利益率%",
        "量能倍數", "穿越MA20", "偏離年線%",
        "綜合得分", "信號標籤"
    ]
    
    # 僅顯示有信號或得分前20的
    top_df = df.sort_values("綜合得分", ascending=False).head(20)
    
    print("\n" + "═" * 100)
    print("  📋  台股前50大權值股 量化因子選股報告")
    print(f"  🕐  分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 100)
    
    # 用 pandas 美化顯示（調整欄位）
    display_df = top_df[[c for c in display_cols if c in top_df.columns]].copy()
    display_df["穿越MA20"] = display_df["穿越MA20"].map({True: "✅ 是", False: "—"})
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.1f}'.format)
    
    print(display_df.to_string(index=False))
    print("═" * 100)


def print_interpretation_guide(df: pd.DataFrame):
    """
    列印「探勘結論總結」與解讀指南。
    """
    print("\n" + "╔" + "═"*68 + "╗")
    print("║  🧭  探勘結論解讀指南                                              ║")
    print("╠" + "═"*68 + "╣")
    
    # 找出各類別標的
    undervalued   = df[df["信號標籤"].str.contains("被低估", na=False)]
    momentum      = df[df["信號標籤"].str.contains("趨勢轉多", na=False)]
    golden        = df[df["信號標籤"].str.contains("超跌黃金", na=False)]
    dual_signal   = df[df["信號標籤"].str.contains("被低估", na=False) &
                       df["信號標籤"].str.contains("趨勢轉多", na=False)]
    
    print(f"║  💎 被低估寶藏（{len(undervalued):2d} 檔）：PE < 15 且 PB < 1.5 且 ROE > 8%         ║")
    if len(undervalued) > 0:
        names = "、".join(undervalued["名稱"].head(5).tolist())
        print(f"║     → {names[:55]:<55} ║")
    
    print(f"║  🚀 趨勢轉強（{len(momentum):2d} 檔）：剛穿越月線 且 量能 > 1.2 倍              ║")
    if len(momentum) > 0:
        names = "、".join(momentum["名稱"].head(5).tolist())
        print(f"║     → {names[:55]:<55} ║")
    
    print(f"║  ⭐ 超跌黃金（{len(golden):2d} 檔）：低於年線 >10% 但 ROE > 15%              ║")
    if len(golden) > 0:
        names = "、".join(golden["名稱"].head(5).tolist())
        print(f"║     → {names[:55]:<55} ║")
    
    print("╠" + "═"*68 + "╣")
    print("║  🔥 「必須立刻關注」的數據組合定義：                               ║")
    print("║                                                                    ║")
    print("║  ① 雙重確認（最強）：出現「💎被低估 + 🚀趨勢轉多」同時觸發           ║")
    print("║     → 估值便宜 + 資金開始介入，勝率最高                             ║")
    print("║                                                                    ║")
    print("║  ② 超跌反轉：ROE > 15%（體質好）但股價低於年線 > 20%                ║")
    print("║     → 市場過度懲罰，等待反彈的時機                                  ║")
    print("║                                                                    ║")
    print("║  ③ 量能暴增穿線：量能倍數 > 2.0 且剛穿越 MA20                       ║")
    print("║     → 有大資金進場的技術確認訊號                                    ║")
    print("║                                                                    ║")
    print("║  ⚠️  務必排除：PE < 10 但 ROE < 5% → 這是「價值陷阱」（衰退股）     ║")
    print("╠" + "═"*68 + "╣")
    
    if len(dual_signal) > 0:
        print(f"║  ★ 今日「雙重確認」標的：{len(dual_signal)} 檔                                 ║")
        top = dual_signal.sort_values("綜合得分", ascending=False).head(3)
        for _, row in top.iterrows():
            line = f"     {row['代號']} {row['名稱']}  得分：{row['綜合得分']:.0f}"
            print(f"║  {line:<66} ║")
    else:
        print("║  今日無「雙重確認」標的（可放寬條件或等待時機）                   ║")
    
    print("╚" + "═"*68 + "╝")


# ═══════════════════════════════════════════════════════════════════════
# 【主程式】
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  🔍  台股前50大權值股量化因子選股系統")
    print("  📅  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("═"*60)
    print(f"\n⚙️  開始抓取 {len(TW50_COMPONENTS)} 檔成分股數據...\n")
    
    results = []
    failed  = []
    
    for i, (ticker, name) in enumerate(TW50_COMPONENTS.items(), 1):
        print(f"[{i:2d}/{len(TW50_COMPONENTS)}]", end=" ")
        data = fetch_stock_data(ticker, name)
        
        if data:
            results.append(data)
        else:
            failed.append(f"{ticker}({name})")
        
        # 避免觸發 API 速率限制：每5檔暫停1秒
        if i % 5 == 0:
            time.sleep(1)
    
    if not results:
        print("\n❌ 無法取得任何數據，請檢查網路連線或 API 狀態")
        sys.exit(1)
    
    # ── 建立 DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame(results)
    
    # 計算綜合得分
    df["綜合得分"] = df.apply(calculate_composite_score, axis=1)
    
    # 貼上信號標籤
    df = classify_signals(df)
    
    # 按綜合得分排序
    df = df.sort_values("綜合得分", ascending=False).reset_index(drop=True)
    
    # ── 輸出報表 ────────────────────────────────────────────────────
    print_summary_table(df)
    print_interpretation_guide(df)
    
    # ── 儲存 CSV ────────────────────────────────────────────────────
    csv_path = f"tw50_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig 確保 Excel 中文正常
    print(f"\n💾 完整報告已儲存：{csv_path}")
    
    # ── 繪製矩陣圖 ──────────────────────────────────────────────────
    print("\n🎨 正在繪製風險/回報矩陣圖...")
    plot_risk_reward_matrix(df)
    
    # ── 失敗清單 ────────────────────────────────────────────────────
    if failed:
        print(f"\n⚠️  以下 {len(failed)} 檔因數據不足或停牌而跳過：")
        print("   " + "、".join(failed))
    
    print("\n✅ 分析完成！\n")
    return df


if __name__ == "__main__":
    df = main()
