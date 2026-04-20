###############################################################################
# セフトリアキソン PBPK シミュレーター (Streamlit版)
# ===================================================
# 卒業研究
# 「セフトリアキソンの治療効果と偽胆石リスクを同時評価する
#   PBPKシミュレーターの構築と仮想患者解析」
# に対応する実装。
#
# 目的:
#   - %fT>MIC と飽和指数（SI）を同一モデル上で相対比較する
#   - 患者背景・投与設計・食事条件の違いを教育的に可視化する
#
# 位置づけ:
#   - 教育用・探索用の簡略PBPKモデル
#   - 臨床予測用ツールではない
#
# 参考文献:
#   Alasmari et al. (2023) Front Pharmacol 14:1200828
#   Ewoldt TMJ et al. (2023) J Antimicrob Chemother 78:1059-1065
#   Schleibinger et al. (2015) Br J Clin Pharmacol 80:525-533
#   Shiffman et al. (1990) Gastroenterology 99:1772-1778
###############################################################################

import numpy as np
import pandas as pd

# NumPy 2.0 互換: np.trapz → np.trapezoid
try:
    _trapz = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# 1. モデル固定パラメータ
#    卒論本文の方法・表1に対応する主要パラメータ
# ─────────────────────────────────────────────────────────────────────────────
FIXED = {
    'MW': 554.58, 'BP': 0.55,
    'Bmax_mM': 0.55, 'B50_mM': 0.030, 'theta_ALB': 1.0, 'ALB_ref': 4.0,
    'CO_ref': 390,
    'FQlu': 1.0, 'FQli': 0.25, 'FQki': 0.19, 'FQre': 0.56,
    'FVart': 0.026, 'FVven': 0.051, 'FVlu': 0.008, 'FVli': 0.021,
    'FVki': 0.004, 'FVre': 0.60, 'FVgb': 0.0007,
    'Kplu': 0.22, 'Kpli': 0.22, 'Kpki': 0.22, 'Kpre': 0.22,
    'CLbiliary': 0.22, 'fu_ref': 0.05, 'GFR_filt_fraction': 1.0,
    'Ksp': 1.62e-6, 'SI_threshold': 10.4,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. ODE システム（ノートブック セル4（=Cell 5）と同一）
# ─────────────────────────────────────────────────────────────────────────────
def _pbpk_rhs(t, y, p, dose_schedule, meal_hours):
    IVEN, ART, VEN, LUNG, LIVER, KIDNEY, REST, GB, URINE, BILE_CUM = y

    MW = p['MW']; BP = p['BP']
    BW = p['BW']; ALB = p['ALB']; ALB_ref = p['ALB_ref']
    Bmax_mgL = p['Bmax_mM'] * MW
    B50_mgL = p['B50_mM'] * MW
    Bmax_adj = Bmax_mgL * (ALB / ALB_ref) ** p['theta_ALB']

    Vart = p['FVart'] * BW; Vven = p['FVven'] * BW
    Vlu = p['FVlu'] * BW; Vli = p['FVli'] * BW
    Vki = p['FVki'] * BW; Vre = p['FVre'] * BW
    CO = p['CO_ref'] * (BW / 70.0) ** 0.75
    Qlu = p['FQlu'] * CO; Qli = p['FQli'] * CO
    Qki = p['FQki'] * CO; Qre = p['FQre'] * CO

    Cart = ART / Vart; Cven = VEN / Vven
    Clu = LUNG / Vlu; Cli = LIVER / Vli
    Cki = KIDNEY / Vki; Cre = REST / Vre
    Cp_ven = Cven / BP

    # 非線形蛋白結合
    # CTRXの可変・飽和性蛋白結合を簡略化した1サイトモデル
    Ct_plasma = Cp_ven
    b_coef = B50_mgL + Bmax_adj - Ct_plasma
    c_coef = -Ct_plasma * B50_mgL
    discriminant = b_coef**2 - 4.0 * c_coef
    Cu_plasma = 0.0
    if discriminant > 0.0 and Ct_plasma > 0.0:
        Cu_plasma = (-b_coef + np.sqrt(discriminant)) / 2.0
        Cu_plasma = max(0.0, min(Cu_plasma, Ct_plasma))
    fu = Cu_plasma / Ct_plasma if Ct_plasma > 1e-10 else 1.0
    fu = max(0.01, min(fu, 1.0))

    CLrenal_u = p['GFR'] * 0.06 * p['GFR_filt_fraction']
    Renal_elim = CLrenal_u * Cu_plasma

    Biliary_elim = max(0.0, p['CLbiliary'] * Cp_ven)

    infusion = 0.0
    for t_dose, dose_mg, tinf_h in dose_schedule:
        if t_dose <= t < t_dose + tinf_h:
            infusion = dose_mg / tinf_h
            break

    dIVEN = 0.0
    dART = Qlu * (Clu / p['Kplu'] * BP - Cart)
    dVEN = (Qli * Cli / p['Kpli'] * BP + Qki * Cki / p['Kpki'] * BP
            + Qre * Cre / p['Kpre'] * BP - Qlu * Cven + infusion)
    dLUNG = Qlu * (Cven - Clu / p['Kplu'] * BP)
    dLIVER = Qli * (Cart - Cli / p['Kpli'] * BP) - Biliary_elim
    dKIDNEY = Qki * (Cart - Cki / p['Kpki'] * BP) - Renal_elim
    dREST = Qre * (Cart - Cre / p['Kpre'] * BP)
    gb_rate = _gb_empty_rate(t, meal_hours)
    frac = _gb_frac(t, meal_hours)
    dGB = Biliary_elim * frac - gb_rate * GB
    dURINE = Renal_elim
    dBILE_CUM = Biliary_elim

    return [dIVEN, dART, dVEN, dLUNG, dLIVER, dKIDNEY, dREST, dGB, dURINE, dBILE_CUM]


# ─────────────────────────────────────────────────────────────────────────────
# 3. シミュレーション・解析関数（ノートブック セル6 と同一）
# ─────────────────────────────────────────────────────────────────────────────
# 食事時刻のデフォルトパターン（時刻は 0–24h、毎日繰り返し）
MEAL_PATTERNS = {
    0: [],                    # 絶食
    1: [12.0],                # 1回（昼）
    2: [8.0, 18.0],           # 2回（朝・夕）
    3: [8.0, 12.0, 18.0],     # 3回（朝・昼・夕）
}

# 投与時刻パターン（時刻は 0–24h）
DOSE_HOUR = {
    24: [9.0],          # q24h: 9時
    12: [9.0, 21.0],    # q12h: 9時, 21時
}

REGIMEN_PATTERNS = [
    ('0.5 gを24時間ごと', 500, 24, '#17becf'),
    ('0.5 gを12時間ごと', 500, 12, '#8c564b'),
    ('1 gを24時間ごと', 1000, 24, '#2ca02c'),
    ('1 gを12時間ごと', 1000, 12, '#1f77b4'),
    ('2 gを24時間ごと', 2000, 24, '#ff7f0e'),
    ('2 gを12時間ごと', 2000, 12, '#d62728'),
]

# 胆嚢パラメータ
# 教育用に単純化した胆嚢流入・排出モデル
GB_K_BASE = 0.05   # 基礎排出速度 (/h) — 食間のMMCによる微量排出
GB_K_MEAL = 1.0    # 食事刺激排出速度 (/h) — CCK刺激（食後60-70%排出を再現）
GB_MEAL_DUR = 2.0  # 食事刺激の持続時間 (h) — 生理学的に1-2h
GB_FRAC_FASTING = 0.5   # 空腹時の胆嚢到達率（Oddi括約筋閉鎖、大部分が胆嚢へ）
GB_FRAC_MEAL = 0.0      # 食後の胆嚢到達率（Oddi括約筋開放、十二指腸へ直接流入）


def _gb_empty_rate(t, meal_hours):
    """時刻 t における胆嚢排出速度定数を返す。
    食事後 GB_MEAL_DUR 時間は CCK 刺激により排出が亢進する。"""
    t_day = t % 24.0
    for mh in meal_hours:
        if mh <= t_day < mh + GB_MEAL_DUR:
            return GB_K_BASE + GB_K_MEAL
    return GB_K_BASE


def _gb_frac(t, meal_hours):
    """時刻 t における胆嚢到達率を返す。
    空腹時: Oddi括約筋閉鎖 → 胆汁は胆嚢へ流入。
    食後: CCK → Oddi括約筋開放 → 胆汁は十二指腸へ直接流入。"""
    t_day = t % 24.0
    for mh in meal_hours:
        if mh <= t_day < mh + GB_MEAL_DUR:
            return GB_FRAC_MEAL
    return GB_FRAC_FASTING


def run_simulation(patient, dosing, sim_duration_h=None):
    dose_mg = dosing['dose_mg']
    tinf_h = dosing['tinf_h']
    ii_h = dosing['ii_h']
    n_doses = dosing['n_doses']

    pat = dict(patient)
    meals_per_day = pat.pop('meals_per_day', 3)
    meal_hours = tuple(MEAL_PATTERNS.get(meals_per_day, MEAL_PATTERNS[3]))

    # 投与スケジュール: 時刻ベース (q24h→9時, q12h→9時/21時)
    dose_times_per_day = DOSE_HOUR.get(ii_h, [9.0])
    dose_schedule = []
    doses_added = 0
    day = 0
    while doses_added < n_doses:
        for dh in dose_times_per_day:
            if doses_added >= n_doses:
                break
            dose_schedule.append((day * 24.0 + dh, dose_mg, tinf_h))
            doses_added += 1
        day += 1

    if sim_duration_h is None:
        sim_duration_h = dose_schedule[-1][0] + 24

    p = {**FIXED, **pat}
    y0 = [0.0] * 10
    fast = dosing.get('_fast', False)
    dt = 0.5 if fast else 0.1
    t_eval = np.arange(0, sim_duration_h + dt * 0.5, dt)
    t_eval = t_eval[t_eval <= sim_duration_h]

    sol = solve_ivp(
        fun=lambda t, y: _pbpk_rhs(t, y, p, dose_schedule, meal_hours),
        t_span=(0.0, sim_duration_h), y0=y0, t_eval=t_eval,
        method='LSODA', atol=1e-8 if fast else 1e-10,
        rtol=1e-6 if fast else 1e-8, max_step=0.5 if fast else 0.1,
    )

    _, ART, VEN, LUNG, LIVER, KIDNEY, REST, GB, URINE, BILE_CUM = sol.y

    BW = p['BW']; BP = p['BP']; MW = p['MW']
    FVven = p['FVven']; FVgb = p['FVgb']
    Bmax_mgL = p['Bmax_mM'] * MW
    B50_mgL = p['B50_mM'] * MW
    Bmax_adj = Bmax_mgL * (p['ALB'] / p['ALB_ref']) ** p['theta_ALB']

    Cp_total = (VEN / (FVven * BW)) / BP
    fu_arr = np.ones_like(Cp_total)
    for i, ct in enumerate(Cp_total):
        b = B50_mgL + Bmax_adj - ct
        c = -ct * B50_mgL
        disc = b**2 - 4.0 * c
        cu = 0.0
        if disc > 0.0 and ct > 0.0:
            cu = (-b + np.sqrt(disc)) / 2.0
            cu = max(0.0, min(cu, ct))
        fu_arr[i] = max(0.01, min(cu / ct if ct > 1e-10 else 1.0, 1.0))

    Cp_free = Cp_total * fu_arr
    C_bile = GB / (FVgb * BW)
    SI = (p['Ca_bile'] * 0.001) * (C_bile / MW * 0.001) / FIXED['Ksp']

    return pd.DataFrame({
        'time': sol.t, 'Cp_total': Cp_total, 'Cp_free': Cp_free,
        'C_bile': C_bile, 'SI': SI, 'FU': fu_arr,
        'cum_renal': URINE, 'cum_biliary': BILE_CUM,
        'LIVER': LIVER, 'KIDNEY': KIDNEY, 'LUNG': LUNG,
    })


def _last_dose_time(ii_h, n_doses):
    """最終投与の開始時刻を返す（投与スケジュールと同じロジック）。"""
    dose_times_per_day = DOSE_HOUR.get(ii_h, [9.0])
    doses_added = 0
    day = 0
    last_t = 0.0
    while doses_added < n_doses:
        for dh in dose_times_per_day:
            if doses_added >= n_doses:
                break
            last_t = day * 24.0 + dh
            doses_added += 1
        day += 1
    return last_t


def calc_fTMIC(df, mic, ii_h, n_doses):
    t_last = _last_dose_time(ii_h, n_doses)
    ss = df[(df['time'] >= t_last) & (df['time'] <= t_last + ii_h)]
    if len(ss) < 2:
        return np.nan
    return np.sum(ss['Cp_free'] > mic) / len(ss) * 100.0


def calc_max_SI(df, ii_h, n_doses):
    t_prev = _last_dose_time(ii_h, n_doses - 1) if n_doses > 1 else 0.0
    ss = df[df['time'] >= t_prev]
    return float(ss['SI'].max())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="セフトリアキソン PBPK シミュレーター",
    page_icon="💊",
    layout="wide",
)

st.title("セフトリアキソン PBPK シミュレーター")
st.caption("教育用・探索用：治療効果指標と偽胆石リスク指標の相対比較")

# ── Sidebar ──
with st.sidebar:
    st.header("患者パラメータ")
    bw = st.slider("体重 (kg)", 35, 120, 40, 1)
    alb = st.slider("血清アルブミン値 (g/dL)", 1.5, 4.5, 2.8, 0.1)
    gfr = st.slider("GFR (mL/min)", 15, 120, 45, 1)
    meals_option = st.selectbox(
        "食事回数（/日）",
        ["絶食", "1回（昼）", "2回（朝・夕）", "3回（朝・昼・夕）"],
        index=0,
    )
    meals_map = {"絶食": 0, "1回（昼）": 1, "2回（朝・夕）": 2, "3回（朝・昼・夕）": 3}
    meals_per_day = meals_map[meals_option]
    if meals_per_day > 0:
        meal_times_str = "、".join([f"{int(h)}時" for h in MEAL_PATTERNS[meals_per_day]])
        st.caption(f"食事時刻: {meal_times_str}（食後{GB_MEAL_DUR:.1f}h胆嚢収縮）")
    ca_bile = 5.0   # 胆嚢胆汁中 Ca²⁺ (mmol/L) 固定
    # 胆嚢到達率は時刻依存（_gb_frac関数で制御: 空腹時0.5, 食後0.0）
    
    st.markdown("---")
    st.header("投与設計")
    dose_mg = st.selectbox("1回投与量 (mg)", [500, 1000, 1500, 2000], index=1)
    tinf_min = 30  # 添付文書準拠: 点滴静注30分
    ii_h = st.selectbox("投与間隔 (時間)", [12, 24], index=1)
    ndoses = int(7 * 24 / ii_h)  # 治療期間7日固定
    dose_times_str = "、".join([f"{int(h)}時" for h in DOSE_HOUR[ii_h]])
    st.caption(f"投与時刻: {dose_times_str}（7日間、点滴時間30分）")

    st.markdown("---")
    st.header("感染症パラメータ")
    mic_presets = {
        "Enterobacterales (S ≤ 1)": 1.0,
        "Enterobacterales (I = 2)": 2.0,
        "S. pneumoniae 髄膜炎 (S ≤ 0.5)": 0.5,
        "S. pneumoniae 非髄膜炎 (S ≤ 1)": 1.0,
        "H. influenzae (S ≤ 2)": 2.0,
        "N. meningitidis (S ≤ 0.12)": 0.12,
        "N. gonorrhoeae (S ≤ 0.25)": 0.25,
        "カスタム": -1,
    }
    mic_choice = st.selectbox(
        "MIC プリセット (CLSI M100 準拠)", list(mic_presets.keys()), index=3)
    mic_val = mic_presets[mic_choice]
    if mic_val < 0:
        mic_val = st.number_input("カスタム MIC (mg/L)", 0.0625, 4.0, 1.0, 0.125)
    ftmic_target = st.slider("目標 %fT>MIC (%)", 40, 100, 60, 5)

# Patient / dosing dicts
patient = {
    'BW': bw, 'ALB': alb, 'GFR': gfr,
    'Ca_bile': ca_bile,
    'meals_per_day': meals_per_day,
}
dosing = {
    'dose_mg': dose_mg, 'tinf_h': tinf_min / 60.0,
    'ii_h': ii_h, 'n_doses': ndoses,
}


# ── Cached simulation ──
@st.cache_data(show_spinner="シミュレーション実行中...")
def cached_sim(patient_tuple, dosing_tuple):
    p = dict(patient_tuple)
    d = dict(dosing_tuple)
    return run_simulation(p, d)


def to_tuple(d):
    return tuple(sorted(d.items()))


# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 血漿中濃度推移",
    "🎯 %fT>MIC 解析",
    "⚠️ 偽胆石リスク",
    "🔬 感度分析",
    "🗺️ ヒートマップ",
    "ℹ️ モデル情報",
])

# ==========================================================================
# Tab 1: PK Profile（ノートブック セル9-10相当）
# ==========================================================================
with tab1:
    df = cached_sim(to_tuple(patient), to_tuple(dosing))

    # PK Summary
    t_last = _last_dose_time(ii_h, ndoses)
    ss = df[(df['time'] >= t_last) & (df['time'] <= t_last + ii_h)]
    ftmic = calc_fTMIC(df, mic_val, ii_h, ndoses)
    max_si = calc_max_SI(df, ii_h, ndoses)
    dose_total = dose_mg * ndoses
    pct_renal = df['cum_renal'].iloc[-1] / dose_total * 100
    pct_biliary = df['cum_biliary'].iloc[-1] / dose_total * 100

    # AUC (定常状態1投与間隔, 台形法)
    auc_total = _trapz(ss['Cp_total'], ss['time']) if len(ss) >= 2 else 0.0
    auc_free = _trapz(ss['Cp_free'], ss['time']) if len(ss) >= 2 else 0.0

    # SI 関連指標 (定常状態)
    ss_si = df[df['time'] >= t_last]
    si_above_time = (ss_si['SI'] > FIXED['SI_threshold']).sum() / len(ss_si) * 100 if len(ss_si) > 0 else 0.0
    auc_si = _trapz(ss_si['SI'], ss_si['time']) if len(ss_si) >= 2 else 0.0

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    if len(ss) >= 2:
        col_s1.metric("Cmax total", f"{ss['Cp_total'].max():.1f} mg/L")
        col_s2.metric("Cmin total", f"{ss['Cp_total'].min():.1f} mg/L")
        goal_icon = "✅" if ftmic >= ftmic_target else "❌"
        col_s3.metric("%fT>MIC", f"{ftmic:.1f}%", f"{goal_icon} 目標 {ftmic_target}%")
        si_icon = "⚠️" if max_si > FIXED['SI_threshold'] else "✅"
        col_s4.metric("最大 SI", f"{max_si:.2f}", f"{si_icon} 閾値 {FIXED['SI_threshold']}")

    # Plots
    col_pk, col_exc = st.columns(2)

    with col_pk:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['Cp_total'],
            name='総濃度', line=dict(color='#2c3e50', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['Cp_free'],
            name='遊離型濃度', line=dict(color='#e74c3c', width=2),
        ))
        fig.update_layout(
            title=f"{dose_mg}mg {ii_h}時間毎 (ALB={alb} g/dL, GFR={gfr} mL/min)",
            xaxis_title="時間 (h)", yaxis_title="血漿中濃度 (mg/L)",
            yaxis=dict(rangemode='tozero'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=420, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_exc:
        fig_exc = go.Figure()
        fig_exc.add_trace(go.Scatter(
            x=df['time'], y=df['cum_renal'] / dose_total * 100,
            name=f'腎排泄 ({pct_renal:.0f}%dose)', line=dict(color='#3498db', width=2),
        ))
        fig_exc.add_trace(go.Scatter(
            x=df['time'], y=df['cum_biliary'] / dose_total * 100,
            name=f'胆汁排泄 ({pct_biliary:.0f}%dose)', line=dict(color='#2ecc71', width=2),
        ))
        fig_exc.update_layout(
            title="排泄経路",
            xaxis_title="時間 (h)", yaxis_title="累積排泄率 (%dose)",
            yaxis=dict(rangemode='tozero'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=420, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_exc, use_container_width=True)

    # Additional info
    with st.expander("PK 要約テーブル（定常状態）"):
        if len(ss) >= 2:
            summary = pd.DataFrame({
                'パラメータ': [
                    'Cmax total (mg/L)', 'Cmin total (mg/L)',
                    'Cmax free (mg/L)', 'Cmin free (mg/L)',
                    'AUCss total (mg·h/L)', 'AUCss free (mg·h/L)',
                    'fu at Cmax (%)', 'fu at Cmin (%)',
                    '腎排泄 (%dose)', '胆汁排泄 (%dose)',
                ],
                '値': [
                    f"{ss['Cp_total'].max():.1f}",
                    f"{ss['Cp_total'].min():.1f}",
                    f"{ss['Cp_free'].max():.2f}",
                    f"{ss['Cp_free'].min():.3f}",
                    f"{auc_total:.1f}",
                    f"{auc_free:.2f}",
                    f"{ss.loc[ss['Cp_total'].idxmax(), 'FU'] * 100:.1f}",
                    f"{ss.loc[ss['Cp_total'].idxmin(), 'FU'] * 100:.1f}",
                    f"{pct_renal:.1f}",
                    f"{pct_biliary:.1f}",
                ],
            }).set_index('パラメータ')
            st.dataframe(summary, use_container_width=True)

# ==========================================================================
# Tab 2: %fT>MIC Analysis（ノートブック セル11相当）
# ==========================================================================
with tab2:
    col_ftmic1, col_ftmic2 = st.columns(2)

    with col_ftmic1:
        # 定常状態の遊離型濃度推移
        t_prev2 = _last_dose_time(ii_h, max(ndoses - 2, 1)) if ndoses > 2 else 0.0
        t_end_plot = t_last
        t_start_plot = t_prev2
        ss_plot = df[(df['time'] >= t_start_plot) & (df['time'] <= t_end_plot)]

        fig_ft = go.Figure()
        fig_ft.add_trace(go.Scatter(
            x=ss_plot['time'], y=ss_plot['Cp_free'],
            name='遊離型 Cp', line=dict(color='#e74c3c', width=2),
            fill='tozeroy', fillcolor='rgba(231,76,60,0.05)',
        ))
        # MIC超過域を緑で塗る（MICライン〜Cp曲線の間のみ）
        cp_clipped = np.maximum(ss_plot['Cp_free'].values, mic_val)
        fig_ft.add_trace(go.Scatter(
            x=ss_plot['time'], y=[mic_val] * len(ss_plot),
            name='_mic_base', line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ))
        fig_ft.add_trace(go.Scatter(
            x=ss_plot['time'], y=cp_clipped,
            name='MIC超過域', fill='tonexty',
            fillcolor='rgba(39,174,96,0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True,
        ))
        fig_ft.add_hline(y=mic_val, line_dash='dash', line_color='#2c3e50',
                         annotation_text=f'MIC = {mic_val} mg/L',
                         annotation_position='top left')
        fig_ft.update_layout(
            title=f"定常状態 fT>MIC (%fT>MIC = {ftmic:.1f}%)",
            xaxis_title="時間 (h)", yaxis_title="遊離型血漿中濃度 (mg/L)",
            yaxis=dict(rangemode='tozero'),
            height=450, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_ft, use_container_width=True)

    with col_ftmic2:
        # 投与量・投与間隔比較（6パターン）
        fig_dc = go.Figure()
        for lbl, d, ii_cmp, col_c in REGIMEN_PATTERNS:
            nd_cmp = max(3, int(7 * 24 / ii_cmp))
            dos_cmp = {**dosing, 'dose_mg': d, 'ii_h': ii_cmp,
                       'n_doses': nd_cmp}
            df_d = cached_sim(to_tuple(patient), to_tuple(dos_cmp))
            ft_d = calc_fTMIC(df_d, mic_val, ii_cmp, nd_cmp)
            t_cmp_last = _last_dose_time(ii_cmp, nd_cmp)
            t_cmp_prev2 = _last_dose_time(ii_cmp, max(nd_cmp - 2, 1)) if nd_cmp > 2 else 0.0
            ss_d = df_d[(df_d['time'] >= t_cmp_prev2) & (df_d['time'] <= t_cmp_last)]
            fig_dc.add_trace(go.Scatter(
                x=ss_d['time'], y=ss_d['Cp_free'],
                name=f'{lbl}（%fT>MIC {ft_d:.0f}%）',
                line=dict(color=col_c, width=2),
            ))
        fig_dc.add_hline(y=mic_val, line_dash='dash', line_color='black')
        fig_dc.update_layout(
            title=f"投与量比較 (ALB={alb}, GFR={gfr}, MIC={mic_val} mg/L)",
            xaxis_title="時間 (h)", yaxis_title="遊離型血漿中濃度 (mg/L)",
            yaxis=dict(rangemode='tozero'),
            height=450, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_dc, use_container_width=True)

    # Result box
    achieved = ftmic >= ftmic_target
    icon = "✅" if achieved else "⚠️"
    bg = '#eaf4fd' if achieved else '#fde8e8'
    border = 'green' if achieved else 'red'
    st.markdown(
        f"""<div style="background:{bg}; border-left:4px solid {border};
        padding:12px; border-radius:4px;">
        <strong>{icon} {'目標達成' if achieved else '目標未達成'}</strong><br>
        %fT&gt;MIC = <strong>{ftmic:.1f}%</strong>（目標: {ftmic_target}%）<br>
        MIC = {mic_val:.3f} mg/L | {dose_mg} mg q{ii_h}h |
        ALB = {alb:.1f} g/dL | GFR = {gfr} mL/min
        </div>""",
        unsafe_allow_html=True,
    )

# ==========================================================================
# Tab 3: Pseudolithiasis（ノートブック セル12相当）
# ==========================================================================
with tab3:
    col_si1, col_si2 = st.columns(2)

    with col_si1:
        si_thresh = FIXED['SI_threshold']
        fig_si = go.Figure()
        fig_si.add_trace(go.Scatter(
            x=df['time'], y=df['SI'],
            name='飽和指数 (SI)', line=dict(color='#27ae60', width=2),
        ))
        # 準安定限界超過域を赤で塗る
        si_clipped = np.maximum(df['SI'].values, si_thresh)
        fig_si.add_trace(go.Scatter(
            x=df['time'], y=[si_thresh] * len(df),
            name='_si_base', line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ))
        fig_si.add_trace(go.Scatter(
            x=df['time'], y=si_clipped,
            name='準安定限界超過域', fill='tonexty',
            fillcolor='rgba(231,76,60,0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True,
        ))
        fig_si.add_hline(y=si_thresh, line_dash='dash',
                         line_color='#e74c3c',
                         annotation_text=f"準安定限界 (SI={si_thresh})")
        fig_si.add_hline(y=1.0, line_dash='dot', line_color='#f39c12',
                         annotation_text='Ksp (SI=1.0)')
        fig_si.update_layout(
            title=f"Ca-CTRX 飽和指数 (Ca²⁺={ca_bile} mmol/L) — 最大 SI = {max_si:.2f}",
            xaxis_title="時間 (h)", yaxis_title="飽和指数 (SI)",
            height=450, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_si, use_container_width=True)

    with col_si2:
        fig_cb = go.Figure()
        fig_cb.add_trace(go.Scatter(
            x=df['time'], y=df['C_bile'],
            name='胆嚢内 CTRX', line=dict(color='#8e44ad', width=2),
        ))
        fig_cb.update_layout(
            title="胆嚢内セフトリアキソン濃度推移",
            xaxis_title="時間 (h)", yaxis_title="胆嚢内 CTRX 濃度 (mg/L)",
            height=450, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_cb, use_container_width=True)

    # Risk assessment
    if max_si < 1.0:
        risk = "極低（Ksp未満）"; bg = "#eaf4fd"
    elif max_si < FIXED['SI_threshold']:
        risk = "低（準安定域内）"; bg = "#eaf4fd"
    elif si_above_time < 30:
        risk = "中（一過性に超過）"; bg = "#fef5e7"
    else:
        risk = "高（持続的超過）"; bg = "#fde8e8"

    st.markdown(
        f"""<div style="background:{bg}; border-left:4px solid #f39c12;
        padding:12px; border-radius:4px;">
        <strong>偽胆石リスク: {risk}</strong><br>
        最大飽和指数 = <strong>{max_si:.2f}</strong>（準安定限界: {FIXED['SI_threshold']}）<br>
        SI-AUC（定常状態） = <strong>{auc_si:.1f}</strong> · h<br>
        胆汁中 CTRX 最高濃度 = <strong>{df['C_bile'].max():.0f} mg/L</strong><br>
        準安定限界超過時間: <strong>{si_above_time:.0f}%</strong><br>
        {dose_mg} mg q{ii_h}h | Ca²⁺ = {ca_bile:.1f} mmol/L | GFR = {gfr} | 食事 {meals_per_day}回/日
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.info(
        "**SI の解釈に関する注意:**\n\n"
        "- CLbiliary、Ksp、SIの定義式はいずれも文献に基づいています。\n"
        "- 一方、胆嚢到達率や排出パラメータは仮定値であり、**SIの絶対値は定量的に信頼できません**。\n"
        "- **条件間の相対比較**（投与量・ALB・GFR・食事条件の違い）は同一モデル上の比較であり、**妥当性があります**。"
    )

# ==========================================================================
# Tab 4: Sensitivity Analysis（ノートブック セル14-15相当）
# ==========================================================================
with tab4:
    sa_type = st.radio(
        "感度分析パラメータ",
        ["体重の影響", "血清アルブミン値の影響", "GFR の影響", "食事回数の影響"],
        horizontal=True,
    )

    if sa_type == "体重の影響":
        values = [35, 45, 55, 70, 90]
        colors_5 = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
        col_sa1, col_sa2 = st.columns(2)

        fig_bw1 = go.Figure()
        fig_bw2 = go.Figure()
        for bw_v, col_c in zip(values, colors_5):
            df_bw = cached_sim(to_tuple({**patient, 'BW': bw_v}), to_tuple(dosing))
            ft = calc_fTMIC(df_bw, mic_val, ii_h, ndoses)
            si_bw = calc_max_SI(df_bw, ii_h, ndoses)
            fig_bw1.add_trace(go.Scatter(
                x=df_bw['time'], y=df_bw['Cp_free'],
                name=f'{bw_v}kg (fT={ft:.0f}%)',
                line=dict(color=col_c, width=1.5),
            ))
            fig_bw2.add_trace(go.Scatter(
                x=df_bw['time'], y=df_bw['SI'],
                name=f'{bw_v}kg (SI={si_bw:.1f})',
                line=dict(color=col_c, width=1.5),
            ))

        fig_bw1.add_hline(y=mic_val, line_dash='dash', line_color='black')
        fig_bw1.update_layout(
            title=f"体重の影響 — %fT>MIC (MIC={mic_val} mg/L)",
            xaxis_title="時間 (h)", yaxis_title="遊離型血漿中濃度 (mg/L)",
            yaxis=dict(rangemode='tozero'), height=450,
        )
        fig_bw2.add_hline(y=FIXED['SI_threshold'], line_dash='dash', line_color='#e74c3c',
                         annotation_text=f"準安定限界 ({FIXED['SI_threshold']})")
        fig_bw2.update_layout(
            title="体重の影響 — 偽胆石リスク",
            xaxis_title="時間 (h)", yaxis_title="飽和指数 (SI)",
            height=450,
        )

        with col_sa1:
            st.plotly_chart(fig_bw1, use_container_width=True)
        with col_sa2:
            st.plotly_chart(fig_bw2, use_container_width=True)

        st.info(
            "セフトリアキソンは固定用量で投与されるため、"
            "体重が軽い患者ではピーク濃度は高くなりやすい。\n\n"
            "一方、本モデルでは体重低下に伴う分布容積および濃度時間推移の変化により、"
            "fT>MICはむしろ低下する傾向がみられる。\n\n"
            "- **fT>MIC**: 体重が大きいほど半減期が長く、MIC超過時間を維持しやすい\n"
            "- **偽胆石リスク**: 体重が軽いほどピーク濃度上昇に伴い胆嚢内濃度も上昇する"
        )

    elif sa_type == "血清アルブミン値の影響":
        values = [1.5, 2.0, 2.5, 3.0, 4.0]
        colors_5 = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
        col_sa1, col_sa2 = st.columns(2)

        fig_a1 = go.Figure()
        fig_a2 = go.Figure()
        for a, col_c in zip(values, colors_5):
            df_a = cached_sim(to_tuple({**patient, 'ALB': a}), to_tuple(dosing))
            ft = calc_fTMIC(df_a, mic_val, ii_h, ndoses)
            si_a = calc_max_SI(df_a, ii_h, ndoses)
            fig_a1.add_trace(go.Scatter(
                x=df_a['time'], y=df_a['Cp_free'],
                name=f'ALB={a} g/dL (fT={ft:.0f}%)',
                line=dict(color=col_c, width=1.5),
            ))
            fig_a2.add_trace(go.Scatter(
                x=df_a['time'], y=df_a['SI'],
                name=f'ALB={a} (SI={si_a:.1f})',
                line=dict(color=col_c, width=1.5),
            ))

        fig_a1.add_hline(y=mic_val, line_dash='dash', line_color='black')
        fig_a1.update_layout(
            title="血清アルブミン値の影響 — %fT>MIC",
            xaxis_title="時間 (h)", yaxis_title="遊離型血漿中濃度 (mg/L)",
            yaxis=dict(rangemode='tozero'), height=450,
        )
        fig_a2.add_hline(y=FIXED['SI_threshold'], line_dash='dash', line_color='#e74c3c',
                         annotation_text=f"準安定限界 ({FIXED['SI_threshold']})")
        fig_a2.update_layout(
            title="血清アルブミン値の影響 — 偽胆石リスク",
            xaxis_title="時間 (h)", yaxis_title="飽和指数 (SI)",
            height=450,
        )

        with col_sa1:
            st.plotly_chart(fig_a1, use_container_width=True)
        with col_sa2:
            st.plotly_chart(fig_a2, use_container_width=True)

        st.info(
            "低アルブミン血症では1サイト飽和結合モデルにより"
            "遊離型分率（fu）が上昇し、遊離型濃度および腎排泄が変動します。\n\n"
            "- **%fT>MIC**: fu上昇により遊離型濃度は変動するが、遊離型クリアランスも増加するため、影響は単純ではない\n"
            "- **SI**: 本モデルでは胆汁排泄を総濃度ベースの固定CLで実装しているため、ALBの影響は生理学的期待と逆転しうる"
        )

    elif sa_type == "GFR の影響":
        values = [15, 30, 60, 90, 120]
        colors_5 = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
        col_sa1, col_sa2 = st.columns(2)

        fig_g1 = go.Figure()
        fig_g2 = go.Figure()
        for g, col_c in zip(values, colors_5):
            df_g = cached_sim(to_tuple({**patient, 'GFR': g}), to_tuple(dosing))
            ft = calc_fTMIC(df_g, mic_val, ii_h, ndoses)
            si_g = calc_max_SI(df_g, ii_h, ndoses)
            fig_g1.add_trace(go.Scatter(
                x=df_g['time'], y=df_g['Cp_free'],
                name=f'GFR={g} (fT={ft:.0f}%)',
                line=dict(color=col_c, width=1.5),
            ))
            fig_g2.add_trace(go.Scatter(
                x=df_g['time'], y=df_g['SI'],
                name=f'GFR={g} (SI={si_g:.1f})',
                line=dict(color=col_c, width=1.5),
            ))

        fig_g1.add_hline(y=mic_val, line_dash='dash', line_color='black')
        fig_g1.update_layout(
            title=f"GFR の影響 — %fT>MIC (MIC={mic_val} mg/L)",
            xaxis_title="時間 (h)", yaxis_title="遊離型血漿中濃度 (mg/L)",
            yaxis=dict(rangemode='tozero'), height=450,
        )
        fig_g2.add_hline(y=FIXED['SI_threshold'], line_dash='dash', line_color='#e74c3c',
                         annotation_text=f"準安定限界 ({FIXED['SI_threshold']})")
        fig_g2.update_layout(
            title="GFR の影響 — 偽胆石リスク",
            xaxis_title="時間 (h)", yaxis_title="飽和指数 (SI)",
            height=450,
        )

        with col_sa1:
            st.plotly_chart(fig_g1, use_container_width=True)
        with col_sa2:
            st.plotly_chart(fig_g2, use_container_width=True)

        st.info(
            "GFR低下により腎クリアランス（GFR × fu）が減少し、"
            "血漿中濃度が上昇します。\n\n"
            "- **fT>MIC**: 濃度上昇により達成率が向上。セフトリアキソンは胆汁排泄も"
            "約40%を担うため、軽度〜中等度のCKDでは一般に減量不要とされる\n"
            "- **偽胆石リスク**: 腎排泄減少の代償として胆汁排泄の相対的寄与が増加し、"
            "胆嚢内濃度が上昇する可能性がある"
        )

    else:  # 食事回数の影響
        meal_values = [0, 1, 2, 3]
        meal_labels = ['絶食', '1回/日', '2回/日', '3回/日']
        colors_4 = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
        col_sa1, col_sa2 = st.columns(2)

        fig_m1 = go.Figure()
        fig_m2 = go.Figure()
        for m, lbl, col_c in zip(meal_values, meal_labels, colors_4):
            df_m = cached_sim(
                to_tuple({**patient, 'meals_per_day': m}), to_tuple(dosing))
            si_m = calc_max_SI(df_m, ii_h, ndoses)
            fig_m1.add_trace(go.Scatter(
                x=df_m['time'], y=df_m['C_bile'],
                name=f'{lbl}', line=dict(color=col_c, width=1.5),
            ))
            fig_m2.add_trace(go.Scatter(
                x=df_m['time'], y=df_m['SI'],
                name=f'{lbl} (SI={si_m:.1f})',
                line=dict(color=col_c, width=1.5),
            ))

        fig_m1.update_layout(
            title="食事回数の影響 — 胆嚢内CTRX濃度",
            xaxis_title="時間 (h)", yaxis_title="胆嚢内 CTRX 濃度 (mg/L)",
            yaxis=dict(rangemode='tozero'), height=450,
        )
        fig_m2.add_hline(y=FIXED['SI_threshold'], line_dash='dash', line_color='#e74c3c',
                         annotation_text=f"準安定限界 ({FIXED['SI_threshold']})")
        fig_m2.update_layout(
            title="食事回数の影響 — 偽胆石リスク",
            xaxis_title="時間 (h)", yaxis_title="飽和指数 (SI)",
            height=450,
        )

        with col_sa1:
            st.plotly_chart(fig_m1, use_container_width=True)
        with col_sa2:
            st.plotly_chart(fig_m2, use_container_width=True)

        st.info(
            "**食事回数は%fT>MICには影響しません**（血漿中濃度は変わらない）。\n\n"
            "食事はSI（偽胆石リスク）に2つの経路で影響します。\n\n"
            "1. **胆嚢排出の促進**: CCK刺激による胆嚢収縮で胆汁が排出され、胆嚢内CTRX蓄積が減少します。\n"
            "2. **胆嚢流入の抑制**: 食後はOddi括約筋が開放され、肝胆汁が十二指腸へ直接流入するため、胆嚢への薬物流入が減少します。\n\n"
            "絶食患者では両方の防御機構が失われ、偽胆石リスクが大幅に上昇します。"
        )

# ==========================================================================
# Tab 5: Heatmaps（ノートブック セル16-18相当）
# ==========================================================================
with tab5:
    hm_type = st.radio(
        "ヒートマップ種類",
        ["%fT>MIC (ALB × GFR)", "最大 SI (ALB × GFR)",
         "%fT>MIC (GFR × 投与パターン)", "最大 SI (GFR × 投与パターン)"],
        horizontal=True,
    )

    if hm_type == "%fT>MIC (ALB × GFR)":
        with st.spinner("ヒートマップ計算中..."):
            alb_range = np.arange(1.5, 4.6, 0.5)
            gfr_range = np.arange(15, 121, 15)
            grid = np.zeros((len(alb_range), len(gfr_range)))

            dosing_fast = {**dosing, '_fast': True}
            for i, a in enumerate(alb_range):
                for j, g in enumerate(gfr_range):
                    df_hm = cached_sim(
                        to_tuple({**patient, 'ALB': float(a), 'GFR': float(g)}),
                        to_tuple(dosing_fast))
                    grid[i, j] = calc_fTMIC(df_hm, mic_val, ii_h, ndoses)

            fig_hm = go.Figure(data=go.Heatmap(
                z=grid,
                x=[str(int(g)) for g in gfr_range],
                y=[f'{a:.1f}' for a in alb_range],
                colorscale=[[0, '#e74c3c'], [0.6, '#f1c40f'], [1, '#27ae60']],
                zmin=0, zmax=100,
                text=np.round(grid, 0).astype(int).astype(str),
                texttemplate='%{text}%', textfont=dict(size=12),
                colorbar=dict(title='%fT>MIC'),
            ))
            fig_hm.update_layout(
                title=f"%fT>MIC ヒートマップ ({dose_mg}mg {ii_h}時間毎, MIC={mic_val} mg/L, 目標: {ftmic_target}%以上)",
                xaxis_title="GFR (mL/min)", yaxis_title="血清アルブミン値 (g/dL)",
                xaxis=dict(type='category'), height=500,
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    elif hm_type == "最大 SI (ALB × GFR)":
        with st.spinner("ヒートマップ計算中..."):
            alb_range2 = np.arange(1.5, 4.6, 0.5)
            gfr_range3 = np.arange(15, 121, 15)
            grid_si2 = np.zeros((len(alb_range2), len(gfr_range3)))

            dosing_fast2 = {**dosing, '_fast': True}
            for i, a in enumerate(alb_range2):
                for j, g in enumerate(gfr_range3):
                    df_hm2 = cached_sim(
                        to_tuple({**patient, 'ALB': float(a), 'GFR': float(g)}),
                        to_tuple(dosing_fast2))
                    grid_si2[i, j] = calc_max_SI(df_hm2, ii_h, ndoses)

            fig_si2 = go.Figure(data=go.Heatmap(
                z=grid_si2,
                x=[str(int(g)) for g in gfr_range3],
                y=[f'{a:.1f}' for a in alb_range2],
                colorscale=[[0, '#27ae60'], [0.5, '#f1c40f'], [1, '#e74c3c']],
                text=np.round(grid_si2, 1).astype(str),
                texttemplate='%{text}', textfont=dict(size=12),
                colorbar=dict(title='最大SI'),
            ))
            fig_si2.update_layout(
                title=f"最大飽和指数ヒートマップ (ALB × GFR)\n{dose_mg}mg {ii_h}時間毎, 食事{meals_per_day}回/日",
                xaxis_title="GFR (mL/min)", yaxis_title="血清アルブミン値 (g/dL)",
                xaxis=dict(type='category'), height=500,
            )
            st.plotly_chart(fig_si2, use_container_width=True)

    elif hm_type == "%fT>MIC (GFR × 投与パターン)":
        with st.spinner("ヒートマップ計算中..."):
            gfr_range_ft = np.arange(15, 121, 15)
            dose_range_ft = [(label, dose, ii_h) for label, dose, ii_h, _ in REGIMEN_PATTERNS]
            grid_ft = np.zeros((len(dose_range_ft), len(gfr_range_ft)))

            for i, (dlabel, d, ii_cmp) in enumerate(dose_range_ft):
                nd_cmp = max(3, int(7 * 24 / ii_cmp))
                for j, g in enumerate(gfr_range_ft):
                    df_hm = cached_sim(
                        to_tuple({**patient, 'GFR': float(g)}),
                        to_tuple({**dosing, 'dose_mg': d, 'ii_h': ii_cmp,
                                  'n_doses': nd_cmp, '_fast': True}))
                    grid_ft[i, j] = calc_fTMIC(df_hm, mic_val, ii_cmp, nd_cmp)

            fig_ft_hm = go.Figure(data=go.Heatmap(
                z=grid_ft,
                x=[str(int(g)) for g in gfr_range_ft],
                y=[d[0] for d in dose_range_ft],
                colorscale=[[0, '#e74c3c'], [0.6, '#f1c40f'], [1, '#27ae60']],
                zmin=0, zmax=100,
                text=np.round(grid_ft, 0).astype(int).astype(str),
                texttemplate='%{text}%', textfont=dict(size=12),
                colorbar=dict(title='%fT>MIC'),
            ))
            fig_ft_hm.update_layout(
                title=f"%fT>MIC ヒートマップ (ALB={alb}g/dL, MIC={mic_val} mg/L, 目標: {ftmic_target}%以上)",
                xaxis_title="GFR (mL/min)", yaxis_title="投与パターン",
                xaxis=dict(type='category'), height=500,
            )
            st.plotly_chart(fig_ft_hm, use_container_width=True)

    else:  # 最大 SI (GFR × 投与パターン)
        with st.spinner("ヒートマップ計算中..."):
            gfr_range2 = np.arange(15, 121, 15)
            dose_range2 = [(label, dose, ii_h) for label, dose, ii_h, _ in REGIMEN_PATTERNS]
            grid_si = np.zeros((len(dose_range2), len(gfr_range2)))

            for i, (dlabel, d, ii_cmp) in enumerate(dose_range2):
                nd_cmp = max(3, int(7 * 24 / ii_cmp))
                for j, g in enumerate(gfr_range2):
                    df_hm = cached_sim(
                        to_tuple({**patient, 'GFR': float(g)}),
                        to_tuple({**dosing, 'dose_mg': d, 'ii_h': ii_cmp,
                                  'n_doses': nd_cmp, '_fast': True}))
                    grid_si[i, j] = calc_max_SI(df_hm, ii_cmp, nd_cmp)

            fig_si_hm = go.Figure(data=go.Heatmap(
                z=grid_si,
                x=[str(int(g)) for g in gfr_range2],
                y=[d[0] for d in dose_range2],
                colorscale=[[0, '#27ae60'], [0.5, '#f1c40f'], [1, '#e74c3c']],
                text=np.round(grid_si, 1).astype(str),
                texttemplate='%{text}', textfont=dict(size=12),
                colorbar=dict(title='最大SI'),
            ))
            fig_si_hm.update_layout(
                title=f"最大飽和指数ヒートマップ (ALB={alb}g/dL, Ca²⁺={ca_bile}mmol/L, 食事{meals_per_day}回/日)\n準安定限界: SI>{FIXED['SI_threshold']}",
                xaxis_title="GFR (mL/min)", yaxis_title="投与パターン",
                xaxis=dict(type='category'), height=500,
            )
            st.plotly_chart(fig_si_hm, use_container_width=True)

# ==========================================================================
# Tab 6: Model Info（ノートブック セル3相当）
# ==========================================================================
with tab6:
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("モデル構造")
        st.info(
            "簡略化全身PBPKモデル（7組織コンパートメント）:\n\n"
            "- 動脈血・静脈血プール\n"
            "- 肺（心拍出量の全量を受ける）\n"
            "- 肝臓（胆汁排泄: well-stirred model）\n"
            "- 腎臓（GFR依存の腎クリアランス）\n"
            "- 残余組織（一括）\n"
            "- 胆嚢（食事時刻依存の排出モデル）\n\n"
            "全組織: well-stirred model（灌流律速型分布）\n\n"
            "Kp値: PK-Sim® v12で算出し、肺・肝臓・腎臓・残余組織に同一値（0.22）を適用"
        )

        st.subheader("非線形蛋白結合")
        st.info(
            "Michaelis-Menten型アルブミン結合 (Ewoldt et al. 2023):\n\n"
            "Ct = Cu + Bmax × (ALB/ALB_ref)^θ × Cu / (Kd + Cu)\n\n"
            "- Bmax = 0.55 mmol/L (≈305 mg/L, 健常人)\n"
            "- Kd = 0.030 mmol/L (≈17 mg/L)\n"
            "- θ_ALB = 1.0（アルブミンに比例）"
        )

        st.subheader("胆汁クリアランス")
        st.info(
            "CLbiliary（見かけのCL）= 0.22 L/h\n\n"
            "胆汁排泄量 = CLbiliary × Cp_total（**総血漿中濃度ベース**）\n\n"
            "fu（遊離型分率）の変動は胆汁排泄へ直接反映されない\n\n"
            "低ALB時: fu上昇 → 腎排泄は増加するが、胆汁排泄は変化しない"
        )

    with col_m2:
        st.subheader("薬物パラメータ（セフトリアキソン）")
        df_param = pd.DataFrame({
            'パラメータ': ['分子量', 'logP', 'pKa（酸性）', 'BP比',
                       'fu（公称値）', 'CLrenal', 'CLbiliary', '投与経路'],
            '値': ['554.58 g/mol', '-1.7', '2.7', '0.55',
                  '5-25%（濃度依存的）', '≈ GFR × fu（糸球体濾過）',
                  '0.22 L/h (3.67 mL/min)', '静脈内点滴のみ'],
        }).set_index('パラメータ')
        st.dataframe(df_param, use_container_width=True)

        st.subheader("評価指標の閾値")
        df_thresh = pd.DataFrame({
            '指標': ['%fT>MIC', '飽和指数 (SI)'],
            '閾値': ['≥ 60%', '> 10.4 で沈殿リスク'],
            '根拠': [
                'セファロスポリン系PK/PDターゲット (Craig 1998; Drusano 2003)',
                '胆汁中Ca塩の結晶化実験に基づく参考値 (Shiffman et al. 1990)',
            ],
        }).set_index('指標')
        st.dataframe(df_thresh, use_container_width=True)

        st.subheader("制限事項")
        st.error(
            "- Kp値はPK-Sim® v12に基づく単一値（0.22）を肺・肝臓・腎臓・残余組織へ共通適用している\n"
            "- 本モデルは簡略化PBPKモデルであり、各組織の分布特性を十分には表現できない可能性がある\n"
            "- 胆嚢到達率および排出パラメータは仮定値であり、臨床データによるキャリブレーションは未実施\n"
            "- 胆汁排泄は総濃度ベースの固定CLとして実装しており、低ALB時のfu上昇が胆汁排泄へ直接反映されない\n"
            "- そのため、SIに対するALBの影響は生理学的期待と逆転しうる\n"
            "- 胆嚢内での水分吸収による胆汁濃縮は未反映であり、SIの絶対値は定量的に信頼できない\n"
            "- 腎排泄は糸球体濾過のみで記述し、尿細管分泌は考慮していない\n"
            "- 個体間変動（IIV）および連続点滴は未実装\n"
            "- **教育・研究目的のみ。臨床判断には使用しないでください。**"
        )

        st.subheader("参考文献")
        st.info(
            "- Alasmari F et al. (2023) *Front Pharmacol* 14:1200828 — PBPK構造および胆汁クリアランス設定の参考\n"
            "- Ewoldt TMJ et al. (2023) *J Antimicrob Chemother* 78:1059-1065 — 蛋白結合と遊離型分率の背景\n"
            "- Schleibinger M et al. (2015) *Br J Clin Pharmacol* 80:525-533 — ICU患者における蛋白結合・PKの背景\n"
            "- Shiffman ML et al. (1990) *Gastroenterology* 99:1772-1778 — Ca-CTRX 溶解度積\n"
            "- Craig WA (1998) *Clin Infect Dis* 26:1-10 — β-ラクタム系PK/PDターゲット\n"
            "- Drusano GL (2003) *Clin Infect Dis* 36(Suppl 1):S42-S50 — 耐性抑制のためのPK/PD投与設計\n"
            "- Roberts JA et al. (2014) *Clin Infect Dis* 58:1072-1083 — DALI study"
        )

    st.markdown("---")
    st.markdown("### モデルの留意点")
    st.markdown("""
| 項目 | 本モデルの扱い | 実際 |
|------|--------------|------|
| **腎排泄** | 糸球体濾過のみ（GFR × fu） | セフトリアキソンは一部尿細管分泌もある |
| **胆汁クリアランス** | 総血漿中濃度ベースの固定見かけクリアランス | 実際には能動輸送を含むより複雑な挙動をとる可能性がある |
| **蛋白結合** | 1サイト飽和結合モデル | CTRXの可変・飽和性蛋白結合を教育用に簡略化して表現 |
| **胆嚢到達率** | 時刻依存（空腹時0.5 / 食後0.0、Oddi括約筋の開閉を反映） | 個人差・食事内容で変動 |
| **胆嚢排出** | 食事時刻依存（CCK刺激、基礎排出+食後亢進） | 食事内容・個人差で変動、パラメータは仮定値 |
| **胆嚢濃縮** | 未反映（胆嚢容積一定） | 水分吸収により5〜20倍に濃縮される。絶食時ほど貯留時間が長く濃縮が進む |
| **腸肝循環** | 省略（セフトリアキソンは腸管内で分解されるため臨床的に無視可） | 他の薬物では考慮が必要な場合あり |
""")
