# セフトリアキソン PBPK シミュレーター（試行錯誤中）

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ceftriaxone-pbpk-simulator.streamlit.app/)

scipy.integrate + Streamlit によるセフトリアキソンの生理学的薬物動態（PBPK）シミュレーター。

## 概要

このシミュレーターは以下の臨床的問いに対応するために設計されました：

1. **治療効果（%fT>MIC）**: 低アルブミン・CKD患者で現行投与レジメンが目標を達成できるか
2. **偽胆石リスク**: 胆汁中セフトリアキソン濃度がCa-CTRX沈殿閾値を超えるか

## モデルの特徴

### 構造

- 簡略化全身PBPKモデル（7コンパートメント）
  - 動脈血 / 静脈血 / 肺 / 肝臓 / 腎臓 / 残余組織 / 胆嚢
- Well-stirred model（灌流律速）
- Poulin-Theil推定によるKp値
- ODE求解: `scipy.integrate.solve_ivp`（LSODA法）

### セフトリアキソン固有の機能

- **非線形蛋白結合**: Michaelis-Menten型（Gijsen et al. 2023, JAC）
  - Bmax = 0.55 mmol/L, Kd = 0.030 mmol/L
  - アルブミン濃度依存（θ_ALB = 1.0）
- **GFR依存腎クリアランス**: 糸球体濾過による遊離型排泄
- **胆汁排泄**: CLbiliary = 0.22 L/h（Alasmari et al. 2023）
- **胆嚢コンパートメント**: 胆汁濃縮・貯留（食事回数による排出速度調整）

### 調整可能な患者パラメータ

- 体重（30–120 kg）
- 血清アルブミン値（1.0–5.0 g/dL）
- GFR（5–150 mL/min）
- 胆汁中 Ca²⁺ 濃度（1.0–10.0 mmol/L）
- 胆汁濃縮係数（1–10倍）
- 食事回数（0–3回/日）

### 投与設計

- 投与量: 500 / 1000 / 2000 mg
- 点滴時間: 5–60分
- 投与間隔: 12h / 24h
- 投与回数: 1–14回
- MICプリセット（カスタム値も設定可）

## アプリのタブ構成

| タブ | 内容 |
|------|------|
| 📈 血漿中濃度推移 | 総濃度・遊離型濃度の経時推移、排泄経路（腎/胆汁）、PK要約テーブル |
| 🎯 %fT>MIC 解析 | 定常状態の遊離型濃度 vs MIC、5パターン投与量比較 |
| ⚠️ 偽胆石リスク | Ca-CTRX飽和指数（SI）推移、胆嚢内濃度推移、リスク評価 |
| 🔬 感度分析 | GFR / アルブミン値による%fT>MIC・SIへの影響 |
| 🗺️ ヒートマップ | ALB×GFRの%fT>MICマップ、GFR×投与パターンのSIマップ、ALB×GFRのSIマップ |
| ℹ️ モデル情報 | モデル構造・パラメータ・閾値・制限事項・参考文献 |

## ファイル構成

```
ceftriaxone-pbpk-simulator/
├── README.md              # このファイル
├── streamlit_app.py       # Streamlitアプリ（ODE・UI一体型）
└── requirements.txt       # Python依存パッケージ
```

## 依存パッケージ

- numpy
- scipy
- pandas
- streamlit (≥ 1.32.0)
- plotly

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 制限事項

- Kp値はPK-Sim最適化値ではなく推定値
- 胆嚢モデルは簡略化（腸肝循環なし）
- 腎排泄は糸球体濾過のみ（尿細管分泌省略）
- 個体間変動（IIV）未実装
- BP比は条件によらず一定と仮定
- **教育・研究目的のみ。臨床判断には使用しないでください。**

## 参考文献

- Alasmari F et al. (2023) *Front Pharmacol* 14:1200828
- Gijsen M et al. (2023) *J Antimicrob Chemother* 78:1059-1067
- Schleibinger M et al. (2015) *Br J Clin Pharmacol* 80:1142-1151
- Shiffman ML et al. (1990) *Gastroenterology* 99:1772
- Craig WA (1998) *Clin Infect Dis* 26:1-10
- Drusano GL (2004) *Clin Infect Dis* 39:S45-S53
