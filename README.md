# セフトリアキソン PBPK シミュレーター

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ceftriaxone-pbpk-simulator.streamlit.app/)

治療効果と偽胆石リスクの同時評価

## 概要

このシミュレーターは以下の臨床的問いに対応するために設計されました：

1. **治療効果**: 任意の患者・投与レジメンで、治療目標（%fT>MIC）を達成できるか
2. **偽胆石リスク**: 任意の患者・投与レジメンで、胆嚢内Ca-CTRX飽和指数（SI）が準安定限界を超えるか

## モデルの特徴

### 構造

- 簡略化PBPKモデル（7コンパートメント）
  - 動脈血 / 静脈血 / 肺 / 肝臓 / 腎臓 / 残余組織 / 胆嚢
- Well-stirred model（灌流律速）
- Poulin-Theil推定によるKp値
- ODE求解: `scipy.integrate.solve_ivp`（LSODA法）

### セフトリアキソン固有の機能

- **非線形蛋白結合**: Michaelis-Menten型（Ewoldt et al. 2023, JAC）
  - Bmax = 0.55 mmol/L, Kd = 0.030 mmol/L
  - アルブミン濃度依存（θ_ALB = 1.0）
- **GFR依存腎クリアランス**: 糸球体濾過による遊離型排泄
- **胆汁排泄**: CLbiliary = 0.22 L/h（Alasmari et al. 2023）
- **胆嚢コンパートメント**: 胆汁濃縮・貯留
  - 食事時刻依存の排出モデル（CCK刺激による胆嚢収縮を反映）
  - 基礎排出速度 0.05 /h（MMC）+ 食後刺激 1.0 /h（2時間持続、CCK応答）

### 調整可能な患者パラメータ

- 体重（35–100 kg）
- 血清アルブミン値（1.5–4.5 g/dL）
- GFR（15–120 mL/min）
- 食事回数（0–3回/日、時刻は自動設定: 8時/12時/18時）

### 固定パラメータ（胆嚢モデル）

- 胆嚢胆汁中 Ca²⁺ 濃度: 5.0 mmol/L
- 胆嚢到達率: 時刻依存（空腹時 0.5 / 食後 0.0、Oddi括約筋の開閉を反映）
- Ca-CTRX 溶解度積 (Ksp): 1.62 × 10⁻⁶（Shiffman et al. 1990）
- 準安定限界: SI > 10.4（同上、in vitro 実験に基づく参考値）

### 投与設計

- 1回投与量: 500 / 1000 / 2000 mg
- 点滴時間: 30 / 60分
- 投与間隔: 12h / 24h
- 投与時刻: q24h → 9時、q12h → 9時/21時
- 治療期間: 7日（固定）
- MICプリセット（CLSI M100 準拠、カスタム値も設定可）
  - Enterobacterales / S. pneumoniae（髄膜炎・非髄膜炎）/ H. influenzae / N. meningitidis / N. gonorrhoeae
- 目標 %fT>MIC: 40–100%（スライダーで調整可）

## アプリのタブ構成

| タブ | 内容 |
| ------ | ------ |
| 📈 血漿中濃度推移 | 総濃度・遊離型濃度の経時推移、排泄経路（腎/胆汁）、PK要約テーブル（AUCss含む） |
| 🎯 %fT>MIC 解析 | 定常状態の遊離型濃度 vs MIC、投与レジメン比較（5パターン） |
| ⚠️ 偽胆石リスク | Ca-CTRX飽和指数（SI）推移、SI-AUC、超過時間割合、リスク評価 |
| 🔬 感度分析 | 体重・血清アルブミン値・GFR・食事回数による%fT>MIC・SIへの影響 |
| 🗺️ ヒートマップ | %fT>MIC（ALB×GFR / GFR×投与パターン）、最大SI（ALB×GFR / GFR×投与パターン） |
| ℹ️ モデル情報 | モデル構造・パラメータ・閾値・制限事項・参考文献 |

## ファイル構成

```text
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
- 胆嚢到達率（空腹時0.5/食後0.0）および排出パラメータは仮定値
- 腸肝循環は臨床的に無視できるため省略（腸管内で分解）
- 腎排泄は糸球体濾過のみ（尿細管分泌省略）
- 個体間変動（IIV）未実装
- BP比は条件によらず一定と仮定
- **教育・研究目的のみ。臨床判断には使用しないでください。**

## 参考文献

- Alasmari F et al. (2023) *Front Pharmacol* 14:1200828
- Ewoldt TMJ et al. (2023) *J Antimicrob Chemother* 78:1059-1065
- Schleibinger M et al. (2015) *Br J Clin Pharmacol* 80:525-533
- Shiffman ML et al. (1990) *Gastroenterology* 99:1772-1778
- Craig WA (1998) *Clin Infect Dis* 26:1-10
- Drusano GL (2003) *Clin Infect Dis* 36(Suppl 1):S42-S50
- Roberts JA et al. (2014) *Clin Infect Dis* 58:1072-1083
