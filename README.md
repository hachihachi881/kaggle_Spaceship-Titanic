# Spaceship Titanic データセット README

## 概要

このコンペティションでは、宇宙船タイタニック号が**時空異常と衝突した際に乗客が別次元へ転送されたかどうか**を予測することが課題です。

予測に役立てるため、損傷した船内コンピュータから回収された**乗客の個人記録データ**が提供されています。

機械学習モデルを用いて、乗客が別次元へ**転送されたか (Transported)** を予測します。

---

# データ構成

## train.csv

乗客のおよそ **3分の2（約8700人分）** の個人記録が含まれています。
機械学習モデルの **学習用データ**として使用します。

### 各カラムの説明

#### PassengerId

各乗客の一意のID。

`gggg_pp` の形式で表されます。

* `gggg` : 同じグループ番号
* `pp` : グループ内の個人番号

同じグループの乗客は家族であることが多いですが、必ずしもそうとは限りません。

---

#### HomePlanet

乗客が出発した惑星（通常は居住している惑星）。

---

#### CryoSleep

航行中に**冷凍睡眠（仮死状態）**に入ることを選択したかどうか。

* `True` : 冷凍睡眠状態
* `False` : 通常状態

冷凍睡眠中の乗客は自室に留まります。

---

#### Cabin

客室番号。

形式

```
deck/num/side
```

* `deck` : デッキ番号
* `num` : 客室番号
* `side` : 船の左右

side の意味

* `P` = Port（左舷）
* `S` = Starboard（右舷）

---

#### Destination

乗客が下船する予定の惑星。

---

#### Age

乗客の年齢。

---

#### VIP

航行中に**VIPサービス**を利用したかどうか。

* `True` : VIPサービス利用
* `False` : 利用なし

---

#### 船内施設の利用金額

以下は船内の高級施設での利用金額です。

* `RoomService`
* `FoodCourt`
* `ShoppingMall`
* `Spa`
* `VRDeck`

---

#### Name

乗客の氏名（名・姓）。

---

#### Transported

乗客が**別次元へ転送されたかどうか**を示します。

* `True` : 転送された
* `False` : 転送されていない

このカラムが**予測対象（目的変数）**です。

---

# test.csv

残り約 **3分の1（約4300人分）** の乗客データです。

このデータには **Transported 列は含まれていません。**

機械学習モデルを用いて **Transported を予測**します。

---

# sample_submission.csv

提出フォーマットのサンプルファイルです。

### カラム

#### PassengerId

テストデータの乗客ID。

#### Transported

各乗客について以下のどちらかを予測して記入します。

```
True
False
```

---

# コンペティションの目的

提供された乗客データを用いて、機械学習モデルを構築し、
**乗客が別次元へ転送されたかどうかを高精度で予測すること**が目的です。

---

# 実装 (code.ipynb)

## 使用ライブラリ

- `pandas` / `numpy` : データ処理
- `scikit-learn` : 前処理・モデル評価
- `XGBoost` / `LightGBM` : 分類モデル

## パイプライン概要

### 1. データ読み込み

`train.csv` と `test.csv` を読み込み、前処理のために結合 (`all_df`) します。

```python
all_df = pd.concat([df, test_df])
```

### 2. 特徴量エンジニアリング

| 特徴量 | 内容 |
|--------|------|
| `TotalSpend` | 5つの船内施設の合計支出 |
| `NoSpend` | 支出が0かどうか（冷凍睡眠との相関が高い） |
| `LuxurySpend` | Spa + VRDeck（贅沢支出） |
| `NecessitySpend` | RoomService + FoodCourt（生活支出） |
| `Log_*` / `LogTotalSpend` | 各支出の対数変換（スキュー補正） |
| `Deck` | Cabin から抽出したデッキ（A〜G） |
| `CabinNum` | Cabin から抽出した区画番号 |
| `Side` | Cabin から抽出した船のP/S側 |
| `AgeBin` | 年齢を Child / Teen / Young / Adult / Senior に分類 |
| `Group` / `GroupSize` | PassengerId からグループ情報を抽出 |
| `IsAlone` | グループサイズが1かどうか（一人旅フラグ） |

### 3. 前処理

- 支出列の欠損値を `0` で補完
- `pd.get_dummies` でカテゴリ変数をone-hot encoding
- 残った欠損値を中央値 → `0` の順で補完

### 4. モデル学習

XGBoost と LightGBM の両方を学習し、バリデーション精度の高い方を採用します。

**XGBoost パラメータ**
```python
XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
              subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30)
```

**LightGBM パラメータ**
```python
LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
               subsample=0.8, colsample_bytree=0.8)
```

train/valid = 80/20 で分割し、early stopping で過学習を防ぎます。

### 5. 精度推移

| 手順 | モデル | Accuracy |
|------|--------|----------|
| ベースライン | LogisticRegression | 0.7746 |
| 特徴量追加 | LogisticRegression | 0.7832 |
| モデル刷新 | XGBoost / LightGBM | **0.8033** |

### 6. 重要特徴量 (Top 5)

1. `CabinNum`（区画番号）
2. `LuxurySpend`（Spa + VRDeck）
3. `Age`
4. `FoodCourt`
5. `TotalSpend`

### 7. 提出ファイル生成

```python
submission.to_csv("subnission.csv", index=False)
```

