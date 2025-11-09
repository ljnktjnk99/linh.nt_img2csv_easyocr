# OCR Table to CSV 変換ツール

EasyOCRを使用して、表の画像をCSVファイルに変換するツールです。
Hough Line Transformを用いた自動傾き補正機能を搭載しています。

## 実行方法

### 1. 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

**注意**: 初回実行時、EasyOCRが自動的にモデルファイル（約700MB）をダウンロードします。

### 2. 実行

#### 基本的な使い方

```bash
python img_to_csv.py table.png
```

実行後、以下のファイルが生成されます：
- `output.csv` - 変換結果のCSVファイル
- `table_deskewed.png` - 傾き補正後の画像（傾きがある場合のみ）

#### オプション付きの実行

```bash
# 出力ファイル名を指定
python img_to_csv.py table.png --output result.csv

# デバッグモード（検出された直線を可視化）
python img_to_csv.py table.png --debug
```

#### ヘルプの表示

```bash
python img_to_csv.py --help
```

## 処理フローの概要

以下の3ステップで画像をCSVに変換します：

```
入力画像 → [1] 傾き検出・補正 → [2] OCR実行 → [3] 表構造解析 → CSV出力
```

### ステップ1: 傾き検出・補正

**Hough Line Transform**を使用して画像の傾きを自動検出し、補正します。

- グレースケール化 → エッジ検出 → 直線検出 → 角度計算
- 0.5度以上の傾きがある場合のみ補正を実行

### ステップ2: OCR実行

**EasyOCR**で文字認識を行います。

- 対応言語：日本語、英語
- テキスト、座標、信頼度スコアを取得

### ステップ3: 表構造解析とCSV出力

テキストボックスの座標から表構造を推定します。

- Y座標で行をグループ化
- X座標で列をソート
- UTF-8 BOMエンコーディングでCSV出力