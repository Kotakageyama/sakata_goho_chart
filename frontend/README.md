# World ID奨学金支援システム - フロントエンド

## 概要

World ID認証後にユーザータイプを選択し、各ダッシュボードにリダイレクトするフロントエンドアプリケーションです。

## 実装済み機能

### 1. ユーザータイプ選択 (UserTypeSelection.tsx)
- 3つのカードを表示：
  - 奨学生 (🎓)
  - 個人支援者 (👤)
  - 企業支援者 (🏢)
- 選択後、Context に保存してダッシュボードへリダイレクト

### 2. React Router設定
- `/select-type` - ユーザータイプ選択画面
- `/student/dashboard` - 奨学生ダッシュボード
- `/individual-supporter/dashboard` - 個人支援者ダッシュボード
- `/corporate-supporter/dashboard` - 企業支援者ダッシュボード

### 3. Context API
- `UserTypeContext`: ユーザータイプの状態管理
- `useUserType`: カスタムフック

### 4. レスポンシブデザイン
- モバイル対応済み
- モダンなUI/UXデザイン
- カードホバーエフェクト

## ディレクトリ構成

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── UserTypeSelection.tsx
│   │   ├── UserTypeSelection.css
│   │   ├── StudentDashboard.tsx
│   │   ├── IndividualSupporterDashboard.tsx
│   │   ├── CorporateSupporterDashboard.tsx
│   │   └── Dashboard.css
│   ├── contexts/
│   │   └── UserTypeContext.tsx
│   ├── App.tsx
│   ├── App.css
│   ├── index.tsx
│   └── index.css
├── package.json
├── tsconfig.json
└── README.md
```

## セットアップ手順

### 1. 依存関係のインストール
```bash
cd frontend
npm install
```

### 2. アプリケーションの起動
```bash
npm start
```

### 3. ブラウザでアクセス
```
http://localhost:3000
```

## 主要な技術スタック

- **React 18.2.0** - UIライブラリ
- **TypeScript** - 型安全性
- **React Router 6.3.0** - ルーティング
- **Context API** - 状態管理
- **CSS3** - スタイリング

## 使用方法

1. アプリケーションを起動すると、ユーザータイプ選択画面が表示されます
2. 3つのカードから自分の役割を選択します：
   - **奨学生**: 奨学金を受け取る学生
   - **個人支援者**: 個人として学生を支援する人
   - **企業支援者**: 企業として学生を支援する組織
3. 選択後、自動的に対応するダッシュボードにリダイレクトされます
4. 各ダッシュボードでは、役割に応じた機能が利用できます

## 実装の特徴

### ユーザータイプ選択フロー
- 美しいカードUIで直感的な選択
- ホバーエフェクトとスムーズなアニメーション
- アクセシビリティを考慮した設計

### 状態管理
- Context APIによる軽量な状態管理
- TypeScriptによる型安全性
- カスタムフックによる再利用可能な状態ロジック

### レスポンシブデザイン
- モバイルファーストのアプローチ
- フレキシブルなレイアウト
- タッチフレンドリーなUI

## 今後の拡張予定

- World ID認証機能の統合
- 各ダッシュボードの詳細機能実装
- 多言語対応
- PWA対応
- テスト実装

## 完了条件の確認

✅ **認証後に正しく選択画面へ遷移**: `/select-type` ルートで選択画面が表示される
✅ **選択結果に応じて URL が変わる**: 各ユーザータイプに対応したダッシュボードURLにリダイレクト
✅ **UserTypeSelection.tsx で 3 つの Card を表示**: 奨学生、個人支援者、企業支援者の3つのカードを実装
✅ **React Router で /:type/dashboard ルートを設定**: 各ユーザータイプに対応したダッシュボードルートを設定