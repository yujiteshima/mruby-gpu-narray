# qrcode-generator

URL や任意の文字列を埋め込んだ QR コードを生成する Ruby 製 CLI ツールです。

- 任意の文字列（例:「スターください」）を QR コードに埋め込める
- URL を QR コードに埋め込める
- URL と文字列の両方を指定した場合は「文字列 + 改行 + URL」として埋め込む
- `--text` 指定時は既定で QR コードの**中央に文字列を目に見える形でも表示**する
  （`--label-pos bottom` で画像下部への表示にも変更可能）
- 出力形式: PNG / SVG / ターミナル表示（ANSI）

> **文字列の「埋め込み」について**: QR コード本体の白黒パターンにはデータとして
> 文字列がエンコードされており、スマートフォン等でスキャンすると読み取れます。
> それとは別に、このツールは QR コード中央（または下部）に文字列をテキストとして
> 描画します。中央に重ねても読み取れるよう、誤り訂正レベルは最高の H（30% まで
> 復元可能）で生成し、文字列のボックスは安全な大きさに収めています。

## セットアップ

Ruby 3.x が必要です。

```sh
bundle install
```

（Bundler を使わない場合は `gem install rqrcode` でも動作します）

PNG にラベル（画像下の文字列）を描画する場合は [ImageMagick](https://imagemagick.org/)
が必要です（`magick` または `convert` コマンド）。SVG のラベル描画に追加の依存はありません。

```sh
# Ubuntu / Debian
sudo apt install imagemagick
# macOS
brew install imagemagick
```

## 使い方

```sh
# 文字列「スターください」を埋め込んだ QR コードを PNG で出力
bin/qrcode --text "スターください" -o star.png

# URL を埋め込んだ QR コードを SVG で出力
bin/qrcode --url "https://github.com/yujiteshima" -o profile.svg

# URL と文字列の両方を埋め込む（リーダーにはメッセージとリンクの両方が表示される）
bin/qrcode --url "https://github.com/yujiteshima/qrcode-generator" \
           --text "スターください" \
           -o star_repo.png

# 出力ファイルを指定しない場合はターミナルに QR コードを表示
bin/qrcode --text "スターください"
```

### オプション

| オプション | 説明 |
| --- | --- |
| `-t`, `--text TEXT` | QR コードに埋め込む文字列 |
| `-u`, `--url URL` | QR コードに埋め込む URL |
| `-o`, `--output FILE` | 出力ファイル（`.png` / `.svg`）。省略時はターミナル表示 |
| `-s`, `--size N` | 1 モジュールあたりのピクセル数（既定: 10） |
| `-l`, `--label TEXT` | 画像に表示する文字列（既定: `--text` の内容） |
| `-p`, `--label-pos POS` | 表示位置: `center` = QR コード中央（既定）/ `bottom` = 画像下部 |
| `--no-label` | 画像への文字列表示を行わない |
| `--font PATH` | PNG ラベル用フォントファイル（既定: 日本語フォントを自動検出。環境変数 `QRCODE_FONT` でも指定可） |
| `-h`, `--help` | ヘルプを表示 |

`--text` と `--url` は少なくとも一方の指定が必要です。

## サンプル

`examples/` に生成済みのサンプルがあります。

- `examples/star_kudasai.png` —「スターください」を埋め込んだ QR コード（中央にも文字列を表示）
- `examples/star_kudasai.svg` — 同上の SVG 版
- `examples/star_with_url.png` —「スターください」+ URL を埋め込んだ QR コード（スキャン結果はテキスト扱い）
- `examples/star_link_only.png` — データは URL のみで、中央の表示だけ「スターください」
  （スキャンするとすぐリンクを開ける。スター依頼にはこちらがおすすめ）

再生成する場合:

```sh
bin/qrcode --text "スターください" -o examples/star_kudasai.png
bin/qrcode --text "スターください" -o examples/star_kudasai.svg
bin/qrcode --text "スターください" --url "https://github.com/yujiteshima/mruby-gpu-narray" -o examples/star_with_url.png
bin/qrcode --url "https://github.com/yujiteshima/mruby-gpu-narray" --label "スターください" -o examples/star_link_only.png
```

> URL には実在するページを指定してください。存在しない URL を埋め込むと、
> QR コードは正しく読み取れてもアクセス時に 404 になります。

## ライブラリとして使う

```ruby
require_relative "lib/qr_code_generator"

payload = QRCodeGenerator.build_payload(url: "https://example.com", text: "スターください")
qr = QRCodeGenerator.generate(payload)
QRCodeGenerator.write(qr, "out.png", module_size: 10)
```

## テスト

```sh
ruby test/test_qr_code_generator.rb
```
