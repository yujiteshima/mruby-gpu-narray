require "rqrcode"
require "tempfile"

# URL や文字列を QR コードに変換するためのモジュール
module QRCodeGenerator
  class Error < StandardError; end

  # PNG ラベル描画に使う日本語対応フォントの探索候補 (環境変数 QRCODE_FONT で上書き可)
  FONT_CANDIDATES = [
    ENV["QRCODE_FONT"],
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",       # Linux (Noto)
    "/usr/share/fonts/opentype/unifont/unifont_jp.otf",             # Linux (Unifont JP)
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",                 # Linux (WenQuanYi)
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",               # macOS
    "/System/Library/Fonts/Hiragino Sans GB.ttc",                   # macOS
    "C:/Windows/Fonts/meiryo.ttc",                                  # Windows
    "C:/Windows/Fonts/msgothic.ttc",                                # Windows
  ].compact.freeze

  # 中央ラベルが QR コードを覆う割合の上限 (誤り訂正レベル H は最大30%の欠損を復元
  # できるが、余裕を持たせて4割の幅 × 2割弱の高さに収める)
  CENTER_BOX_WIDTH_RATIO = 0.42
  CENTER_BOX_HEIGHT_RATIO = 0.16

  module_function

  # url と text から QR コードに埋め込むデータを組み立てる。
  # 両方指定された場合は「文字列 + 改行 + URL」の形で埋め込む
  # (多くの QR リーダーはメッセージとリンクの両方を表示できる)。
  def build_payload(url: nil, text: nil)
    parts = [text, url].compact.map(&:strip).reject(&:empty?)
    raise Error, "url か text の少なくとも一方を指定してください" if parts.empty?

    parts.join("\n")
  end

  # 中央にラベルを重ねても読み取れるよう、誤り訂正レベルは最高の :h を既定とする
  def generate(payload, level: :h)
    RQRCode::QRCode.new(payload, level: level)
  end

  # 拡張子に応じて .png / .svg で保存する。
  # label を渡すと文字列を目に見える形で描画する。
  # label_position は :center (QR コード中央に重ねる) か :bottom (画像下部に付ける)。
  def write(qr, path, module_size: 10, label: nil, label_position: :center, font: nil)
    label = nil if label && label.strip.empty?

    case File.extname(path).downcase
    when ".png"
      write_png(qr, path, module_size: module_size, label: label,
                          label_position: label_position, font: font)
    when ".svg"
      File.write(path, svg_document(qr, module_size: module_size, label: label,
                                        label_position: label_position))
    else
      raise Error, "対応していない出力形式です (.png / .svg のみ): #{path}"
    end
  end

  def imagemagick_command
    ["magick", "convert"].find { |cmd| system(cmd, "-version", out: File::NULL, err: File::NULL) }
  end

  def find_label_font
    FONT_CANDIDATES.find { |f| File.exist?(f) }
  end

  def write_png(qr, path, module_size: 10, label: nil, label_position: :center, font: nil)
    png = qr.as_png(module_px_size: module_size, border_modules: 4)
    return png.save(path) unless label

    im = imagemagick_command
    raise Error, "PNG へのラベル描画には ImageMagick が必要です (magick / convert が見つかりません)。" \
                 "--no-label でラベルなし出力もできます" unless im

    font ||= find_label_font
    font_args = font ? ["-font", font] : []

    Tempfile.create(["qrcode", ".png"]) do |tmp|
      png.save(tmp.path)
      args =
        if label_position == :center
          # 中央に白抜きボックス + 文字列を重ねる (label: は折り返さず、
          # -size のボックスに収まるよう文字サイズを自動調整する)
          box_w = (png.width * CENTER_BOX_WIDTH_RATIO).round
          box_h = (png.height * CENTER_BOX_HEIGHT_RATIO).round
          [im, tmp.path,
           "(", "-background", "white", "-fill", "black", *font_args,
           "-size", "#{box_w}x#{box_h}", "-gravity", "center",
           "label:#{label}",
           "-bordercolor", "white", "-border", "#{module_size / 2}",
           ")",
           "-gravity", "center", "-composite", path]
        else
          pointsize = [(module_size * 2.4).round, 12].max
          [im, tmp.path,
           "(", "-background", "white", "-fill", "black", *font_args,
           "-pointsize", pointsize.to_s,
           "-size", "#{png.width}x", "-gravity", "center",
           "caption:#{label}", ")",
           "-append", path]
        end
      system(*args) or raise Error, "ImageMagick によるラベル描画に失敗しました"
    end
  end

  # ラベル付き (または無し) の SVG ドキュメントを純 Ruby で組み立てる
  def svg_document(qr, module_size: 10, label: nil, label_position: :center)
    inner = qr.as_svg(module_size: module_size)
    return inner unless label

    width = inner[/width="(\d+)"/, 1].to_i
    height = inner[/height="(\d+)"/, 1].to_i
    body = inner.sub(/<\?xml[^>]*\?>\s*/, "")

    if label_position == :center
      svg_with_center_label(body, width, height, label)
    else
      svg_with_bottom_label(body, width, height, label, module_size)
    end
  end

  def svg_with_center_label(body, width, height, label)
    lines = label.split("\n")
    max_chars = lines.map { |l| l.each_char.sum { |c| c.bytesize > 1 ? 1.0 : 0.55 } }.max
    box_w = (width * CENTER_BOX_WIDTH_RATIO).round
    box_h = (height * CENTER_BOX_HEIGHT_RATIO).round
    # 全角文字の幅 ≒ フォントサイズとして、ボックス幅と高さの両方に収まるサイズを選ぶ
    font_size = [(box_w * 0.94 / max_chars), box_h * 0.8 / lines.length].min.floor
    line_height = (font_size * 1.2).round
    box_x = (width - box_w) / 2
    box_y = (height - box_h) / 2
    text_top = height / 2 - line_height * (lines.length - 1) / 2

    texts = lines.each_with_index.map do |line, i|
      %(  <text x="#{width / 2}" y="#{text_top + line_height * i}" text-anchor="middle" ) +
        %(dominant-baseline="central" font-family="'Noto Sans CJK JP','Hiragino Sans','Meiryo',sans-serif" ) +
        %(font-size="#{font_size}" fill="#000000">#{escape_xml(line)}</text>)
    end.join("\n")

    <<~SVG
      <?xml version="1.0" standalone="yes"?>
      <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="#{width}" height="#{height}">
        #{body}
        <rect x="#{box_x}" y="#{box_y}" width="#{box_w}" height="#{box_h}" rx="6" fill="#ffffff"/>
      #{texts}
      </svg>
    SVG
  end

  def svg_with_bottom_label(body, width, height, label, module_size)
    font_size = [module_size * 2.4, 12].max.round
    line_height = (font_size * 1.5).round
    lines = label.split("\n")
    label_height = line_height * lines.length + font_size

    texts = lines.each_with_index.map do |line, i|
      y = height + line_height * (i + 1)
      %(  <text x="#{width / 2}" y="#{y}" text-anchor="middle" ) +
        %(font-family="'Noto Sans CJK JP','Hiragino Sans','Meiryo',sans-serif" ) +
        %(font-size="#{font_size}" fill="#000000">#{escape_xml(line)}</text>)
    end.join("\n")

    <<~SVG
      <?xml version="1.0" standalone="yes"?>
      <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="#{width}" height="#{height + label_height}">
        <rect width="100%" height="100%" fill="#ffffff"/>
        #{body}
      #{texts}
      </svg>
    SVG
  end

  def escape_xml(str)
    str.gsub("&", "&amp;").gsub("<", "&lt;").gsub(">", "&gt;")
  end
end
