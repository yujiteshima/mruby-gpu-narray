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

  module_function

  # url と text から QR コードに埋め込むデータを組み立てる。
  # 両方指定された場合は「文字列 + 改行 + URL」の形で埋め込む
  # (多くの QR リーダーはメッセージとリンクの両方を表示できる)。
  def build_payload(url: nil, text: nil)
    parts = [text, url].compact.map(&:strip).reject(&:empty?)
    raise Error, "url か text の少なくとも一方を指定してください" if parts.empty?

    parts.join("\n")
  end

  def generate(payload)
    RQRCode::QRCode.new(payload)
  end

  # 拡張子に応じて .png / .svg で保存する。
  # label を渡すと QR コードの下にその文字列を目に見える形で描画する。
  def write(qr, path, module_size: 10, label: nil, font: nil)
    label = nil if label && label.strip.empty?

    case File.extname(path).downcase
    when ".png"
      write_png(qr, path, module_size: module_size, label: label, font: font)
    when ".svg"
      File.write(path, svg_document(qr, module_size: module_size, label: label))
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

  def write_png(qr, path, module_size: 10, label: nil, font: nil)
    png = qr.as_png(module_px_size: module_size, border_modules: 4)
    return png.save(path) unless label

    im = imagemagick_command
    raise Error, "PNG へのラベル描画には ImageMagick が必要です (magick / convert が見つかりません)。" \
                 "--no-label でラベルなし出力もできます" unless im

    font ||= find_label_font
    pointsize = [(module_size * 2.4).round, 12].max

    Tempfile.create(["qrcode", ".png"]) do |tmp|
      png.save(tmp.path)
      args = [im, tmp.path,
              "(", "-background", "white", "-fill", "black",
              *(font ? ["-font", font] : []),
              "-pointsize", pointsize.to_s,
              "-size", "#{png.width}x", "-gravity", "center",
              "caption:#{label}", ")",
              "-append", path]
      system(*args) or raise Error, "ImageMagick によるラベル描画に失敗しました"
    end
  end

  # ラベル付き (または無し) の SVG ドキュメントを純 Ruby で組み立てる
  def svg_document(qr, module_size: 10, label: nil)
    inner = qr.as_svg(module_size: module_size)
    return inner unless label

    width = inner[/width="(\d+)"/, 1].to_i
    height = inner[/height="(\d+)"/, 1].to_i
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
        #{inner.sub(/<\?xml[^>]*\?>\s*/, "")}
      #{texts}
      </svg>
    SVG
  end

  def escape_xml(str)
    str.gsub("&", "&amp;").gsub("<", "&lt;").gsub(">", "&gt;")
  end
end
