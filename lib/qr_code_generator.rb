require "rqrcode"

# URL や文字列を QR コードに変換するためのモジュール
module QRCodeGenerator
  class Error < StandardError; end

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

  # 拡張子に応じて .png / .svg で保存する
  def write(qr, path, module_size: 10)
    case File.extname(path).downcase
    when ".png"
      qr.as_png(module_px_size: module_size, border_modules: 4).save(path)
    when ".svg"
      File.write(path, qr.as_svg(module_size: module_size))
    else
      raise Error, "対応していない出力形式です (.png / .svg のみ): #{path}"
    end
  end
end
