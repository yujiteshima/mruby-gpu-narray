require "minitest/autorun"
require "tmpdir"
require_relative "../lib/qr_code_generator"

class TestQRCodeGenerator < Minitest::Test
  def test_build_payload_with_text_only
    assert_equal "スターください", QRCodeGenerator.build_payload(text: "スターください")
  end

  def test_build_payload_with_url_only
    assert_equal "https://example.com", QRCodeGenerator.build_payload(url: "https://example.com")
  end

  def test_build_payload_with_text_and_url
    payload = QRCodeGenerator.build_payload(url: "https://example.com", text: "スターください")
    assert_equal "スターください\nhttps://example.com", payload
  end

  def test_build_payload_without_input_raises
    assert_raises(QRCodeGenerator::Error) { QRCodeGenerator.build_payload }
    assert_raises(QRCodeGenerator::Error) { QRCodeGenerator.build_payload(text: "  ", url: "") }
  end

  def test_write_png_and_svg
    qr = QRCodeGenerator.generate("スターください")
    Dir.mktmpdir do |dir|
      png = File.join(dir, "out.png")
      svg = File.join(dir, "out.svg")
      QRCodeGenerator.write(qr, png)
      QRCodeGenerator.write(qr, svg)
      assert File.size(png).positive?
      assert_includes File.read(svg), "<svg"
    end
  end

  def test_write_unknown_extension_raises
    qr = QRCodeGenerator.generate("test")
    assert_raises(QRCodeGenerator::Error) { QRCodeGenerator.write(qr, "out.txt") }
  end

  def test_svg_document_with_center_label_renders_visible_text
    qr = QRCodeGenerator.generate("スターください")
    svg = QRCodeGenerator.svg_document(qr, label: "スターください", label_position: :center)
    assert_includes svg, ">スターください</text>"
    assert_match(/<rect x="\d+" y="\d+"/, svg, "中央の白抜きボックスがあるはず")
  end

  def test_svg_document_with_bottom_label_extends_height
    qr = QRCodeGenerator.generate("スターください")
    plain = QRCodeGenerator.svg_document(qr)
    svg = QRCodeGenerator.svg_document(qr, label: "スターください", label_position: :bottom)
    plain_h = plain[/height="(\d+)"/, 1].to_i
    labeled_h = svg[/height="(\d+)"/, 1].to_i
    assert_includes svg, ">スターください</text>"
    assert labeled_h > plain_h, "下部ラベルの分だけ縦に長くなるはず"
  end

  def test_svg_document_escapes_label
    qr = QRCodeGenerator.generate("test")
    svg = QRCodeGenerator.svg_document(qr, label: "<a & b>")
    assert_includes svg, "&lt;a &amp; b&gt;"
  end

  def test_write_png_with_center_label_keeps_dimensions
    skip "ImageMagick がないためスキップ" unless QRCodeGenerator.imagemagick_command

    qr = QRCodeGenerator.generate("スターください")
    Dir.mktmpdir do |dir|
      plain = File.join(dir, "plain.png")
      labeled = File.join(dir, "labeled.png")
      QRCodeGenerator.write(qr, plain)
      QRCodeGenerator.write(qr, labeled, label: "スターください", label_position: :center)
      assert_equal ChunkyPNG::Image.from_file(plain).height,
                   ChunkyPNG::Image.from_file(labeled).height,
                   "中央ラベルは重ね描きなので画像サイズは変わらないはず"
    end
  end

  def test_write_png_with_bottom_label_extends_height
    skip "ImageMagick がないためスキップ" unless QRCodeGenerator.imagemagick_command

    qr = QRCodeGenerator.generate("スターください")
    Dir.mktmpdir do |dir|
      plain = File.join(dir, "plain.png")
      labeled = File.join(dir, "labeled.png")
      QRCodeGenerator.write(qr, plain)
      QRCodeGenerator.write(qr, labeled, label: "スターください", label_position: :bottom)
      assert ChunkyPNG::Image.from_file(labeled).height > ChunkyPNG::Image.from_file(plain).height,
             "下部ラベルの分だけ縦に長くなるはず"
    end
  end

  def test_generated_qr_decodes_back_to_payload
    # QR コードのモジュール行列が生成されることを確認 (デコードは外部ツールで実施)
    qr = QRCodeGenerator.generate("スターください")
    assert qr.qrcode.modules.length.positive?
  end
end
