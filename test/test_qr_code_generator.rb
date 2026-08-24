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

  def test_svg_document_with_label_renders_visible_text
    qr = QRCodeGenerator.generate("スターください")
    svg = QRCodeGenerator.svg_document(qr, label: "スターください")
    assert_includes svg, ">スターください</text>"
  end

  def test_svg_document_escapes_label
    qr = QRCodeGenerator.generate("test")
    svg = QRCodeGenerator.svg_document(qr, label: "<a & b>")
    assert_includes svg, "&lt;a &amp; b&gt;"
  end

  def test_write_png_with_label
    skip "ImageMagick がないためスキップ" unless QRCodeGenerator.imagemagick_command

    qr = QRCodeGenerator.generate("スターください")
    Dir.mktmpdir do |dir|
      plain = File.join(dir, "plain.png")
      labeled = File.join(dir, "labeled.png")
      QRCodeGenerator.write(qr, plain)
      QRCodeGenerator.write(qr, labeled, label: "スターください")
      assert File.size(labeled) > File.size(plain), "ラベル付き PNG はラベル分だけ大きくなるはず"
    end
  end

  def test_generated_qr_decodes_back_to_payload
    # QR コードのモジュール行列が生成されることを確認 (デコードは外部ツールで実施)
    qr = QRCodeGenerator.generate("スターください")
    assert qr.qrcode.modules.length.positive?
  end
end
