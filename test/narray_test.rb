# mruby-gpu-narray -- known-value tests (no CPU-backend parity; Vulkan only).
# Run: mruby test/narray_test.rb

$pass = 0
$fail = 0

def ok(label)
  $pass += 1
  puts "PASS #{label}"
end

def ng(label, detail)
  $fail += 1
  puts "FAIL #{label}: #{detail}"
end

def assert_ary(label, expected, actual, eps = 1e-3)
  unless actual.is_a?(Array)
    return ng(label, "expected Array, got #{actual.class}")
  end
  if expected.size != actual.size
    return ng(label, "size #{expected.size} != #{actual.size}")
  end
  expected.each_index do |i|
    if (expected[i] - actual[i]).abs > eps
      return ng(label, "index #{i}: expected #{expected[i]}, got #{actual[i]}")
    end
  end
  ok(label)
end

def assert_near(label, expected, actual, eps = 1e-3)
  if (expected - actual).abs <= eps
    ok(label)
  else
    ng(label, "expected #{expected}, got #{actual}")
  end
end

def assert_raise(label, klass)
  begin
    yield
  rescue => e
    if e.is_a?(klass)
      return ok(label)
    else
      return ng(label, "expected #{klass}, got #{e.class}: #{e.message}")
    end
  end
  ng(label, "expected #{klass}, but nothing was raised")
end

puts "device: #{GPU.info[:device]} (#{GPU.info[:backend]})"
puts

# ---- construction / host transfer ----
assert_ary("cast + to_a roundtrip", [1.0, 2.0, 3.0], GPU::SFloat.cast([1, 2, 3]).to_a)
assert_ary("SFloat[] sugar",        [4.0, 5.0, 6.0], GPU::SFloat[4, 5, 6].to_a)
assert_ary("seq default",           [0.0, 1.0, 2.0, 3.0], GPU::SFloat.new(4).seq.to_a)
assert_ary("seq start/step",        [10.0, 12.0, 14.0], GPU::SFloat.new(3).seq(10, 2).to_a)
assert_ary("zeros",                 [0.0, 0.0, 0.0], GPU::SFloat.zeros(3).to_a)
assert_ary("ones",                  [1.0, 1.0, 1.0], GPU::SFloat.ones(3).to_a)
assert_ary("fill",                  [7.0, 7.0], GPU::SFloat.new(2).fill(7).to_a)
assert_ary("head",                  [0.0, 1.0], GPU::SFloat.new(5).seq.head(2))

# ---- shape metadata ----
n5 = GPU::SFloat.new(5)
assert_near("size",  5, n5.size)
assert_near("ndim",  1, n5.ndim)
assert_ary("shape",  [5], n5.shape)

# ---- element-wise: array (op) array ----
a = GPU::SFloat[1, 2, 3, 4]
b = GPU::SFloat[10, 20, 30, 40]
assert_ary("a + b", [11.0, 22.0, 33.0, 44.0], (a + b).to_a)
assert_ary("a - b", [-9.0, -18.0, -27.0, -36.0], (a - b).to_a)
assert_ary("a * b", [10.0, 40.0, 90.0, 160.0], (a * b).to_a)
assert_ary("b / a", [10.0, 10.0, 10.0, 10.0], (b / a).to_a)

# ---- scalar ops (array on the left) ----
assert_ary("a + 100",  [101.0, 102.0, 103.0, 104.0], (a + 100).to_a)
assert_ary("a - 1",    [0.0, 1.0, 2.0, 3.0], (a - 1).to_a)
assert_ary("a * 2",    [2.0, 4.0, 6.0, 8.0], (a * 2).to_a)
assert_ary("a * 0.5",  [0.5, 1.0, 1.5, 2.0], (a * 0.5).to_a)
assert_ary("b / 10",   [1.0, 2.0, 3.0, 4.0], (b / 10).to_a)
assert_ary("-a",       [-1.0, -2.0, -3.0, -4.0], (-a).to_a)

# ---- chained expression (the headline demo) ----
seq5 = GPU::SFloat.new(5).seq          # 0,1,2,3,4
assert_ary("a * 2 + 1", [1.0, 3.0, 5.0, 7.0, 9.0], (seq5 * 2 + 1).to_a)

# ---- reduction: sum / mean ----
# length not a multiple of 256, exercises the tail guard + partial workgroup.
assert_near("sum 1..1000 (n=1000)", 500500.0, GPU::SFloat.new(1000).seq(1, 1).sum)
# integer-valued FP32 sums are exact up to 2^24; 1e6 ones sum exactly.
assert_near("sum 1M ones",          1_000_000.0, GPU::SFloat.new(1_000_000).fill(1.0).sum)
assert_near("mean of ones",         1.0, GPU::SFloat.ones(512).mean)

# ---- errors ----
assert_raise("shape mismatch -> ArgumentError", ArgumentError) do
  GPU::SFloat[1, 2, 3] + GPU::SFloat[1, 2]
end
assert_raise("scalar * narray -> TypeError", TypeError) do
  2 * GPU::SFloat[1, 2, 3]
end
assert_raise("narray * String -> TypeError", TypeError) do
  GPU::SFloat[1, 2, 3] * "nope"
end

# ---- summary ----
puts
puts "#{$pass + $fail} tests, #{$pass} passed, #{$fail} failed"
puts($fail > 0 ? "SOME TESTS FAILED" : "ALL TESTS PASSED")
