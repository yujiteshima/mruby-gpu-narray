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

# ---- spectral transform: rfft / magnitude / power_spectrum ----
# n = 8 against a hand-computed DFT. A constant signal puts all energy in bin 0.
const8 = GPU::SFloat.new(8).fill(2.0).rfft
assert_near("rfft(const) size",   8, const8.size)
assert_near("rfft(const) bin 0",  16.0, const8.to_a[0][0])
assert_near("rfft(const) bin 0 imag", 0.0, const8.to_a[0][1])
assert_near("rfft(const) bin 1",  0.0, const8.to_a[1][0])

# cos(2*pi*k*t/n) has magnitude n/2 at bin k (and at bin n-k).
PI = 3.141592653589793

# Builds cos(2*pi*freq*t/len) for t in 0...len. (Math comes from the default
# gembox's mruby-math.)
def tone(len, freq, sine = false)
  a = []
  t = 0
  while t < len
    ang = 2 * PI * freq * t / len
    a << (sine ? Math.sin(ang) : Math.cos(ang))
    t += 1
  end
  a
end

n    = 16
cos3 = GPU::SFloat.cast(tone(n, 3))
spec = cos3.rfft
assert_near("rfft(cos) bin 3 real", n / 2.0, spec.to_a[3][0], 1e-2)
assert_near("rfft(cos) bin 3 imag", 0.0,     spec.to_a[3][1], 1e-2)
assert_near("rfft(cos) bin 2 real", 0.0,     spec.to_a[2][0], 1e-2)

# sin has its energy in the imaginary part, negative for the forward transform.
sin5 = GPU::SFloat.cast(tone(n, 5, true))
assert_near("rfft(sin) bin 5 imag", -(n / 2.0), sin5.rfft.to_a[5][1], 1e-2)

# magnitude / power default to the non-redundant half (n/2 bins).
mag = cos3.rfft.magnitude
assert_near("magnitude size",  n / 2, mag.size)
assert_near("magnitude bin 3", n / 2.0, mag.to_a[3], 1e-2)
assert_near("magnitude bin 2", 0.0,     mag.to_a[2], 1e-2)
assert_near("power bin 3",     (n / 2.0) * (n / 2.0), cos3.power_spectrum.to_a[3], 1e-1)
assert_near("magnitude count arg size", 4, cos3.rfft.magnitude(4).size)

# Multi-tone: peaks must land on the expected bins and nowhere else.
n2  = 64
lo  = tone(n2, 5)
hi  = tone(n2, 11)
mix = []
lo.each_index { |i| mix << lo[i] + hi[i] }

# Each tone contributes n2/2 = 32 in magnitude, so 1024 in power; every other
# bin is ~0. Anything above (n2/4)^2 = 256 is a peak.
power = GPU::SFloat.cast(mix).power_spectrum.to_a
peaks = []
power.each_index { |i| peaks << i if power[i] > (n2 / 4.0) * (n2 / 4.0) }
assert_ary("multi-tone peak bins", [5, 11], peaks)

# A length that is not a power of two must be rejected.
assert_raise("rfft(non power of two) -> ArgumentError", ArgumentError) do
  GPU::SFloat.new(6).seq.rfft
end
assert_raise("SComplex#sum -> NoMethodError", NoMethodError) do
  GPU::SFloat.new(8).seq.rfft.sum
end

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

# Element counts are uint32_t on the GPU side; a larger request must be
# refused rather than wrapped around into a different length.
assert_raise("size past uint32 -> ArgumentError", ArgumentError) do
  GPU::SFloat.new(2**32 + 10)
end

# One dispatch covers 256 elements per workgroup, bounded by the device's
# maxComputeWorkGroupCount. Past that the result is up to the driver, so it is
# rejected instead. Sized off the device, and skipped where the limit is high
# enough that probing it would mean a multi-gigabyte allocation.
max_elems = GPU.info[:max_workgroups] * 256
if max_elems <= 64_000_000
  assert_raise("array past one dispatch -> ArgumentError", ArgumentError) do
    GPU::SFloat.new(max_elems + 256).sum
  end
else
  puts "SKIP array past one dispatch (device allows #{max_elems} elements per dispatch)"
end

def assert_true(label, cond, detail = "false")
  cond ? ok(label) : ng(label, detail)
end

# ---- deferred submission ----
#
# Dispatches are recorded and submitted together when a result is read.
# These check the batching itself, the sync points, and the two things that
# could silently go wrong: a host write racing a queued read, and a buffer
# freed by the GC while a queued dispatch still references it.

GPU.sync_mode = :deferred
GPU.sync   # earlier tests may have left work queued; start the count from zero
assert_true("default sync mode is :deferred", GPU.sync_mode == :deferred, GPU.sync_mode.inspect)

dv = GPU::SFloat.new(1024).seq
chain = dv * 2 + 1 - 3 + 4
assert_true("four operators are recorded, not submitted", GPU.pending == 4, "pending = #{GPU.pending}")
assert_ary("deferred chain reads back correctly",
           [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], chain.head(8))
assert_true("reading a result flushes the batch", GPU.pending == 0, "pending = #{GPU.pending}")

dr = dv * 2
dv.fill(0.0)          # host write to the input of a queued dispatch
assert_ary("a host write flushes the queued reader first", [0.0, 2.0, 4.0, 6.0], dr.head(4))
dv.seq

fresh = GPU::SFloat.new(4)
dv * 3                # queued
fresh.fill(1.0)       # a buffer the batch never touched must not force a flush
assert_true("writing an untouched buffer keeps the batch pending", GPU.pending == 1, "pending = #{GPU.pending}")
GPU.sync

long = dv
300.times { long = long * 1.0 }   # more dispatches than the descriptor pool holds
assert_ary("a batch longer than the descriptor pool still runs", [0.0, 1.0, 2.0, 3.0], long.head(4))

gy = nil
40.times { gy = (dv + 1) * 2 - 2 }   # (dv + 1) and its double are garbage on the next iteration
GC.start
assert_ary("garbage intermediates outlive the batch that reads them", [0.0, 2.0, 4.0, 6.0], gy.head(4))

ds = GPU::SFloat.cast([1.0, 2.0, 3.0, 4.0])
assert_near("sum flushes and reads the partials", 20.0, (ds * 2).sum)

dsig = GPU::SFloat.new(16).seq
spec_deferred = dsig.power_spectrum.to_a
GPU.sync_mode = :eager
spec_eager = dsig.power_spectrum.to_a
assert_ary("rfft agrees between deferred and eager", spec_eager, spec_deferred)
dsig * 2
assert_true("eager mode submits every dispatch", GPU.pending == 0, "pending = #{GPU.pending}")
assert_true("sync_mode reports :eager", GPU.sync_mode == :eager, GPU.sync_mode.inspect)
GPU.sync_mode = :deferred
GPU.sync
assert_true("GPU.sync with nothing pending is a no-op", GPU.pending == 0)
assert_raise("unknown sync mode -> ArgumentError", ArgumentError) { GPU.sync_mode = :later }

# ---- summary ----
puts
puts "#{$pass + $fail} tests, #{$pass} passed, #{$fail} failed"
puts($fail > 0 ? "SOME TESTS FAILED" : "ALL TESTS PASSED")
