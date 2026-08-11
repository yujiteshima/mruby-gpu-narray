# mruby-gpu-narray -- spectral analysis
#
# Synthesises a two-tone signal, transforms it on the GPU, and reports the
# bins the energy lands in. On a Raspberry Pi 5 this runs on the VideoCore VII.

PI = 3.141592653589793

N         = 1024        # transform length, a power of two
SAMPLE_HZ = 1024.0      # one bin == 1 Hz, which keeps the output readable
TONES     = [50, 200]   # Hz
AMPS      = [1.0, 0.5]

puts "device : #{GPU.info[:device]}"
puts "backend: #{GPU.info[:backend]} (Vulkan API #{GPU.info[:api_version]})"
puts

# ---- synthesise the signal on the host ----
samples = []
t = 0
while t < N
  v = 0.0
  TONES.each_index { |i| v += AMPS[i] * Math.cos(2 * PI * TONES[i] * t / SAMPLE_HZ) }
  samples << v
  t += 1
end

signal = GPU::SFloat.cast(samples)
puts "signal   = #{signal.inspect}"

# ---- transform: 1 + log2(N) compute dispatches, nothing copied to the host ----
spectrum = signal.rfft
puts "spectrum = #{spectrum.inspect}"

power = spectrum.power_spectrum     # N / 2 bins, still in GPU memory
puts "power    = #{power.inspect}"
puts

# ---- .to_a is the first host copy ----
bins = power.to_a
peak = 0.0
bins.each_index { |i| peak = bins[i] if bins[i] > peak }

puts "bins above 1% of the peak:"
puts "  bin   freq(Hz)      power"
bins.each_index do |i|
  next if bins[i] < peak * 0.01
  freq = i * SAMPLE_HZ / N
  puts "  #{i.to_s.rjust(4)}  #{freq.round(1).to_s.rjust(8)}  #{bins[i].round(1).to_s.rjust(10)}"
end
puts
puts "expected tones: #{TONES.inspect} Hz"
