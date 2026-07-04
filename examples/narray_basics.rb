# mruby-gpu-narray -- basics
#
# The same code you would prototype with, running on the GPU.
# On a Raspberry Pi 5 this executes on the VideoCore VII.

puts "device : #{GPU.info[:device]}"
puts "backend: #{GPU.info[:backend]} (Vulkan API #{GPU.info[:api_version]})"
puts

# Numo-like construction. Data lives on the GPU from the start.
a = GPU::SFloat.new(8).seq        # 0,1,2,...,7   (like Numo::SFloat.new(8).seq)
puts "a          = #{a.inspect}"

# Arithmetic runs as GPU compute. Nothing is copied to the host yet.
b = a * 2 + 1                     # 1,3,5,...,15
puts "a * 2 + 1  = #{b.inspect}"

# Element-wise between two arrays.
c = GPU::SFloat[10, 20, 30, 40, 50, 60, 70, 80]
puts "c          = #{c.inspect}"
puts "b + c      = #{(b + c).inspect}"
puts "c / 10     = #{(c / 10).inspect}"
puts "-c         = #{(-c).inspect}"

# Reductions.
puts
puts "b.sum      = #{b.sum}"        # 1+3+...+15 = 64
puts "b.mean     = #{b.mean}"       # 8.0

# .to_a is the only place data comes back to the host (an ordinary mruby Array).
puts
puts "b.to_a     = #{b.to_a.inspect}"

# A bigger reduction to show it is really parallel work, not a Ruby loop.
big = GPU::SFloat.new(1_000_000).fill(1.0)
puts
puts "1M ones, sum = #{big.sum}"    # 1000000.0
