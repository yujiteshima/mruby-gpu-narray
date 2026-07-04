# Ruby-side sugar for GPU::NArray / GPU::SFloat.
#
# The heavy lifting (allocation, transfer, arithmetic, reduction) is in C.
# Here we add the conveniences that are easier and clearer to express in Ruby:
# shape/ndim, mean, a Numo-like #inspect, and the zeros/ones/[] constructors.
#
# These classes are first defined in C (src/gpu_narray.c); the definitions
# below reopen them.

module GPU
  class NArray
    # 1-D for now. shape/ndim exist so code reads like Numo and so that the
    # 2-D extension later only has to change what these return.
    def ndim
      1
    end

    def shape
      [size]
    end

    def mean
      sum / size
    end

    # Numo-like preview, e.g.:
    #   GPU::SFloat(shape=[1024]) [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ..., (1024 total)]
    def inspect
      n = size
      shown = n < 6 ? n : 6
      body = head(shown).join(", ")
      body += ", ..., (#{n} total)" if n > shown
      "#{self.class}(shape=#{shape.inspect}) [#{body}]"
    end
    alias to_s inspect
  end

  class SFloat
    # Numo::SFloat.zeros(n) / .ones(n) equivalents.
    def self.zeros(n)
      new(n).fill(0.0)
    end

    def self.ones(n)
      new(n).fill(1.0)
    end

    # GPU::SFloat[1, 2, 3] == GPU::SFloat.cast([1, 2, 3])
    def self.[](*elements)
      cast(elements)
    end
  end
end
