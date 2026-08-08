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

    # Convenience for the common spectral path: rfft then power. The complex
    # spectrum stays on the GPU and is dropped once this returns.
    def power_spectrum(count = nil)
      count ? rfft.power_spectrum(count) : rfft.power_spectrum
    end
  end

  # A complex spectrum, as produced by GPU::SFloat#rfft.
  #
  # The buffer is interleaved (re, im), so a length-k spectrum occupies 2k
  # floats. NArray's inherited methods count floats, so the ones that should
  # count *points* are adjusted here.
  class SComplex
    def size
      super / 2
    end
    alias length size

    def shape
      [size]
    end

    # [[re, im], ...] -- the inherited to_a returns the interleaved floats.
    def to_a
      interleaved_to_pairs(super)
    end

    def head(k)
      interleaved_to_pairs(super(2 * k))
    end

    def inspect
      n = size
      shown = n < 3 ? n : 3
      body = head(shown).map { |c| "(#{c[0]}#{c[1] < 0 ? '-' : '+'}#{c[1].abs}i)" }.join(", ")
      body += ", ..., (#{n} total)" if n > shown
      "#{self.class}(shape=#{shape.inspect}) [#{body}]"
    end
    alias to_s inspect

    # Summing interleaved re/im components would be meaningless, so the
    # inherited reductions are withdrawn rather than left to mislead.
    # NoMethodError (not NotImplementedError) so that an ordinary
    # `rescue => e` catches it -- NotImplementedError is a ScriptError.
    def sum
      raise NoMethodError,
            "#{self.class}#sum is not defined; use #magnitude or #power_spectrum"
    end

    def mean
      raise NoMethodError,
            "#{self.class}#mean is not defined; use #magnitude or #power_spectrum"
    end

    private

    def interleaved_to_pairs(flat)
      out = []
      i = 0
      while i < flat.size
        out << [flat[i], flat[i + 1]]
        i += 2
      end
      out
    end
  end
end
