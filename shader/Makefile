GLSLC ?= glslangValidator

SHADERS = add.comp sub.comp mul.comp div.comp scale.comp adds.comp sum.comp \
          fft_bitrev.comp fft_stage.comp cmag.comp
SPIRV = $(SHADERS:.comp=.spv)

all: $(SPIRV)

%.spv: %.comp
	$(GLSLC) -V $< -o $@

clean:
	rm -f $(SPIRV)

.PHONY: all clean
