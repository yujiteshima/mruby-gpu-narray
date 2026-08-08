# mruby-gpu-narray -- shader loading failures.
#
# Kept out of narray_test.rb because the GPU context is a process-wide
# singleton: pointing GPU.init at a bad shader directory here would leave every
# later test without pipelines. Run this file on its own:
#
#   mruby test/shader_error_test.rb <shader-dir>
#
# With no argument it checks the missing-file case only. Pass a directory of
# unreadable-but-present .spv files to also check the driver-rejects case; the
# harness that generates those lives in the CI workflow.

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

# Asserts that the block raises, and that the message mentions `expect` but not
# `avoid` -- the point of these cases is *which* fix the message sends you to.
def assert_message(label, expect, avoid)
  begin
    yield
  rescue => e
    if !e.message.include?(expect)
      return ng(label, "expected the message to mention #{expect.inspect}, got #{e.message.inspect}")
    end
    if avoid && e.message.include?(avoid)
      return ng(label, "message should not mention #{avoid.inspect}: #{e.message.inspect}")
    end
    return ok(label)
  end
  ng(label, "nothing was raised")
end

bad_dir = ARGV[0]

if bad_dir
  # The .spv files exist and load, and the driver refuses them. Reporting this
  # as "not compiled" would send the user to re-run a build step that already
  # ran, so the message has to say the opposite.
  GPU.init(bad_dir)
  assert_message("driver-rejected shader names the driver", "driver rejected", "make -C shader") do
    GPU::SFloat[1, 2] + GPU::SFloat[3, 4]
  end
else
  # No .spv files at all: here "run make" *is* the right advice.
  GPU.init("/nonexistent/mruby-gpu-narray-shaders")
  assert_message("missing shader tells you to build it", "make -C shader", nil) do
    GPU::SFloat[1, 2] + GPU::SFloat[3, 4]
  end
end

puts
puts "#{$pass + $fail} tests, #{$pass} passed, #{$fail} failed"
puts($fail > 0 ? "SOME TESTS FAILED" : "ALL TESTS PASSED")
