# Convenience wrapper around CMake presets. The presets are the source of
# truth; this Makefile just shortens common commands.
#
#   make configure [P=host]      cmake --preset <P>
#   make build      [P=host]     cmake --build --preset <P>
#   make test       [P=host]     ctest --preset <P>
#   make coverage                full coverage run + 100% gate
#   make clean      [P=host]     remove build/<P>
#   make fmt                     clang-format (check mode: CI=1 to enforce)
#   make tidy                    clang-tidy over changed sources (needs build/)
#
P ?= host

configure:
	cmake --preset $(P)

build:
	cmake --build --preset $(P)

test:
	cmake --build --preset $(P) --target test

coverage:
	cmake --preset coverage
	cmake --build --preset coverage
	ctest --preset coverage
	python3 scripts/coverage.py --build-dir build/coverage --source-dir csrc --min 100

clean:
	rm -rf build/$(P)

fmt:
	@if [ "$$CI" = "1" ]; then EXTRA=--Werror; fi; \
	clang-format --dry-run $$EXTRA -i \
	  $$(git ls-files '*.cpp' '*.hpp' '*.h' '*.cu' '*.cuh')

tidy:
	@command -v clang-tidy >/dev/null 2>&1 || { echo "clang-tidy not installed"; exit 1; }; \
	clang-tidy -p build/$(P) $$(git ls-files '*.cpp' '*.hpp')

.PHONY: configure build test coverage clean fmt tidy
