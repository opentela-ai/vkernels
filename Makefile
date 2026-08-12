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
PYTHON ?= python3
ARGS ?= list

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
	python3 meta/scripts/coverage.py --build-dir build/coverage --source-dir src/c --min 100

clean:
	rm -rf build/$(P)

fmt:
	@if [ "$$CI" = "1" ]; then EXTRA=--Werror; fi; \
	clang-format --dry-run $$EXTRA -i \
	  $$(git ls-files '*.cpp' '*.hpp' '*.h' '*.cu' '*.cuh')

tidy:
	@command -v clang-tidy >/dev/null 2>&1 || { echo "clang-tidy not installed"; exit 1; }; \
	clang-tidy -p build/$(P) $$(git ls-files '*.cpp' '*.hpp')

# Python CLI (src/python/) — managed by uv (see pyproject.toml):
#   make py-sync         uv sync -> .venv/ + editable install + uv.lock
#   make vkl ARGS=list   run `vkl list` (default ARGS=list)
#   make py-test         run the CLI's unittest suite (or: uv run pytest)
#   make py-lock         refresh uv.lock
vkl:
	uv run python -m vkernels.cli $(ARGS)

py-test:
	uv run python -m unittest discover -s tests/python -v

py-sync:
	uv sync

py-lock:
	uv lock

# Rust bindings (src/rust/) — cargo builds the C++ library itself:
#   make rust-test       cargo test (host path by default)
#   VKERNELS_RUST_CUDA=ON make rust-test   also compile the CUDA kernels
rust-test:
	cargo test --manifest-path src/rust/Cargo.toml

.PHONY: configure build test coverage clean fmt tidy vkl py-test py-sync py-lock rust-test
