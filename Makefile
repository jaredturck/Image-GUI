.PHONY: macos-dmg macos-check macos-lock

macos-dmg:
	./scripts/build_macos.sh

macos-check:
	./scripts/build_macos.sh --check

macos-lock:
	MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile requirements.txt requirements-macos.txt \
		--constraints requirements-macos-constraints.txt \
		--python-platform aarch64-apple-darwin \
		--python-version 3.13 \
		--no-build \
		--generate-hashes \
		--no-annotate \
		--custom-compile-command 'make macos-lock' \
		--output-file requirements-lock-macos.txt
