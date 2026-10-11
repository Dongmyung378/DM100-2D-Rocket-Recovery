# Runtime compatibility

[한국어](runtime.ko.md) | [Project](../README.md)

The runtime patch keeps the frozen numerical source and results unchanged while
automating installation and providing the application's font and video encoder.
The numerical revision remains 1.1.0. The patch release tag is `v1.1.0-runtime.1`.

## Supported automatic setup

- Windows 10/11 x86-64: run `launch-dm100.bat`.
- Linux x86-64 with glibc 2.28 or newer: run `sh launch-dm100.sh`.
- Use `--setup-only` to prepare dependencies, or `--doctor` to check runtime assets.

No GPU, system FFmpeg, system Korean font or administrator installation is needed.
An existing Python 3.12 is reused. Otherwise the launchers download uv 0.13.0,
verify the official archive and executable SHA256 values, then install Python
3.12.15 inside `.local/uv-python/` without changing shell profiles or the Windows
registry. First setup and environment repair need internet access and a writable
project folder. Linux's first download uses curl or wget and tar/sha256sum.

The Python launcher validates the isolated environment, current source path,
application version and every active runtime dependency against
`requirements/runtime-constraints.txt`. It also checks the bundled font hash and
performs a real H.264 encoding probe before accepting the environment. A broken
environment is preserved under `.local/environment-backups/` and rebuilt.

Python location and pip destination overrides are removed from child processes.
Network certificate/proxy settings remain available. An explicit `DM100_PYTHON`
must point to Python 3.12; otherwise setup stops with a useful explanation.
`DM100_USE_MANAGED_PYTHON=1` exercises the automatic-download path when needed.

## Fonts and video

The unmodified Noto Sans KR font and SIL OFL 1.1 license are shipped in both source
archives and wheels. Replay rendering registers this font directly, including in
minimal containers. It does not depend on the OS font list. The tested
`imageio-ffmpeg` package supplies the encoder; unrelated system or environment
overrides do not select a different encoder for the application.

## Verification

The [recorded compatibility checks](results/runtime-compatibility.json) cover
fresh managed Windows and Linux Python 3.12.15 environments, Korean and space
characters in checkout paths, local HTTP startup, Korean H.264 generation,
contaminated environment variables, offline reuse, dependency drift repair and
installed wheel assets. Missing-glyph warnings are treated as test failures.
The Windows and Linux environments run on the same host; this is environment
verification, not external review. An optional historical artifact test remains
skipped because those private old records are not distributed.

The frozen full-flight evidence still binds to the same 17 source files. This
patch does not reinterpret the historical environment in `reproduction.json`;
the new compatibility record describes the subsequent runtime verification.
