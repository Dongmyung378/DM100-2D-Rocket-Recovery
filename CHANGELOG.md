# Release notes

[한국어](CHANGELOG.ko.md)

## Runtime patch: v1.1.0-runtime.1

Automatic project-local Python setup on Windows and Linux, complete application
runtime constraints, dependency version repair and a diagnostic command.
Ship the unmodified Korean font and its license in source archives and wheels,
and use the packaged FFmpeg encoder. Fresh managed environments each passed
790 tests and 11 subtests, including Korean rendering with missing-glyph warnings
treated as failures. Real UI and H.264 smoke checks, offline reuse, Windows
dependency drift and folder relocation repair passed. Frozen numerical source
and acceptance results remain unchanged. See [runtime checks](docs/runtime.md).

## 1.1.0

Adds a causal attitude and gyro-bias estimator, bounded synthetic angular sensor
errors and scheduled packet omissions. Default ideal angular channels retain
the existing observation path.

The public study defines 96 new coupled evaluation conditions, 48 paired sensor
conditions, feature ablations, full-flight step-size comparisons and one-factor
parameter sensitivity. Implementation and inputs are frozen before final
evaluation. Original acceptance limits remain unchanged.

Curated per-flight evidence, standard-library evidence validators and numerical
reports make the results reviewable without historical local files. An
independent SciPy integration reference checks three smooth plant segments.
Fresh Windows and Linux installation checks, tested dependency constraints and
English/Korean documentation improve reproduction.

See [recorded results](docs/results.md) for measured outcomes and limitations.
This release remains a theoretical 2D concept simulation.

## 1.0

Local bilingual launch and recovery simulator, fixed 100 t design, original
286-condition regression manifest, conservation and first-contact assessment,
telemetry plots and optional trajectory replay. Windows launcher repairs
environments after folder relocation.
