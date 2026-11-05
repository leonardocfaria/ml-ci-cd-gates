# ml-ci-cd-gates

## Goal
Build "CI/CD for ML with quality gates". A repo template where every PR trains a small model in CI, compares it with the current champion on a fixed eval set, fails the build on a metric regression or a data-schema change, and publishes an evaluation report artifact. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [x] CI workflow trains and evaluates
- [x] regression gate fails on a deliberately bad change (demonstrated in a test)
- [x] report artifact produced
- [x] Tests pass with one command; CI workflow runs them
- [x] README with setup, usage, architecture and results
- [x] Self code review done
- [x] Draft PR open

