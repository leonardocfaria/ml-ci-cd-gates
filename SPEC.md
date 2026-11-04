# ml-ci-cd-gates

## Goal
Build "CI/CD for ML with quality gates". A repo template where every PR trains a small model in CI, compares it with the current champion on a fixed eval set, fails the build on a metric regression or a data-schema change, and publishes an evaluation report artifact. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [ ] CI workflow trains and evaluates
- [ ] regression gate fails on a deliberately bad change (demonstrated in a test)
- [ ] report artifact produced
- [ ] Tests pass with one command; CI workflow runs them
- [ ] README with setup, usage, architecture and results
- [ ] Self code review done
- [ ] Draft PR open

