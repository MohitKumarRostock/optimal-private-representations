# Maintaining the research repository

## Scope of changes

Treat a change to mathematical definitions, privacy accounting, data generation, random-number sequencing, a fitted algorithm, or the evaluation metric as a scientific change. Explain the previous behavior, the revised behavior, and the affected experiments. A documentation or plotting change can preserve numerical results, but its scope should still be explicit.

The public capped learner and the uncapped experiment implementation have different roles. The former exposes the bounded output-count optimization. The latter reproduces recorded fits and timing measurements. Keep their identities explicit in metadata and reports.

## Validation before publication

Run the focused numerical tests and package smoke workflow after changes to code or imports. Validate reference hashes and record-key uniqueness after changes to distributed data. Recompute an affected summary or diagnostic when its implementation changes. A full simulation is appropriate when the protocol or fitting algorithm changes; it is unnecessary for a spelling correction or a plot-label adjustment.

Continuous integration exercises installation, focused tests, and reference verification. Full simulations and serial benchmarks are explicit research workflows, with their own output directories and run metadata.

## Reference data and generated outputs

Keep the reference measurements immutable within a published version. Write regenerated artifacts to `outputs/`, which is excluded from version control. When reference data must change, document the reason, preserve the preceding revision, update the data dictionary and checksums, and assign a new version.

A protocol should identify the configuration grid, methods, number of independent replicates, random-number construction, evaluation rule, uncertainty calculation, and source/environment provenance. Preserve historical source hashes even when portable packaging changes import statements; record the migration separately.

## Versioning and citation

Use a descriptive branch for development and an immutable commit to identify a scientific snapshot. A release tag may provide a convenient human-readable version, but preserve the commit in the experiment record and citation. Do not claim an archival DOI until one has actually been issued.

The citation record describes this software and its experimental artifacts independently of any particular paper. A later manuscript can cite the repository version. Repository documentation should remain intelligible without that manuscript or its theorem numbering.

## Reporting defects

A useful issue report identifies the source commit, command, environment, observed behavior, expected behavior, and a minimal synthetic example. State whether the issue concerns a mathematical claim, the floating-point implementation, the experiment protocol, or presentation. Do not include confidential records or private randomness in a public report.
