# Release preparation

Before publishing a model or claiming a completed milestone:

- Document the actual implementation status, limitations, and reproducible checks.
- Publish measured evaluation with configuration, revisions, uncertainty, and contamination controls.
- Prepare a model card and data card covering the artifacts actually released.
- Verify source licenses, attribution, and the chosen code and weight licenses.
- Review files and Git history for secrets, private plans, raw data, and unapproved artifacts.
- Verify a clean clone can reproduce the documented setup and checks.

Weights and training data are separate releases; creating a public source repository does not authorize their upload.
