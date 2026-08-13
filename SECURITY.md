# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability.

Use GitHub private vulnerability reporting or a repository security advisory.

Include:

- affected component
- reproduction steps
- impact
- suggested remediation, when known

Remove tokens, credentials, private datasets, and customer data from logs and screenshots.

## Model artifact security

This project uses `joblib` to persist a scikit-learn pipeline.

`joblib`/pickle-style model artifacts can execute code when loaded. Only load artifacts from trusted sources.

The service verifies `model.joblib` SHA-256 against `model_manifest.json` before loading.

This protects against accidental mismatch or a changed model file relative to the manifest.

It does **not** provide publisher authenticity because an attacker who can replace both the model and manifest can create a new matching hash.

Stronger production distribution may require:

- trusted container registry permissions
- immutable release processes
- artifact attestations or signed provenance
- controlled deployment identities
- restricted artifact write access

## Secrets

Never commit:

- `.env`
- access tokens
- registry credentials
- private CRM exports
- production database credentials

If a secret is committed, rotate it. Deleting it from the latest commit does not remove it from Git history.

## Data security

The committed dataset is synthetic.

Do not replace it with a real CRM export and commit that export to a public repository.

Real lead data may contain personal or commercially sensitive information and requires appropriate access controls, retention rules, and privacy review.

## Supported code

Security fixes target the current `main` branch.
