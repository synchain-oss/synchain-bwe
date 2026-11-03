# Security Policy

## Supported versions

Only the latest release line receives security fixes; please upgrade to it and reproduce there first.

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub Private Vulnerability Reporting: repository page →
Security → Report a vulnerability.

If that channel is unavailable, email **contact@synchain.ca** with `[security]` in the subject.

Reports in English or Chinese are both welcome. Please include the synchain-bwe commit or version, your operating
system and Python version, the command you ran and the steps to reproduce. **Do not send voice recordings** of anyone
who has not agreed to it, and do not include credentials.

## Response targets

- First response: within 3 working days
- Fix or mitigation: 14 days for high severity, 30 days for medium severity
- Disclosure: a public advisory (GHSA) is published 7 days after the fix ships

## Scope

- In scope: loading or running anything other than the pinned model files (the SHA-256 check, the config check, the
  weights-only load); audio or any other data leaving the machine; the tool overwriting its inputs or writing outside
  the output paths it was given; supply-chain issues in the dependencies as pinned here and in the CI workflow.
- Out of scope: the quality of the model's guesses; bugs in LavaSR, Vocos, PyTorch or other dependencies themselves
  (report those upstream); scenarios that already require arbitrary code execution on your machine.
