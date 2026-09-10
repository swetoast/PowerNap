# Security policy

## Supported versions

PowerNap is currently a pre-release project. Security fixes are applied to the latest development version.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could enable privilege escalation, arbitrary file writes, unsafe hardware limits, command execution, or disclosure of sensitive local data.

Use the repository owner's private security reporting channel after the GitHub repository is created. Add the final contact or GitHub private vulnerability-reporting link here before public release.

Include:

- Affected version
- Operating system and kernel
- Relevant CPU, GPU, and driver
- Reproduction steps
- Expected and observed behavior
- Potential impact

PowerNap does not execute configured shell fragments. Any new adapter must continue to use fixed argument structures and validated paths and values.
