# Orbit Security and Compliance

Lumen Labs treats customer data security as a core product feature. This page summarizes
the controls that protect data stored in Orbit.

## Certifications

Lumen Labs holds a SOC 2 Type II report, audited annually by an independent firm, and has
been ISO 27001 certified since 2023. An independent third party performs a penetration test
of Orbit every year.

## Encryption

All customer data is encrypted at rest with AES-256. Data in transit is protected with
TLS 1.2 or higher.

## Data residency

Orbit stores customer data in two regions: the United States (Oregon) and the European
Union (Frankfurt). EU data residency is available on the Business and Enterprise plans.

## Single sign-on and provisioning

Orbit supports single sign-on through SAML 2.0 and OpenID Connect (OIDC) on the Business and
Enterprise plans. SCIM user provisioning is available on the Enterprise plan only.

## Two-factor authentication

Users can enable two-factor authentication with any TOTP authenticator app. Admins can make
two-factor authentication mandatory for their workspace on all paid plans.

## Data retention

Deleted tasks and projects stay in the trash for 30 days before they are permanently
removed. After an account is cancelled, its data is deleted within 90 days. Backups are
retained for 35 days.

## Vulnerability disclosure

Security researchers can report vulnerabilities to security@lumenlabs.example. The Lumen
Labs bug bounty program pays rewards of up to $5,000 for valid reports.
