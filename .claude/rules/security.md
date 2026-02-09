# Security Rules (Always Active)

- NEVER write API keys, secrets, passwords, or private keys into any file
- Credentials ONLY via environment variables or `.env` files (gitignored)
- NEVER log full API keys — mask all but last 4 characters
- NEVER commit `.env`, `*.pem`, `*.key`, or credential files
- Validate all exchange API responses — don't trust input
- Use HMAC-SHA256 for exchange request signing (per exchange docs)
- Rate limit all outbound API calls — never exceed exchange limits
- Sanitize all user inputs to dashboard/config API
- Dependency scanning: check for known vulns before adding packages
- Pin all dependencies with exact versions in requirements.txt / Cargo.lock
