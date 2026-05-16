# Additional Tools

Drop custom `.ts` or `.js` tools here.

Rules:

- Include a top-of-file docstring or matching `<tool-name>.md`.
- Read config from the repository `.env` file when needed.
- Do not assume network access.
- Do not store secrets in generated output.

Agents should scan this folder before claiming a tool does not exist.
