# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

### How to Report

1. **Do not** create a public issue for security vulnerabilities
2. **Email** the maintainers directly (if available) or use GitHub's private vulnerability reporting
3. **Include** as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment** within 48 hours
- **Initial assessment** within 1 week
- **Regular updates** on the progress
- **Credit** in the security advisory (if desired)

## Security Considerations

### API Keys and Environment Variables

- **Never commit** API keys or sensitive data to the repository
- **Use environment variables** for all sensitive configuration
- **Rotate API keys** regularly
- **Review** the `.env.example` file for proper configuration

### Document Processing

- **Validate** all uploaded documents before processing
- **Sanitize** file paths and names
- **Limit** file sizes and types as appropriate
- **Consider** running OCR processing in sandboxed environments

### LLM Integration

- **Validate** all inputs sent to LLM providers
- **Monitor** API usage and costs
- **Implement** rate limiting for API calls
- **Review** conversation logs for sensitive information

### Data Storage

- **Secure** vector stores and document metadata
- **Consider** encryption for sensitive document content
- **Implement** proper access controls
- **Regular** cleanup of temporary files

## Security Best Practices

### For Users

1. **Keep dependencies updated** using `pip install -r requirements.txt --upgrade`
2. **Use strong API keys** and rotate them regularly
3. **Monitor** your API usage for unusual activity
4. **Secure** your `.env` file with appropriate permissions
5. **Review** documents before processing for sensitive content

### For Contributors

1. **Follow** secure coding practices
2. **Validate** all inputs and outputs
3. **Use** parameterized queries if database interactions are added
4. **Implement** proper error handling that doesn't leak information
5. **Test** for common vulnerabilities (injection, path traversal, etc.)

## Dependencies Security

We regularly review our dependencies for known vulnerabilities:

- **Monitor** security advisories for all dependencies
- **Update** dependencies promptly when security issues are discovered
- **Use** tools like `safety` to scan for known vulnerabilities:
  ```bash
  pip install safety
  safety check
  ```

## Disclosure Policy

- **Responsible disclosure** is appreciated and will be credited
- **Coordinated disclosure** timeline will be agreed upon case-by-case
- **Public disclosure** will only happen after fixes are available
- **Security advisories** will be published for significant vulnerabilities

## Contact

For security-related questions or concerns:
- Use GitHub's security reporting features
- Check existing security discussions before creating new ones
- For non-sensitive security questions, use GitHub issues with the `security` label

Thank you for helping keep this project secure!