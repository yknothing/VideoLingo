# Contributing to VideoLingo

Thank you for your interest in contributing to VideoLingo! This document provides guidelines for contributing to this project.

## Repository Information

**Primary Repository**: `yknothing/VideoLingo`

This is the main development repository for VideoLingo. All contributions should be directed here.

**Note**: The original repository `Huanshere/VideoLingo` is maintained separately. This fork (`yknothing/VideoLingo`) is the active development branch.

## Getting Started

### 1. Fork and Clone

```bash
# Clone the repository
git clone https://github.com/yknothing/VideoLingo.git
cd VideoLingo

# Add your fork as a remote (if you forked it)
git remote add myfork https://github.com/YOUR_USERNAME/VideoLingo.git
```

### 2. Set Up Development Environment

```bash
# Install dependencies
python install.py

# Or use Docker for development
./deploy.sh
```

### 3. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

## Making Changes

### Code Standards

- Use English for all comments and print statements
- Follow the existing code style (see `.cursorrules`)
- Use large comment blocks with `# ------------` for major sections
- Minimize internal function comments
- Avoid complex function annotations

### Testing

Before submitting your changes:

1. Test your changes locally
2. Run the application: `streamlit run st.py`
3. Verify all functionality works as expected
4. Check for any errors or warnings

## Submitting Changes

### 1. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add your feature description"

# Or for bug fixes
git commit -m "fix: fix issue description"
```

### Commit Message Format

Use conventional commit messages:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 2. Push to Your Branch

```bash
# Push to the yknothing/VideoLingo repository
git push -u origin feature/your-feature-name
```

### 3. Create a Pull Request

1. Go to https://github.com/yknothing/VideoLingo
2. Click "New Pull Request"
3. Select your feature branch
4. Ensure the base repository is `yknothing/VideoLingo` and base branch is `main`
5. Fill out the PR template with:
   - Clear description of changes
   - Testing details
   - Related issues (if any)
6. Submit the PR

## Pull Request Guidelines

### Target Repository

**IMPORTANT**: All Pull Requests must target `yknothing/VideoLingo`, not the upstream repository.

- **Base repository**: `yknothing/VideoLingo`
- **Base branch**: `main` (or as specified)
- **Compare branch**: Your feature/fix branch

### PR Checklist

Before submitting, ensure:

- [ ] Code follows project standards
- [ ] All tests pass locally
- [ ] Documentation updated (if needed)
- [ ] No sensitive information (API keys, passwords) committed
- [ ] Commit messages are clear and descriptive
- [ ] PR targets `yknothing/VideoLingo` repository

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited

## Development Workflow

### Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes and commit
git add .
git commit -m "feat: implement new feature"

# 3. Push to origin
git push -u origin feature/my-feature

# 4. Create PR on GitHub targeting yknothing/VideoLingo
```

### Bug Fixes

```bash
# 1. Create fix branch
git checkout -b fix/bug-description

# 2. Fix the issue and commit
git add .
git commit -m "fix: resolve bug description"

# 3. Push and create PR
git push -u origin fix/bug-description
```

## Getting Help

- Check existing issues and PRs first
- Open a new issue for bugs or feature requests
- Join discussions in existing issues
- Review documentation in `docs/` directory

## Important Notes

1. **Repository Target**: Always ensure your PRs target `yknothing/VideoLingo`
2. **Code Quality**: Maintain high code quality and follow existing patterns
3. **Documentation**: Update documentation for significant changes
4. **Testing**: Test thoroughly before submitting
5. **Communication**: Be responsive to review comments

## Resources

- [CLAUDE.md](CLAUDE.md) - Development guidelines for Claude Code
- [README.md](README.md) - Project overview and setup
- [Architecture Documentation](ARCHITECTURE.md) - System architecture details
- [Docker Deployment](docs/pages/docs/docker.zh-CN.md) - Docker setup guide

## License

By contributing to VideoLingo, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to VideoLingo! Your efforts help make this project better for everyone.
