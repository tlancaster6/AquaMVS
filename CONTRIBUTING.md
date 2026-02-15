# Contributing to AquaCal

Thank you for your interest in contributing to AquaCal. This guide will help you get started with development.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tlancaster6/AquaCal.git
   cd AquaCal
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests to verify your setup:
   ```bash
   python -m pytest tests/
   ```

## Code Style

- **Formatter**: Ruff (run `ruff format src/ tests/` before committing)
- **Linter**: Ruff (run `ruff check src/ tests/` to check for issues)
- **Docstrings**: Google style
- **Type hints**: Use `numpy.typing.NDArray` with shape information in docstrings
- **Imports ordering**:
  1. Standard library imports
  2. Third-party imports (numpy, scipy, opencv, etc.)
  3. Local imports (from aquacal)

  Separate each group with a blank line.

This project uses pre-commit hooks. Install with `pre-commit install` to automatically check formatting and linting before each commit.

## Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Skip slow optimization tests:
```bash
python -m pytest tests/ -m "not slow"
```

Run a single test file:
```bash
python -m pytest tests/unit/test_camera.py -v
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and add tests
4. Ensure all tests pass
5. Format and lint your code with ruff (`ruff format src/ tests/ && ruff check src/ tests/`)
6. Commit your changes with a clear message (see Commit Messages below)
7. Push to your fork and submit a pull request

### Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automated versioning and changelog generation. Format your commit messages as:

```
<type>(<scope>): <description>

[optional body]
```

**Types:**
- `feat:` A new feature (triggers minor version bump, e.g., 1.0.0 → 1.1.0)
- `fix:` A bug fix (triggers patch version bump, e.g., 1.0.0 → 1.0.1)
- `docs:` Documentation changes only
- `test:` Adding or updating tests
- `refactor:` Code changes that neither fix bugs nor add features
- `chore:` Maintenance tasks (dependencies, tooling, etc.)

**Scope** (optional): The area of the codebase affected, e.g., `calibration`, `cli`, `core`

**Examples:**
- `feat(calibration): add support for tilted water surfaces`
- `fix(projection): correct Newton-Raphson convergence criterion`
- `docs: update README installation instructions`
- `test(synthetic): add test for multi-camera pose graph`

## Deprecation Policy

When deprecating functionality in AquaCal:

1. **Add a deprecation warning** using `warnings.warn()` with `DeprecationWarning`:
   ```python
   import warnings

   def old_function():
       warnings.warn(
           "old_function() is deprecated as of version 1.2.0 and will be removed in version 1.4.0. "
           "Use new_function() instead.",
           DeprecationWarning,
           stacklevel=2
       )
       # existing implementation
   ```

2. **Document in CHANGELOG.md** under the "Deprecated" category:
   ```markdown
   ### Deprecated
   - `old_function()` - Use `new_function()` instead (will be removed in 1.4.0)
   ```

3. **Maintain for at least 2 minor versions** before removal.

4. **Document the replacement** in the function's docstring:
   ```python
   def old_function():
       """Original description.

       .. deprecated:: 1.2.0
           Use :func:`new_function` instead. Will be removed in version 1.4.0.
       """
   ```

Always include `stacklevel=2` in warnings to show the caller's location, not the warning itself.
