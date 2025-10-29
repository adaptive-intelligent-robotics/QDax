# Installation

1. Install uv by following the instructions at `https://docs.astral.sh/uv/` or run:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install project dependencies with dev and examples extras:
   ```bash
   uv pip install .[dev,examples]
   ```

3. Run tests:
   ```bash
   uv run pytest -vv tests
   ```

4. Run style checks:
   ```bash
   uv run pre-commit run --all-files
   ```
