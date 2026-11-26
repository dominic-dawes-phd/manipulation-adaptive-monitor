"""Package version, read from the installed metadata of pyproject.toml."""

from importlib.metadata import version

AMON_VERSION: str = version("amon")
