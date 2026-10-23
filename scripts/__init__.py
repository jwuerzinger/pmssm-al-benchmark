"""Analysis, figure, campaign and experiment scripts.

Every script here is run by path, from any directory:

    python scripts/figures/plot_best_config.py --help

The first statement of each file is the repository bootstrap, which puts the
repository root on ``sys.path``. After it, imports are absolute only:
``from pmssm import ...`` for the library and
``from scripts.<folder>.<module> import ...`` for a sibling script. Nothing
else may touch ``sys.path``, and the import graph between scripts must stay
acyclic; ``tests/check_repo_layout.py`` enforces both. ``scripts/README.md``
says what each script produces.
"""
