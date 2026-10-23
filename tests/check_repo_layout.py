#!/usr/bin/env python
"""Guard the repository layout rules that the scripts rely on.

Run from anywhere (``pixi run check`` or ``python tests/check_repo_layout.py``):

  * every tracked Python file compiles;
  * every repository path named in a tracked text file exists
    (``scripts/...``, ``slurm/...``, ``pipelines/...``, ``pmssm/...``,
    ``doc/...``, ``tests/...``), and with ``--paper DIR`` the same for the
    paper's ``.tex`` sources;
  * the only ``sys.path`` manipulation in ``scripts/``, ``pipelines/`` and
    ``tests/`` is the repository bootstrap (plus the two sanctioned submodule
    paths), with the ``parents[N]`` depth matching the file's location;
  * scripts import each other only as ``scripts.<folder>.<module>``;
  * the import graph between scripts and pipelines has no cycle (a cycle
    would execute a script run as ``__main__`` a second time under its
    package name);
  * no file names the old repository location;
  * with ``--help-all``, every command-line script answers ``--help``.

Exit status 1 when anything is found.
"""
# --- repository bootstrap: lets this file run from any directory --------------
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# ------------------------------------------------------------------------------

import argparse
import ast
import os
import py_compile
import re
import subprocess
import tempfile

TEXT_SUFFIXES = {".py", ".sh", ".md", ".toml", ".template", ".yaml", ".yml", ".cfg"}
PATH_RE = re.compile(
    r"(?<![\w/.\-])(scripts|slurm|pipelines|pmssm|doc|tests)/[\w./\-]*\w\.(py|sh|md|png|toml|template|yaml|conf)\b")
FORBIDDEN = ["pMSSM-trafo"]
EXEMPT_PATHS = {"slurm/cluster.conf"}      # gitignored by design, created per cluster
HISTORICAL = "doc/history/"                 # design records kept verbatim; not checked
SELF = Path("tests/check_repo_layout.py")   # this file quotes the patterns it forbids
SANCTIONED_INSERTS = {"sys.path.insert(0, str(_REPO_ROOT))",
                      "sys.path.insert(0, str(_GP_PIPELINE_ROOT))",
                      "sys.path.insert(0, str(_R3MG_SCRIPTS))"}


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True)
    return [Path(p) for p in out.stdout.split("\n") if p and Path(p) != SELF]


def check_compile(files, findings):
    for rel in files:
        if rel.suffix == ".py":
            try:
                py_compile.compile(str(_REPO_ROOT / rel), doraise=True)
            except py_compile.PyCompileError as e:
                findings.append(f"compile: {rel}: {e.msg.splitlines()[-1]}")


def check_paths(files, findings, extra_files=()):
    for rel in list(files) + list(extra_files):
        p = Path(rel)
        if p.suffix not in TEXT_SUFFIXES | {".tex"} or str(p).startswith(HISTORICAL):
            continue
        full = p if p.is_absolute() else _REPO_ROOT / p
        try:
            text = full.read_text(errors="replace")
        except OSError:
            continue
        for m in PATH_RE.finditer(text):
            target = m.group(0)
            if target not in EXEMPT_PATHS and not (_REPO_ROOT / target).exists():
                findings.append(f"path: {rel} names {target}, which does not exist")


def check_syspath(files, findings):
    for rel in files:
        if rel.suffix != ".py" or rel.parts[0] not in ("scripts", "pipelines", "tests"):
            continue
        depth = len(rel.parts) - 1          # pipelines/x.py -> 1, scripts/g/x.py -> 2
        text = (_REPO_ROOT / rel).read_text()
        for i, line in enumerate(text.split("\n"), 1):
            s = line.strip()
            if "sys.path" in s and ("insert" in s or "append" in s):
                if s not in SANCTIONED_INSERTS:
                    findings.append(f"sys.path: {rel}:{i}: {s}")
            if s.startswith("_REPO_ROOT ="):
                want = f"_REPO_ROOT = Path(__file__).resolve().parents[{depth}]"
                if s != want:
                    findings.append(f"bootstrap: {rel}:{i}: expected `{want}`, found `{s}`")


def module_names():
    mods = {}
    for rel in _REPO_ROOT.glob("scripts/*/*.py"):
        if rel.stem != "__init__":
            mods[rel.stem] = f"scripts.{rel.parent.name}.{rel.stem}"
    for rel in _REPO_ROOT.glob("pipelines/*.py"):
        if rel.stem != "__init__":
            mods[rel.stem] = f"pipelines.{rel.stem}"
    return mods


def imports_of(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def check_imports_and_cycles(files, findings):
    mods = module_names()
    qualified = set(mods.values())
    graph = {}
    for rel in files:
        if rel.suffix != ".py" or rel.parts[0] not in ("scripts", "pipelines"):
            continue
        me = mods.get(rel.stem)
        tree = ast.parse((_REPO_ROOT / rel).read_text(), filename=str(rel))
        edges = set()
        for name in imports_of(tree):
            head = name.split(".")[0]
            if head in mods and name == head:
                findings.append(f"import: {rel} imports `{name}` bare; use `{mods[head]}`")
            if head in ("scripts", "pipelines"):
                # scripts.<folder>.<module>[.attr] or scripts.<folder> (from-import of a module)
                parts = name.split(".")
                cand = None
                for k in (3, 2):
                    if ".".join(parts[:k]) in qualified:
                        cand = ".".join(parts[:k])
                        break
                if cand is None and head == "scripts" and len(parts) == 2:
                    continue          # `from scripts.analysis import evaluate_uq` is resolved below
                if cand is None:
                    findings.append(f"import: {rel} imports `{name}`, which is not a module of the tree")
                else:
                    edges.add(cand)
        # `from scripts.<folder> import <module>` form
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.count(".") == 1 \
                    and node.module.split(".")[0] in ("scripts",):
                for a in node.names:
                    q = f"{node.module}.{a.name}"
                    if q in qualified:
                        edges.add(q)
        if me:
            graph[me] = edges
    # cycle search
    state = {}
    stack = []

    def dfs(u):
        state[u] = 1
        stack.append(u)
        for v in graph.get(u, ()):
            if state.get(v) == 1:
                cyc = stack[stack.index(v):] + [v]
                findings.append("cycle: " + " -> ".join(cyc))
            elif v not in state:
                dfs(v)
        stack.pop()
        state[u] = 2

    for u in sorted(graph):
        if u not in state:
            dfs(u)


def check_forbidden(files, findings):
    for rel in files:
        if rel.suffix not in TEXT_SUFFIXES or str(rel).startswith(HISTORICAL):
            continue
        text = (_REPO_ROOT / rel).read_text(errors="replace")
        for word in FORBIDDEN:
            if word in text:
                findings.append(f"forbidden: {rel} mentions `{word}`")


def check_help(files, findings, gp_models):
    env = dict(os.environ, PYTHONPATH=str(gp_models),
               OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    with tempfile.TemporaryDirectory() as cwd:
        for rel in files:
            if rel.suffix != ".py" or rel.parts[0] not in ("scripts", "pipelines") or rel.stem == "__init__":
                continue
            text = (_REPO_ROOT / rel).read_text()
            if not re.search(r"click\.(command|group)|argparse", text):
                continue      # positional-argument scripts have no --help
            r = subprocess.run([sys.executable, str(_REPO_ROOT / rel), "--help"], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                tail = (r.stderr or r.stdout).strip().splitlines()[-1:]
                findings.append(f"help: {rel} exits {r.returncode}: {tail[0] if tail else ''}")
            else:
                print(f"  --help ok  {rel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper", type=Path, default=None, help="directory whose .tex files are checked for repository paths")
    ap.add_argument("--help-all", action="store_true", help="run --help on every command-line script (slow: imports torch per script)")
    ap.add_argument("--gp-models", type=Path, default=_REPO_ROOT / "al_pmssmwithgp" / "model",
                    help="al_pmssmwithgp/model directory put on PYTHONPATH for --help-all")
    args = ap.parse_args()

    files = tracked_files()
    findings = []
    check_compile(files, findings)
    extra = sorted(args.paper.glob("*.tex")) if args.paper else ()
    check_paths(files, findings, extra)
    check_syspath(files, findings)
    check_imports_and_cycles(files, findings)
    check_forbidden(files, findings)
    if args.help_all:
        check_help(files, findings, args.gp_models)

    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s) over {len(files)} tracked files")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
