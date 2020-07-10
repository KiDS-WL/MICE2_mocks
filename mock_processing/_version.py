from os.path import dirname

__version__ = "1.0.3"  # NOTE: keep up-to date with git tag version
try:
    import subprocess
    # determine version from git commits
    label = subprocess.check_output(
        ["git", "describe", "--tags"], cwd=dirname(__file__)).strip()
    __version__ = label.decode("utf-8")
except Exception:
    pass
