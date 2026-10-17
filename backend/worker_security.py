"""Keep service credentials and Python injection settings out of child workers."""

import os
from pathlib import Path
import sys


PYTHON_FLAGS = ("-B", "-E", "-s")


def worker_environment(directory, threads=1):
    root = str(Path(directory).resolve())
    environment = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
                   if name in os.environ}
    paths = [str(Path(sys.executable).parent)]
    if os.name == "nt":
        system_root = os.environ.get("SYSTEMROOT", os.environ.get("WINDIR", r"C:\Windows"))
        environment["SYSTEMROOT"] = system_root
        paths.append(str(Path(system_root) / "System32"))
    else:
        paths.extend(os.defpath.split(os.pathsep))
    environment.update({
        "PATH": os.pathsep.join(paths),
        "HOME": root, "USERPROFILE": root, "TMPDIR": root, "TEMP": root, "TMP": root,
        "XDG_CACHE_HOME": str(Path(root) / "cache"),
        "HF_HOME": str(Path(root) / "huggingface"), "TORCH_HOME": str(Path(root) / "torch"),
        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
    })
    return environment
