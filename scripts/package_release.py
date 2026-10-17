"""Build an allowlisted deployment archive without private/runtime state."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ("backend/*.py", "frontend/*.html", "frontend/*.css", "frontend/*.js",
            "frontend/assets/*.wav", "frontend/vendor/*.js", "tests/*.py", "scripts/*.py",
            "backend/models/*.rnnn", "backend/models/*.md", "scripts/*.ps1", "frontend/assets/*.md",
            ".github/workflows/*.yml", "separation_service/*.py", "separation_service/*.txt",
            "separation_service/*.lock")
FILES = ("README.md",
         "requirements.txt", "requirements.lock", "requirements-dev.txt", "render.yaml",
         "runtime.txt", ".python-version", ".gitignore", ".gitattributes", "Procfile", "start_local.cmd",
         "render-separation.yaml",
         ".dockerignore", "separation_service/Dockerfile",
         "start_separation_local.cmd",
         "frontend/vendor/README.md", "frontend/vendor/LUCIDE-LICENSE")


def main():
    files = {ROOT / name for name in FILES}
    for pattern in PATTERNS:
        files.update(path for path in ROOT.glob(pattern) if path.is_file() and not path.is_symlink())
    target = ROOT / "dist" / "Noise-Cancellation.zip"
    target.parent.mkdir(exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(target) as archive:
        assert archive.testzip() is None
    print(f"Built {target}: {len(files)} files, {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
