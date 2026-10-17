"""Explicit setup download of fixed public weights, never a request-time fetch."""

import argparse
import hashlib
from pathlib import Path
import ssl
import urllib.request

from separation_service.model import ASSETS, REPOSITORY, REVISION, verify_assets


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in ASSETS.items():
        target = directory / name
        if target.is_file() and not target.is_symlink() and target.stat().st_size == size:
            if hashlib.sha256(target.read_bytes()).hexdigest() == digest:
                continue
        temporary = directory / (name + ".partial")
        if target.is_symlink() or temporary.is_symlink():
            raise ValueError("Refusing a symbolic-link model destination.")
        checksum, received = hashlib.sha256(), 0
        try:
            url = "https://huggingface.co/" + REPOSITORY + "/resolve/" + REVISION + "/" + name
            with urllib.request.urlopen(url, timeout=90, context=ssl.create_default_context()) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > size:
                        raise ValueError("The model download exceeded its fixed size.")
                    checksum.update(chunk)
                    output.write(chunk)
            if received != size or checksum.hexdigest() != digest:
                raise ValueError("The model download failed its integrity check.")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print("Verified " + name, flush=True)
    verify_assets(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".tools/separation-model"))
    download(parser.parse_args().output)
