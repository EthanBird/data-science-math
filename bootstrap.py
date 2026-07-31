from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path


parts = sorted(Path("payload").glob("part-*.b64"))
if not parts:
    raise SystemExit("missing payload parts")

encoded = "".join(path.read_text(encoding="ascii") for path in parts)
data = base64.b64decode(encoded, validate=True)

with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
    archive.extractall(".", filter="data")

for path in parts:
    path.unlink()
Path("payload").rmdir()
