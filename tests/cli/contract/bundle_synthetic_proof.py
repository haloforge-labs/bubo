"""Independently check a support-bundle preview, saved event, and ZIP image.

This reads only synthetic fixture output. It never contacts a server or loads a
model. The process driver imports verify(), and the script can also be run on
retained evidence with --preview, --saved, and --zip.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

NAMES = (
    "versions.json",
    "settings.json",
    "status.json",
    "diagnose.json",
    "history.json",
    "logs.json",
    "manifest.json",
)
MAX_MEMBER = 32 * 1024
MAX_TEXT = 128 * 1024
MAX_ZIP = 160 * 1024


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def verify(preview_event, saved_event, zip_path, secrets):
    assert preview_event["type"] == saved_event["type"] == "support_bundle"
    preview, saved = preview_event["payload"], saved_event["payload"]
    assert preview["saved"] is False and saved["saved"] is True
    for field in ("files", "text_preview", "sha256", "zip_bytes", "bundle_sha256"):
        assert preview[field] == saved[field], f"preview/save mismatch: {field}"
    assert tuple(preview["files"]) == NAMES
    assert set(preview["text_preview"]) == set(NAMES)
    assert set(preview["sha256"]) == set(NAMES)

    image = Path(zip_path).read_bytes()
    assert len(image) == preview["zip_bytes"] <= MAX_ZIP
    assert hashlib.sha256(image).hexdigest() == preview["bundle_sha256"]
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        assert tuple(archive.namelist()) == NAMES
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.file_size <= MAX_MEMBER
            data = archive.read(info)
            text = data.decode("utf-8")
            total += len(data)
            assert text == preview["text_preview"][info.filename]
            assert hashlib.sha256(data).hexdigest() == preview["sha256"][info.filename]
            for secret in secrets:
                assert secret.encode() not in data, f"secret in {info.filename}"
            for value in _strings(json.loads(text)):
                for secret in secrets:
                    assert secret not in value, f"secret in {info.filename} JSON"
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema_version"] == 1
        assert [row["name"] for row in manifest["files"]] == list(NAMES[:-1])
        for row in manifest["files"]:
            name = row["name"]
            assert row["bytes"] == len(archive.read(name))
            assert row["sha256"] == preview["sha256"][name]
    assert total <= MAX_TEXT
    for event in (preview_event, saved_event):
        for text in _strings(event):
            for secret in secrets:
                assert secret not in text, "secret in event"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--secret", action="append", default=[])
    args = parser.parse_args()
    verify(json.loads(args.preview.read_text(encoding="utf-8")),
           json.loads(args.saved.read_text(encoding="utf-8")),
           args.zip, args.secret)
    print("Support bundle ZIP, preview, hashes, and privacy: passed")


if __name__ == "__main__":
    main()
