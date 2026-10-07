"""Upload the archives built by make_zenodo_archives.py to a Zenodo draft.

Works on an existing draft (e.g. one started in the web interface) through
Zenodo's draft-files API (``/api/records/<id>/draft/files``). For every local
file it looks at what the draft already holds:

* committed with the same size      -> skipped (already uploaded; add
  ``--verify-existing`` to also compare the MD5, which reads the whole file);
* same name but pending/partial/different -> deleted and uploaded again;
* not on the draft                  -> uploaded.

A file is uploaded as: initiate -> one streamed PUT of the content -> commit.
Each file is re-tried on failure, and after the commit the MD5 that Zenodo
reports is compared with the local one. The record is NEVER published by this
script -- review the draft and publish in the web interface.

The token is read from the ``ZENODO_TOKEN`` environment variable (scopes
``deposit:write`` and ``deposit:actions``) and is never printed.

    python tools/upload_to_zenodo.py --list
    python tools/upload_to_zenodo.py                       # upload everything
    python tools/upload_to_zenodo.py nico_stereo_out_core.zip SHA256SUMS
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import requests

API = "https://zenodo.org/api"
CHUNK = 1 << 22


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class ProgressFile:
    """File-like object that reports progress while requests streams it."""

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.size = path.stat().st_size
        self.sent = 0
        self.t0 = time.time()
        self.last = 0.0

    def __len__(self):
        return self.size

    def read(self, n=-1):
        block = self.f.read(CHUNK if n is None or n < 0 else min(n, CHUNK))
        self.sent += len(block)
        now = time.time()
        if now - self.last > 2 or not block:
            rate = self.sent / max(now - self.t0, 1e-9) / 1e6
            print(f"\r    {self.sent / 1e9:6.2f} / {self.size / 1e9:.2f} GB  {rate:5.1f} MB/s   ",
                  end="", flush=True)
            self.last = now
        return block

    def close(self):
        self.f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="file names to upload (default: all in --dir)")
    ap.add_argument("--record", default="23209897", help="draft ID (the number in the Zenodo URL)")
    ap.add_argument("--dir", type=Path, default=Path("D:/Research/data/nico_stereo_zenodo"))
    ap.add_argument("--list", action="store_true", help="only show the draft's files and the plan")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--verify-existing", action="store_true",
                    help="also compare the MD5 of files that are already complete on the draft")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN is not set")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    base = f"{API}/records/{args.record}/draft/files"

    def entries():
        for attempt in range(5):
            r = s.get(base, timeout=60)
            if r.status_code in (401, 403):
                sys.exit(f"Zenodo refused the token ({r.status_code}); it needs the deposit:write scope")
            if r.status_code == 404:
                sys.exit("draft not found (is the ID right, and is the token from the same account?)")
            if r.status_code < 500:
                r.raise_for_status()
                return {e["key"]: e for e in r.json()["entries"]}
            time.sleep(3 * (attempt + 1))
        sys.exit("Zenodo keeps returning server errors; try again later")

    remote = entries()
    print(f"draft {args.record} holds {len(remote)} file entr{'y' if len(remote) == 1 else 'ies'}:")
    for name, e in sorted(remote.items()):
        print(f"  {name:<48} {e.get('size') or 0:>13,} B  {e.get('status'):<9} {e.get('checksum') or ''}")

    names = args.files or sorted(p.name for p in args.dir.iterdir()
                                 if p.suffix == ".zip" or p.name == "SHA256SUMS")
    plan = []
    for name in names:
        path = args.dir / name
        if not path.is_file():
            sys.exit(f"missing local file {path}")
        e = remote.get(name)
        if e is None:
            plan.append((name, "upload"))
        elif e.get("status") == "completed" and e.get("size") == path.stat().st_size:
            plan.append((name, "check-md5" if args.verify_existing else "skip (already complete)"))
        else:
            plan.append((name, f"replace ({e.get('status')}, {e.get('size') or 0:,} B on the draft)"))
    print("\nplan:")
    for name, what in plan:
        print(f"  {name:<48} {what}")
    if args.list:
        return

    for name, what in plan:
        path = args.dir / name
        size = path.stat().st_size
        print(f"\n{name} ({size / 1e9:.2f} GB)")
        e = remote.get(name)
        if what.startswith("skip"):
            print("    already complete on the draft, skipped")
            continue
        if what == "check-md5":
            print("    complete on the draft; comparing MD5 ...", flush=True)
            if md5_of(path) == (e.get("checksum") or "").replace("md5:", ""):
                print("    identical, skipped")
                continue
            print("    MD5 differs, replacing")
        local_md5 = None

        for attempt in range(1, args.retries + 1):
            if name in entries():                       # pending, partial or replaced copy
                d = s.delete(f"{base}/{name}", timeout=60)
                if d.status_code not in (200, 204):
                    print(f"    could not delete the old entry: {d.status_code} {d.text[:200]}")
                    time.sleep(10)
                    continue
            i = s.post(base, json=[{"key": name}], timeout=60)
            if i.status_code not in (200, 201):
                print(f"    attempt {attempt}: initiate failed, HTTP {i.status_code} {i.text[:200]}")
                time.sleep(10 * attempt)
                continue
            pf = ProgressFile(path)
            try:
                up = s.put(f"{base}/{name}/content", data=pf, timeout=(30, 900),
                           headers={"Content-Type": "application/octet-stream"})
                print()
                if up.status_code not in (200, 201):
                    print(f"    attempt {attempt}: upload failed, HTTP {up.status_code} {up.text[:200]}")
                    time.sleep(10 * attempt)
                    continue
            except requests.RequestException as err:
                print(f"\n    attempt {attempt}: {type(err).__name__}: {str(err)[:200]}")
                time.sleep(10 * attempt)
                continue
            finally:
                pf.close()
            c = s.post(f"{base}/{name}/commit", timeout=300)
            if c.status_code not in (200, 201):
                print(f"    attempt {attempt}: commit failed, HTTP {c.status_code} {c.text[:200]}")
                time.sleep(10 * attempt)
                continue
            break
        else:
            sys.exit(f"giving up on {name} after {args.retries} attempts")

        local_md5 = local_md5 or md5_of(path)
        reported = (c.json().get("checksum") or "").replace("md5:", "")
        if reported != local_md5:
            sys.exit(f"{name}: MD5 mismatch after upload (Zenodo {reported}, local {local_md5})")
        print(f"    uploaded and committed, MD5 verified ({local_md5})")

    print("\nall files are on the draft. Review it and publish in the web interface; "
          "this script does not publish.")


if __name__ == "__main__":
    main()
