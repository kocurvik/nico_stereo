"""Upload the archives built by make_zenodo_archives.py to a Zenodo draft.

Works on an existing draft (e.g. one started in the web interface) through
Zenodo's draft-files API (``/api/records/<id>/draft/files``). Each file goes up
as initiate -> one streamed PUT -> commit, and it is verified afterwards.

Zenodo does not accept resuming inside a file through this API (multipart
uploads are initiated but the part and commit calls are refused with 403), so
the unit of resumption is the file:

* files that are complete on the draft are never touched, so after any
  interruption the script continues with the first file that is missing;
* a file that stalls (no data moves for ``--stall-timeout`` seconds) or fails
  is abandoned, its partial entry is deleted, and the same file is started
  again, up to ``--retries`` times (default 20). To keep the cost of a restart
  small, pack the data into archives of moderate size
  (``make_zenodo_archives.py --max-gb``).

For every local file the script looks at what the draft holds:

* complete with the same size       -> skipped (add ``--verify-existing`` to also
  compare the MD5, which reads the whole local file);
* anything else under that name (empty or partial entry, different size)
                                    -> deleted and uploaded again;
* not on the draft                  -> uploaded.

The record is NEVER published by this script -- review the draft and publish in
the web interface. The token is read from the ``ZENODO_TOKEN`` environment
variable (scopes ``deposit:write`` and ``deposit:actions``) and is never printed.

    python tools/upload_to_zenodo.py --list
    python tools/upload_to_zenodo.py                       # upload everything
    python tools/upload_to_zenodo.py nico_stereo_out_core.zip SHA256SUMS
"""
import argparse
import collections
import hashlib
import os
import re
import sys
import time
from pathlib import Path

import requests

API = "https://zenodo.org/api"
CHUNK = 1 << 20


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK * 4), b""):
            h.update(block)
    return h.hexdigest()


def fmt_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Progress:
    """Progress line with percentage, speed and ETAs (this file and the whole run)."""

    def __init__(self, total_bytes: int):
        self.total = total_bytes          # bytes the run has to move over all files
        self.finished = 0                 # bytes of files that are completely uploaded
        self.file_size = 0
        self.file_done = 0
        self.window = collections.deque() # (time, file_done) samples for the speed
        self.last = 0.0

    def start_file(self, size: int) -> None:
        self.file_size, self.file_done = size, 0
        self.window.clear()
        self.window.append((time.time(), 0))

    def add(self, n: int) -> None:
        self.file_done += n
        now = time.time()
        self.window.append((now, self.file_done))
        while len(self.window) > 2 and now - self.window[0][0] > 30:
            self.window.popleft()
        if now - self.last > 2:
            self.show()
            self.last = now

    def show(self) -> None:
        t0, d0 = self.window[0]
        rate = (self.file_done - d0) / max(time.time() - t0, 1e-9)
        eta = lambda left: fmt_time(left / rate) if rate > 1e3 else "?"
        left_file = self.file_size - self.file_done
        left_all = self.total - self.finished - self.file_done
        print(f"\r    {self.file_done / 1e9:5.2f}/{self.file_size / 1e9:.2f} GB "
              f"{100 * self.file_done / max(self.file_size, 1):5.1f}%  {rate / 1e6:5.1f} MB/s  "
              f"ETA file {eta(left_file)}, all {eta(left_all)}   ", end="", flush=True)


class ProgressFile:
    """File-like object that reports progress while requests streams it."""

    def __init__(self, path: Path, progress: Progress):
        self.f = open(path, "rb")
        self.size = path.stat().st_size
        self.progress = progress

    def __len__(self):
        return self.size

    def read(self, n=-1):
        block = self.f.read(CHUNK if n is None or n < 0 else min(n, CHUNK))
        self.progress.add(len(block))
        return block

    def close(self):
        self.f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="file names to upload (default: all in --dir)")
    ap.add_argument("--record", default="23209897", help="draft ID (the number in the Zenodo URL)")
    ap.add_argument("--dir", type=Path, default=Path("D:/Research/data/nico_stereo_zenodo"))
    ap.add_argument("--list", action="store_true", help="only show the draft's files and the plan")
    ap.add_argument("--retries", type=int, default=20, help="attempts per file")
    ap.add_argument("--stall-timeout", type=int, default=60,
                    help="seconds without any data moving before an upload is abandoned and restarted")
    ap.add_argument("--verify-existing", action="store_true",
                    help="also compare the MD5 of files that are already complete on the draft")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN is not set")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    base = f"{API}/records/{args.record}/draft/files"

    def api(method, url, **kw):
        """Request with retries on connection errors and 5xx answers."""
        last = None
        timeout = kw.pop("timeout", 60)
        for attempt in range(1, args.retries + 1):
            try:
                r = s.request(method, url, timeout=timeout, **kw)
                if r.status_code in (401, 403):
                    sys.exit(f"Zenodo refused the request ({r.status_code} {r.text[:100]}); "
                             "the token needs the deposit:write scope")
                if r.status_code < 500:
                    return r
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
            wait = min(5 * attempt, 60)
            print(f"\n    {method} failed ({last}); retry {attempt}/{args.retries} in {wait}s")
            time.sleep(wait)
        sys.exit(f"giving up on {method} {url}: {last}")

    def entries():
        r = api("GET", base)
        if r.status_code == 404:
            sys.exit("draft not found (is the ID right, and is the token from the same account?)")
        r.raise_for_status()
        return {e["key"]: e for e in r.json()["entries"]}

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
        size = path.stat().st_size
        e = remote.get(name)
        if e is None:
            plan.append((name, "upload"))
        elif e.get("status") == "completed" and e.get("size") == size:
            plan.append((name, "check-md5" if args.verify_existing else "skip (already complete)"))
        else:
            plan.append((name, f"replace ({e.get('status')}, {e.get('size') or 0:,} B on the draft)"))
    print("\nplan:")
    for name, what in plan:
        print(f"  {name:<48} {what}")
    if args.list:
        return

    total = sum((args.dir / n).stat().st_size for n, w in plan if not w.startswith("skip"))
    print(f"\n{total / 1e9:.2f} GB to upload in this run "
          f"({args.retries} attempts per file, stall timeout {args.stall_timeout}s)")
    print("If the draft is also open in the browser, leave it alone while this runs.")
    progress = Progress(total)

    for name, what in plan:
        path = args.dir / name
        size = path.stat().st_size
        print(f"\n{name} ({size / 1e9:.2f} GB)")
        if what.startswith("skip"):
            print("    already complete on the draft, skipped")
            continue
        if what == "check-md5":
            print("    complete on the draft; comparing MD5 ...", flush=True)
            if md5_of(path) == (remote[name].get("checksum") or "").replace("md5:", ""):
                print("    identical, skipped")
                continue
            print("    MD5 differs, replacing")

        for attempt in range(1, args.retries + 1):
            if name in entries():               # empty/partial entry from an earlier try or the web UI
                api("DELETE", f"{base}/{name}")
            i = api("POST", base, json=[{"key": name}])
            if i.status_code not in (200, 201):
                print(f"    attempt {attempt}: initiate failed, HTTP {i.status_code} {i.text[:200]}")
                time.sleep(10)
                continue
            progress.start_file(size)
            pf = ProgressFile(path, progress)
            try:
                # the read timeout also covers a send() that makes no progress: a stalled
                # connection raises here after --stall-timeout seconds
                up = s.put(f"{base}/{name}/content", data=pf,
                           headers={"Content-Type": "application/octet-stream"},
                           timeout=(30, args.stall_timeout))
                ok, why = up.status_code in (200, 201), f"HTTP {up.status_code} {up.text[:150]}"
            except requests.RequestException as err:
                ok, why = False, f"{type(err).__name__}: {str(err)[:150]}"
            finally:
                pf.close()
            print()
            if ok:
                c = api("POST", f"{base}/{name}/commit", timeout=600)
                if c.status_code in (200, 201):
                    break
                why = f"commit failed, HTTP {c.status_code} {c.text[:150]}"
            wait = min(10 * attempt, 120)
            print(f"    attempt {attempt}/{args.retries} failed: {why}\n"
                  f"    restarting {name} in {wait}s (completed files are kept)")
            time.sleep(wait)
        else:
            sys.exit(f"giving up on {name} after {args.retries} attempts")

        info = entries().get(name, {})
        if info.get("status") != "completed" or info.get("size") != size:
            sys.exit(f"{name}: after the commit the draft shows status={info.get('status')}, "
                     f"size={info.get('size')} (expected {size})")
        reported = (info.get("checksum") or c.json().get("checksum") or "").replace("md5:", "")
        local = md5_of(path)
        if re.fullmatch(r"[0-9a-f]{32}", reported) and reported != local:
            sys.exit(f"{name}: MD5 mismatch after upload (Zenodo {reported}, local {local})")
        print(f"    committed, size and MD5 verified ({local})")
        progress.finished += size

    print("\nall files are on the draft. Review it and publish in the web interface; "
          "this script does not publish.")


if __name__ == "__main__":
    main()
