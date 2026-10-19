#!/usr/bin/env python3
"""Download the explicitly identified public reconstruction input."""
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
meta=json.loads((ROOT/'data/oregon1-provenance.json').read_text())
dest=ROOT/'data/raw/oregon1_010331.txt.gz'
if dest.exists():
    payload=dest.read_bytes()
else:
    # Honor standard HTTP(S)_PROXY/NO_PROXY variables; no lab-specific endpoint.
    opener=urllib.request.build_opener()
    with opener.open(meta['url'],timeout=60) as response: payload=response.read()
if hashlib.sha256(payload).hexdigest()!=meta['sha256']:
    raise ValueError('Input SHA-256 differs from the pinned public snapshot')
dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(payload)
print('Verified Oregon-1 2001-03-31 input:',meta['sha256'])
