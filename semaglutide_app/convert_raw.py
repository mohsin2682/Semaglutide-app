"""Convert vendor QTOF data (.d Agilent/Bruker, .wiff SCIEX, .raw Waters) to centroided mzML
with ProteoWizard MSConvert.  Usage:  python convert_raw.py <file_or_folder> [outdir]

Requires `msconvert` on PATH (ProteoWizard).  On a machine without it, use the docker image:
  docker run --rm -v "$PWD":/data proteowizard/pwiz-skyline-i-agree-to-the-vendor-licenses \
      wine msconvert /data/sample.d --mzML --filter "peakPicking vendor msLevel=1-2" -o /data/out
"""
import subprocess
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / "mzML"
out.mkdir(exist_ok=True, parents=True)
cmd = ["msconvert", str(src), "--mzML", "--64", "--zlib",
       "--filter", "peakPicking vendor msLevel=1-2", "-o", str(out)]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
print("Done ->", out)
