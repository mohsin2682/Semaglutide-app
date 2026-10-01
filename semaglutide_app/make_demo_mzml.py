"""Create semaglutide_demo.mzML (synthetic LC-MS/MS run) for testing the app.
Usage: python make_demo_mzml.py [output.mzML]"""
import sys
from sema.demo import generate_demo_run
from sema.mzml_writer import write_mzml

out = sys.argv[1] if len(sys.argv) > 1 else "semaglutide_demo.mzML"
write_mzml(generate_demo_run(), out)
print("written:", out)
