"""Minimal mzML writer (centroided, 64-bit, zlib) for test data."""
from __future__ import annotations

import base64
import zlib
from xml.sax.saxutils import escape

import numpy as np

_HEAD = """<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd" version="1.1.0">
  <cvList count="3">
    <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" version="4.1.0" URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>
    <cv id="UO" fullName="Unit Ontology" version="1.0" URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/>
    <cv id="IMS" fullName="Imaging MS Ontology" version="1.0" URI="https://raw.githubusercontent.com/imzML/imzML/master/imagingMS.obo"/>
  </cvList>
  <fileDescription>
    <fileContent>
      <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>
      <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
    </fileContent>
  </fileDescription>
  <softwareList count="1"><software id="sw1" version="1.0"><cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="semaglutide_demo_generator"/></software></softwareList>
  <instrumentConfigurationList count="1"><instrumentConfiguration id="IC1"><cvParam cvRef="MS" accession="MS:1000031" name="instrument model" value="synthetic QTOF"/></instrumentConfiguration></instrumentConfigurationList>
  <dataProcessingList count="1"><dataProcessing id="dp1"><processingMethod order="0" softwareRef="sw1"><cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/></processingMethod></dataProcessing></dataProcessingList>
"""


def _arr(a, kind: str) -> str:
    raw = zlib.compress(np.asarray(a, "<f8").tobytes())
    b64 = base64.b64encode(raw).decode()
    if kind == "mz":
        cv = '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>'
    else:
        cv = '<cvParam cvRef="MS" accession="MS:1000515" name="intensity array" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>'
    return (f'<binaryDataArray encodedLength="{len(b64)}"><cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            f'<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression"/>{cv}<binary>{b64}</binary></binaryDataArray>')


def write_mzml(scans, path: str):
    scans = sorted(scans, key=lambda s: (s.rt, s.ms_level))
    out = [_HEAD, f'  <run id="semaglutide_demo" defaultInstrumentConfigurationRef="IC1">\n    <spectrumList count="{len(scans)}" defaultDataProcessingRef="dp1">\n']
    for i, s in enumerate(scans):
        tic = float(np.sum(s.inten))
        prec = ""
        if s.ms_level > 1 and s.prec_mz:
            z = f'<cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="{s.prec_z}"/>' if s.prec_z else ""
            prec = (f'<precursorList count="1"><precursor><isolationWindow>'
                    f'<cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="{s.prec_mz:.5f}" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>'
                    f'<cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="2" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>'
                    f'<cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="2" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>'
                    f'</isolationWindow><selectedIonList count="1"><selectedIon>'
                    f'<cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="{s.prec_mz:.5f}" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>{z}'
                    f'</selectedIon></selectedIonList><activation><cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation" value=""/>'
                    f'<cvParam cvRef="MS" accession="MS:1000045" name="collision energy" value="35" unitCvRef="UO" unitAccession="UO:0000266" unitName="electronvolt"/></activation></precursor></precursorList>')
        lvl_name = "MS1 spectrum" if s.ms_level == 1 else "MSn spectrum"
        out.append(
            f'      <spectrum index="{i}" id="scan={i + 1}" defaultArrayLength="{len(s.mz)}">'
            f'<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{s.ms_level}"/>'
            f'<cvParam cvRef="MS" accession="MS:{"1000579" if s.ms_level == 1 else "1000580"}" name="{lvl_name}" value=""/>'
            f'<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>'
            f'<cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>'
            f'<cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="{tic:.6g}"/>'
            f'<scanList count="1"><cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>'
            f'<scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{s.rt:.5f}" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/></scan></scanList>'
            f'{prec}<binaryDataArrayList count="2">{_arr(s.mz, "mz")}{_arr(s.inten, "int")}</binaryDataArrayList></spectrum>\n')
    out.append("    </spectrumList>\n  </run>\n</mzML>\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
