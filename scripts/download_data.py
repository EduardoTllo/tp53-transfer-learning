"""Download the breast cancer cohorts for the deep learning phase from cBioPortal.

Expression, mutations and clinical files come from the official cBioPortal study
archives (datahub.assets.cbioportal.org), the same files the website's
"Download" button serves. The archives are streamed and only the needed files are
extracted, so the .tar.gz is never stored. Small clinical tables (PAM50, ER) for
TCGA and CPTAC come from the cBioPortal REST API. The HGNC gene table is used to
translate old gene symbols (SMC 2018) to current ones. Every run writes a manifest
with the URL, size, MD5, server date and the extracted files.

Usage (from the repository root):
    python scripts/download_data.py --dry-run
    python scripts/download_data.py
    python scripts/download_data.py --studies brca_smc_2018
    python scripts/download_data.py --solo-hgnc
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATAHUB = "https://datahub.assets.cbioportal.org/{study}.tar.gz"
API = "https://www.cbioportal.org/api"
HGNC = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
HEADERS = {"User-Agent": "tp53-dl-download/1.0 (research)"}

# Study -> role in the design and which expression file to keep.
# Role of each study in the design (source or target).
STUDIES = {
    "brca_tcga_gdc": {
        "rol": "fuente (TCGA-BRCA, mismo pipeline GDC que CPTAC)",
        "expresion": r"data_mrna_seq_tpm\.txt$",
    },
    "brca_metabric": {
        "rol": "fuente (microarreglo Illumina HT-12)",
        "expresion": r"data_mrna_illumina_microarray\.txt$",
    },
    "breast_cptac_gdc": {
        "rol": "objetivo (cohorte 1 del manuscrito)",
        "expresion": r"data_mrna_seq_tpm\.txt$",
    },
    "brca_smc_2018": {
        "rol": "objetivo (cohorte 2 del manuscrito)",
        "expresion": r"data_mrna_seq_tpm\.txt$",
    },
}

# Files kept from every archive besides the expression matrix.
KEEP_ALWAYS = [
    r"data_mutations[^/]*\.txt$",
    r"data_clinical_(patient|sample)\.txt$",
    r"meta_study\.txt$",
    r"meta_(mutations|mrna)[^/]*\.txt$",
    r"case_lists/.*\.txt$",
]

# Clinical attributes taken from the REST API (study, attribute, level).
API_CLINICAL = [
    ("brca_tcga_pan_can_atlas_2018", "SUBTYPE", "PATIENT"),          # PAM50 de TCGA
    ("brca_tcga_pub2015", "ER_STATUS_BY_IHC", "SAMPLE"),             # ER de TCGA
    ("brca_cptac_2020", "PAM50", "SAMPLE"),                          # PAM50 de CPTAC
    ("brca_cptac_2020", "ER_UPDATED_CLINICAL_STATUS", "PATIENT"),    # ER de CPTAC
]


def request(url, method="GET"):
    """cBioPortal rejects Python's default User-Agent (HTTP 403 since 2026-10-06), so send our own."""
    return urllib.request.Request(url, method=method, headers=HEADERS)


def api_get(path):
    with urllib.request.urlopen(request(API + path), timeout=120) as r:
        return json.load(r)


def head(url):
    with urllib.request.urlopen(request(url, "HEAD"), timeout=60) as r:
        return int(r.headers.get("Content-Length", 0)), r.headers.get("Last-Modified", "")


class HashingReader:
    """File-like wrapper that hashes and counts bytes while tarfile reads the stream."""

    def __init__(self, raw, total):
        self.raw, self.total, self.read_bytes = raw, total, 0
        self.md5 = hashlib.md5()
        self.last_print = 0.0

    def read(self, n=-1):
        chunk = self.raw.read(n)
        self.md5.update(chunk)
        self.read_bytes += len(chunk)
        now = time.time()
        if now - self.last_print > 5:
            pct = 100 * self.read_bytes / self.total if self.total else 0
            print(f"    {self.read_bytes / 1e6:,.0f} / {self.total / 1e6:,.0f} MB ({pct:.0f}%)", flush=True)
            self.last_print = now
        return chunk


def wanted(rel, expr_pattern):
    return bool(re.search(expr_pattern, rel)) or any(re.search(p, rel) for p in KEEP_ALWAYS)


def download_study(study, cfg, manifest):
    url = DATAHUB.format(study=study)
    size, modified = head(url)
    print(f"\n== {study} ({cfg['rol']}) {size / 1e6:,.0f} MB, servidor: {modified}", flush=True)
    dest = RAW / study
    dest.mkdir(parents=True, exist_ok=True)
    kept = []
    with urllib.request.urlopen(request(url), timeout=600) as resp:
        reader = HashingReader(resp, size)
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for member in tar:
                rel = member.name.split("/", 1)[-1]
                if not member.isfile() or not wanted(rel, cfg["expresion"]):
                    continue
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                kept.append({"archivo": rel, "bytes": member.size})
        # drain the rest so the MD5 covers the whole archive
        while reader.read(1 << 20):
            pass
    if reader.read_bytes != size:
        raise RuntimeError(f"{study}: descargados {reader.read_bytes} de {size} bytes")
    expr_found = any(re.search(cfg["expresion"], k["archivo"]) for k in kept)
    if not expr_found:
        print(f"  ATENCIÓN: no se encontró la matriz de expresión ({cfg['expresion']}). Revisar los nombres.")
    manifest["estudios"][study] = {
        "rol": cfg["rol"],
        "url": url,
        "bytes_archivo": size,
        "md5_archivo": reader.md5.hexdigest(),
        "ultima_modificacion_servidor": modified,
        "archivos_extraidos": kept,
        "matriz_expresion_encontrada": expr_found,
    }
    print(f"  extraídos {len(kept)} archivos en {dest}")


def download_api_clinical(manifest):
    dest = RAW / "api_clinica"
    dest.mkdir(parents=True, exist_ok=True)
    for study, attr, level in API_CLINICAL:
        q = urllib.parse.urlencode({"clinicalDataType": level, "attributeId": attr, "projection": "SUMMARY"})
        rows = api_get(f"/studies/{study}/clinical-data?{q}")
        out = dest / f"{study}__{attr}.tsv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["studyId", "patientId", "sampleId", "attribute", "value"])
            for r in rows:
                w.writerow([r.get("studyId"), r.get("patientId"), r.get("sampleId", ""), r.get("clinicalAttributeId"), r.get("value")])
        manifest["api_clinica"].append({"estudio": study, "atributo": attr, "nivel": level,
                                        "filas": len(rows), "archivo": out.name})
        print(f"  API {study} {attr}: {len(rows)} filas → {out.name}")


def download_hgnc(manifest):
    """HGNC gene table. Used to translate the old gene symbols of SMC 2018 to current ones."""
    size, modified = head(HGNC)
    dest = RAW / "referencias"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "hgnc_complete_set.txt"
    with urllib.request.urlopen(request(HGNC), timeout=300) as r, open(out, "wb") as f:
        reader = HashingReader(r, size)
        shutil.copyfileobj(reader, f)
    manifest["referencias"] = {"hgnc": {"url": HGNC, "bytes": out.stat().st_size, "md5": reader.md5.hexdigest(),
                                        "ultima_modificacion_servidor": modified, "archivo": str(out.relative_to(RAW))}}
    print(f"  HGNC: {out.stat().st_size / 1e6:,.1f} MB → {out.relative_to(RAW)}")


def write_manifest(manifest, prefix):
    out = RAW / f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifiesto: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="solo muestra tamaños, no descarga")
    ap.add_argument("--studies", nargs="*", default=list(STUDIES), help="subconjunto de estudios")
    ap.add_argument("--solo-hgnc", action="store_true", help="baja solo la tabla de genes de HGNC")
    args = ap.parse_args()

    if args.solo_hgnc:
        manifest = {"fecha_descarga": dt.datetime.now().isoformat(timespec="seconds"),
                    "script": "scripts/download_data.py --solo-hgnc"}
        download_hgnc(manifest)
        write_manifest(manifest, "manifiesto_hgnc")
        return

    total = 0
    for s in args.studies:
        size, modified = head(DATAHUB.format(study=s))
        total += size
        print(f"{s:20s} {size / 1e6:8,.0f} MB  ({modified})  → {STUDIES[s]['rol']}")
    print(f"{'TOTAL':20s} {total / 1e6:8,.0f} MB comprimidos (se extraen solo los archivos necesarios)")
    if args.dry_run:
        return

    manifest = {
        "fecha_descarga": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/download_data.py",
        "estudios": {},
        "api_clinica": [],
    }
    for s in args.studies:
        download_study(s, STUDIES[s], manifest)
    print("\n== Clínica desde la API")
    download_api_clinical(manifest)
    print("\n== Tabla de genes de HGNC")
    download_hgnc(manifest)
    write_manifest(manifest, "manifiesto_descarga")


if __name__ == "__main__":
    main()
