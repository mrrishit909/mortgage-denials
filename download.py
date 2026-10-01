"""Step 1: HMDA mortgage applications, Tampa Bay (Hillsborough, Pinellas, Pasco, Hernando), 2022-2024.

    ./venv/bin/python download.py

Source: CFPB/FFIEC HMDA Data Browser API (public loan/application register, already privacy-modified by the CFPB).
Home-purchase applications the lender decided: approved and originated (1), approved but not taken up (2), or denied (3). Files go to data/raw/
(not committed). Counties rather than the metro code, because the metro codes changed in the 2024 data.
"""
import ssl
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
COUNTIES = "12057,12103,12101,12053"
URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?counties={c}&years={y}&actions_taken=1,2,3&loan_purposes=1"

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    for year in (2022, 2023, 2024):
        out = RAW / f"hmda_{year}.csv"
        out.write_bytes(urllib.request.urlopen(URL.format(c=COUNTIES, y=year), context=ctx, timeout=600).read())
        print(year, sum(1 for _ in out.open()) - 1, "rows")
