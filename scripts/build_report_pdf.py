"""Render the Markdown report to PDF using headless Chrome.

Chrome rather than a PDF library because the report is mostly tables and
figures, and a browser already knows how to lay those out, break them across
pages and embed images at print resolution.

The intermediate HTML is written into ``report/`` rather than a system
temporary directory. The Markdown references its figures as
``../plots/...`` and ``../predictions/...``, and those relative paths only
resolve if the page is loaded from the directory the Markdown lives in.

Usage
-----
    python scripts/build_report_pdf.py
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "report"
REPORT_MD = REPORT_DIR / "Segmentation_Benchmarking_Report.md"
REPORT_HTML = REPORT_DIR / "Segmentation_Benchmarking_Report.html"
REPORT_PDF = REPORT_DIR / "Segmentation_Benchmarking_Report.pdf"

BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]

CSS = """
@page { size: Letter; margin: 0.7in; }

body {
  font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #0b0b0b;
  max-width: 100%;
}

h1 { font-size: 20pt; margin: 0 0 6px; letter-spacing: -0.01em; }

h2 {
  font-size: 13.5pt;
  margin: 26px 0 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid #e1e0d9;
  /* Keep a heading with the content it introduces. */
  break-after: avoid;
  break-inside: avoid;
}

h3 {
  font-size: 11.5pt;
  margin: 18px 0 6px;
  break-after: avoid;
  color: #52514e;
}

p { margin: 0 0 9px; orphans: 3; widows: 3; }

strong { font-weight: 640; }

code {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 9pt;
  background: #f4f3ee;
  padding: 1px 4px;
  border-radius: 3px;
}

pre {
  background: #f4f3ee;
  padding: 10px 12px;
  border-radius: 5px;
  font-size: 8.5pt;
  line-height: 1.45;
  overflow-x: auto;
  break-inside: avoid;
}
pre code { background: none; padding: 0; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0 14px;
  font-size: 8.8pt;
  break-inside: avoid;
}
th, td {
  border: 1px solid #e1e0d9;
  padding: 4px 7px;
  text-align: left;
}
th { background: #f4f3ee; font-weight: 640; }
/* Right-alignment is applied per column by align_numeric_columns(), which
   decides from the cell contents. Blanket-aligning every column after the
   first would right-align prose in the task-definition tables. */
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }

img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 10px auto 4px;
  break-inside: avoid;
}

em { color: #52514e; }

ul, ol { margin: 0 0 10px; padding-left: 22px; }
li { margin-bottom: 4px; }

hr { border: none; border-top: 1px solid #e1e0d9; margin: 18px 0; }

blockquote {
  margin: 0 0 10px;
  padding-left: 12px;
  border-left: 3px solid #e1e0d9;
  color: #52514e;
}
"""


def find_browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).is_file():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "No Chromium-family browser found. Install Google Chrome, or render "
        "the Markdown with another tool:\n"
        f"  {REPORT_MD.relative_to(REPO_ROOT)}"
    )


NUMERIC = re.compile(r"^-?[\d,]+\.?\d*%?$")


def align_numeric_columns(html: str) -> str:
    """Right-align only the columns that hold numbers.

    The report contains two kinds of table. Measurement tables want their
    values right-aligned so digits line up down the column; the task and
    policy tables are prose and want the default. Deciding per column from
    the contents handles both without the Markdown having to carry markup
    for it.

    A column counts as numeric when most of its non-empty body cells parse
    as a number - "N/A" is common in these tables and should not disqualify
    a column that is otherwise measurements.
    """


    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Without bs4 the tables stay left-aligned, which is correct for
        # prose and merely less tidy for measurements. Not worth a hard
        # dependency for a cosmetic improvement.
        return html

    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        body_rows = [r for r in rows if r.find_all("td")]
        if not body_rows:
            continue

        columns = max(len(r.find_all("td")) for r in body_rows)
        for column in range(columns):
            values = []
            for row in body_rows:
                cells = row.find_all("td")
                if column < len(cells):
                    values.append(cells[column].get_text(strip=True))
            considered = [v for v in values if v and v != "N/A"]
            if not considered:
                continue
            numeric = sum(1 for v in considered if NUMERIC.match(v))
            if numeric / len(considered) < 0.6:
                continue

            for row in rows:
                cells = row.find_all(["td", "th"])
                if column < len(cells):
                    cell = cells[column]
                    cell["class"] = cell.get("class", []) + ["num"]

    return str(soup)


def build_html() -> Path:
    try:
        import markdown
    except ImportError:
        raise SystemExit(
            "The `markdown` package is required:\n"
            "  pip install -e '.[report]'"
        )

    if not REPORT_MD.is_file():
        raise SystemExit(
            "No report to render. Generate it first:\n"
            "  python scripts/generate_report.py"
        )

    body = markdown.markdown(
        REPORT_MD.read_text(),
        extensions=["tables", "fenced_code", "toc", "attr_list"],
    )
    body = align_numeric_columns(body)
    REPORT_HTML.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>Segmentation Benchmarking Report</title>"
        f"<style>{CSS}</style></head><body>\n{body}\n</body></html>"
    )
    return REPORT_HTML


def main() -> int:
    html = build_html()
    browser = find_browser()

    print(f"Rendering with {Path(browser).name}")
    result = subprocess.run(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={REPORT_PDF}",
            html.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    if not REPORT_PDF.is_file():
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit("Chrome did not produce a PDF.")

    size = REPORT_PDF.stat().st_size / (1024 * 1024)
    print(f"Wrote {REPORT_PDF.relative_to(REPO_ROOT)} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
