"""Combine manual/index.html + part0..part6.html into one Word document (2026-10-06).
1. merge the <main> bodies, prefixing every id/anchor with the part number so links stay unique
2. render each inline <svg> diagram to PNG with headless Chromium (Playwright), since Word drops inline SVG
3. open the merged HTML in Word (COM), embed the pictures, save as .docx
Run: python build_docx.py   -> manual/NAPS-625_여명_오버홀_매뉴얼.docx
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(HERE, "_combined_for_word.html")
IMG_DIR = os.path.join(HERE, "_word_img")
OUT_DOCX = os.path.join(HERE, "NAPS-625_여명_오버홀_매뉴얼.docx")
FILES = ["index.html"] + [f"part{i}.html" for i in range(7)]

CSS = """
body{font-family:'Malgun Gothic',sans-serif;font-size:10.5pt;line-height:1.6;color:#1d2228}
h1{font-size:20pt;color:#8a3a10;margin-top:0} h2{font-size:15pt;color:#8a3a10;border-top:1px solid #ccc;padding-top:6pt}
h3{font-size:12.5pt} h4{font-size:11pt;color:#555}
pre{font-family:Consolas,monospace;font-size:8.5pt;background:#f3efe6;border:1px solid #ddd;padding:6pt;white-space:pre-wrap}
code{font-family:Consolas,monospace;font-size:9pt;background:#f3efe6}
table{border-collapse:collapse;width:100%;font-size:9.5pt} th,td{border:1px solid #bbb;padding:3pt 5pt;vertical-align:top} th{background:#efe9dd}
.box{border:1px solid #ddd;border-left:4pt solid #b4531c;background:#fbf9f5;padding:6pt 9pt;margin:8pt 0}
.toc{border:1px solid #ddd;padding:6pt 12pt;background:#fbf9f5}
figcaption{font-size:9pt;color:#666}
"""


def body_of(html):
    m = re.search(r"<main[^>]*>(.*)</main>", html, re.S)
    return m.group(1) if m else html


def prefix_ids(part, html):
    p = "p" + part
    html = re.sub(r'id="([^"]+)"', lambda m: f'id="{p}-{m.group(1)}"', html)
    html = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{p}-{m.group(1)}"', html)
    return html


def fix_cross_links(html):
    html = re.sub(r'href="part(\d)\.html#([^"]+)"', lambda m: f'href="#p{m.group(1)}-{m.group(2)}"', html)
    html = re.sub(r'href="part(\d)\.html"', lambda m: f'href="#part{m.group(1)}"', html)
    return re.sub(r'href="index\.html"', 'href="#partindex"', html)


chunks = []
for f in FILES:
    tag = "index" if f == "index.html" else f[4]
    html = open(os.path.join(HERE, f), encoding="utf-8").read()
    body = fix_cross_links(prefix_ids(tag, body_of(html)))
    brk = '' if tag == "index" else '<br clear="all" style="page-break-before:always">'
    chunks.append(f'{brk}<a id="part{tag}"></a>{body}')
merged = "\n".join(chunks)

# render inline SVGs to PNG
os.makedirs(IMG_DIR, exist_ok=True)
svgs = re.findall(r"<svg\b.*?</svg>", merged, re.S)
print(f"{len(svgs)} inline SVG diagrams")
if svgs:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page(device_scale_factor=2)
        for i, s in enumerate(svgs):
            page.set_content(f"<html><body style='margin:0;background:#fff;color:#1d2228;font-family:Malgun Gothic'>"
                             f"<div id='w' style='display:inline-block;max-width:760px'>{s}</div></body></html>")
            page.locator("#w").screenshot(path=os.path.join(IMG_DIR, f"fig{i:03d}.png"))
        b.close()
    k = iter(range(len(svgs)))
    merged = re.sub(r"<svg\b.*?</svg>", lambda m: f'<img src="_word_img/fig{next(k):03d}.png" style="max-width:15cm">',
                    merged, flags=re.S)

open(OUT_HTML, "w", encoding="utf-8").write(
    f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>NAPS-625 여명 오버홀 매뉴얼</title>'
    f'<style>{CSS}</style></head><body>{merged}</body></html>')

# Word: open HTML, embed linked pictures, save as docx (wdFormatDocumentDefault = 16)
ps = f"""
$w = New-Object -ComObject Word.Application; $w.Visible = $false
$d = $w.Documents.Open('{OUT_HTML}')
foreach ($s in $d.InlineShapes) {{ if ($s.LinkFormat -ne $null) {{ $s.LinkFormat.SavePictureWithDocument = $true; $s.LinkFormat.BreakLink() }} }}
$d.SaveAs2('{OUT_DOCX}', 16)
$n = $d.ComputeStatistics(2)
$d.Close(); $w.Quit()
"pages: $n"
"""
r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[-800:])
print("saved", OUT_DOCX, os.path.exists(OUT_DOCX) and os.path.getsize(OUT_DOCX))
