#!/usr/bin/env python3
"""Generuje wspólny plan lekcji Mieszka (5b) i Gucia (klasa 1) jako HTML + PDF."""
import html
import subprocess
from pathlib import Path

HERE = Path(__file__).parent

SLOTS = [
    (1, "8:00", "8:45"),
    (2, "8:55", "9:40"),
    (3, "9:50", "10:35"),
    (4, "10:45", "11:30"),
    (5, "11:50", "12:35"),
    (6, "12:45", "13:30"),
    (7, "13:40", "14:25"),
]

# popołudnie: co 30 minut od pełnej godziny, ostatnie zajęcia o 19:30
AFTERNOON = [f"{h}:{m:02d}" for h in range(14, 20) for m in (0, 30) if (h, m) >= (14, 30)]

DAYS = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek"]

# (przedmiot, sala/uwaga) – indeks listy = numer lekcji - 1
MIESZKO = {
    "Poniedziałek": [
        ("Godz. wychowawcza", "s. 12"),
        ("Język polski", "s. 12"),
        ("WF", ""),
        ("Historia", "s. 6"),
        ("Język angielski", "s. 0"),
        ("Matematyka", "s. 0"),
    ],
    "Wtorek": [
        ("Logopedia", "od 15.09"),
        ("Informatyka", "s. 7"),
        ("Język polski", "czytelnia"),
        ("Język polski", "s. 10"),
        ("Matematyka", "s. 6"),
    ],
    "Środa": [
        ("WF", ""),
        ("Matematyka", "s. 7"),
        ("Edukacja zdrowotna", ""),
        ("Historia", "s. 11"),
        ("Język angielski", "s. 13"),
    ],
    "Czwartek": [
        ("Język polski", "s. 13"),
        ("Geografia", "s. 13"),
        ("Plastyka", "s. 18a"),
        ("Matematyka", "s. 15"),
        ("Technika", "s. 10"),
        ("Religia", "s. 6"),
    ],
    "Piątek": [
        ("Biologia", "s. 10"),
        ("Język polski", "s. 18a"),
        ("Muzyka", "czytelnia"),
        ("Język angielski", "s. 13"),
        ("WF", ""),
        ("WF", ""),
        ("TUS", "p. A. Dąbrowska"),
    ],
}

EW = "Edukacja wczesnoszkolna"
GUCIO = {
    "Poniedziałek": [(EW, ""), (EW, ""), (EW, "")],
    "Wtorek": [(EW, ""), (EW, ""), ("WF", ""), (EW, ""), ("Język angielski", "s. 3")],
    "Środa": [(EW, ""), (EW, ""), (EW, ""), (EW, ""), ("Język angielski", "")],
    "Czwartek": [("Informatyka", ""), (EW, ""), (EW, ""), (EW, ""), ("Religia", "s. 4")],
    "Piątek": [(EW, ""), (EW, ""), (EW, ""), ("WF", "")],
}

# kolory przedmiotów (wspólne dla obu chłopców)
COLORS = {
    "Edukacja wczesnoszkolna": "#fff1b8",
    "Język polski": "#d6e8ff",
    "Matematyka": "#ffd9d9",
    "Język angielski": "#d4f5dc",
    "WF": "#ffe0c2",
    "Historia": "#e8dcff",
    "Informatyka": "#d9f2f7",
    "Religia": "#f3e9d8",
    "Geografia": "#d9f0e3",
    "Biologia": "#dff5c8",
    "Plastyka": "#ffd9f0",
    "Muzyka": "#f0dcff",
    "Technika": "#e4e4e4",
    "Edukacja zdrowotna": "#e0f4f0",
    "Godz. wychowawcza": "#eeeeee",
    "Logopedia": "#f5f5dc",
    "TUS": "#f5f5dc",
}


def merged(lessons):
    """Łączy sąsiednie identyczne przedmioty w jedną komórkę (rowspan)."""
    cells = []  # (start_index, span, subject, rooms)
    i = 0
    while i < len(lessons):
        subj, room = lessons[i]
        j = i + 1
        rooms = [room] if room else []
        while j < len(lessons) and lessons[j][0] == subj:
            if lessons[j][1] and lessons[j][1] not in rooms:
                rooms.append(lessons[j][1])
            j += 1
        cells.append((i, j - i, subj, " / ".join(rooms)))
        i = j
    return cells


def end_time(lessons):
    return SLOTS[len(lessons) - 1][2] if lessons else "—"


def build_column(lessons, who):
    """Zwraca słownik: indeks slotu -> HTML komórki (lub None gdy objęta rowspanem)."""
    out = {}
    for start, span, subj, room in merged(lessons):
        color = COLORS.get(subj, "#f0f0f0")
        room_html = f'<span class="room">{html.escape(room)}</span>' if room else ""
        out[start] = (
            f'<td class="lesson {who}" rowspan="{span}" style="background:{color}">'
            f'<span class="subj">{html.escape(subj)}</span>{room_html}</td>'
        )
        for k in range(start + 1, start + span):
            out[k] = None
    for k in range(len(lessons), len(SLOTS)):
        out[k] = f'<td class="free {who}"></td>'
    return out


def build_html():
    cols = {d: (build_column(MIESZKO[d], "m"), build_column(GUCIO[d], "g")) for d in DAYS}

    head_days = "".join(
        f'<th class="day" colspan="2">{d}</th>' for d in DAYS
    )
    head_kids = "".join(
        '<th class="kid m">Mieszko <small>5b</small></th>'
        '<th class="kid g">Gucio <small>kl. 1</small></th>'
        for _ in DAYS
    )
    head_end = "".join(
        f'<th class="end m">lekcje do {end_time(MIESZKO[d])}</th>'
        f'<th class="end g">lekcje do {end_time(GUCIO[d])}</th>'
        for d in DAYS
    )

    rows = []
    for idx, (nr, start, stop) in enumerate(SLOTS):
        tds = []
        for d in DAYS:
            for col in cols[d]:
                cell = col[idx]
                if cell is not None:
                    tds.append(cell)
        rows.append(
            f'<tr><th class="time"><span class="nr">{nr}</span>'
            f'<span class="hrs">{start}<br>{stop}</span></th>{"".join(tds)}</tr>'
        )
    for i, t in enumerate(AFTERNOON):
        cls = "pm first" if i == 0 else "pm"
        tds = "".join(f'<td class="free {who} {cls}"></td>' for _ in DAYS for who in ("m", "g"))
        rows.append(f'<tr class="{cls}"><th class="time {cls}"><span class="pmt">{t}</span></th>{tds}</tr>')

    legend_items = [
        ("Edukacja wczesnoszkolna", "EW"),
        ("Język polski", ""), ("Matematyka", ""), ("Język angielski", ""),
        ("WF", "wychowanie fizyczne"), ("Historia", ""), ("Informatyka", ""),
        ("Religia", ""), ("Edukacja zdrowotna", ""),
    ]
    legend = "".join(
        f'<span class="lg"><i style="background:{COLORS[s]}"></i>{html.escape(s)}'
        + (f' <small>({html.escape(n)})</small>' if n else "")
        + "</span>"
        for s, n in legend_items
    )

    return f"""<!DOCTYPE html>
<html lang="pl"><head><meta charset="utf-8">
<title>Plan lekcji – Mieszko i Gucio</title>
<style>
@page {{ size: A4 landscape; margin: 5mm 7mm; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{ font-family: "Liberation Sans", "DejaVu Sans", Arial, sans-serif; color: #222; font-size: 8.4pt; }}
header {{ display: flex; align-items: baseline; justify-content: space-between; margin: 0 0 2mm; }}
h1 {{ font-size: 14pt; margin: 0; letter-spacing: .2px; }}
h1 span {{ font-weight: normal; color: #666; font-size: 10pt; margin-left: 6px; }}
.note {{ font-size: 8pt; color: #555; }}
.note b {{ color: #2f3e55; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
th, td {{ border: 1px solid #b9b9b9; padding: 0; vertical-align: middle; text-align: center; }}
th.day {{ background: #2f3e55; color: #fff; font-size: 11pt; padding: 1.6mm 0; }}
th.kid {{ font-size: 8.5pt; padding: 1mm 0; }}
th.kid small {{ font-weight: normal; color: #555; }}
th.kid.m {{ background: #dbe7f7; }}
th.kid.g {{ background: #fdebc8; }}
th.end {{ font-size: 6.6pt; font-weight: normal; color: #444; padding: .6mm 0; background: #f6f6f6; }}
th.time {{ width: 14mm; background: #f1f3f6; padding: 0; }}
th.time .nr {{ display: block; font-size: 10.5pt; }}
th.time .hrs {{ display: block; font-size: 6.6pt; font-weight: normal; color: #555; line-height: 1.2; }}
td {{ height: 11.6mm; }}
tr.pm td {{ height: 7.4mm; }}
td.lesson .subj {{ display: block; font-weight: bold; font-size: 8.4pt; line-height: 1.15; padding: 0 1mm; }}
td.lesson .room {{ display: block; font-size: 6.6pt; color: #555; margin-top: .5mm; }}
td.free {{ background: #fff; border-style: dashed; border-color: #cfcfcf; }}
tr.pm td.free {{ border-style: solid; border-color: #d6d6d6; }}
tr.pm th.time {{ background: #f7f8fa; }}
th.time .pmt {{ display: block; font-size: 8pt; font-weight: bold; color: #333; }}
.first {{ border-top: 2px solid #2f3e55 !important; }}
td.free.m {{ border-right-style: solid; }}
footer {{ display: flex; justify-content: space-between; align-items: center; margin-top: 1.8mm; font-size: 7.4pt; color: #444; gap: 4mm; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1.2mm 4mm; }}
.lg i {{ display: inline-block; width: 3mm; height: 3mm; border: 1px solid #aaa; vertical-align: -0.5mm; margin-right: 1.2mm; }}
.lg small {{ color: #777; }}
.free-hint {{ white-space: nowrap; }}
.free-hint i {{ display: inline-block; width: 5mm; height: 3mm; border: 1px dashed #bbb; vertical-align: -0.5mm; margin-right: 1.2mm; }}
</style></head>
<body>
<header>
  <h1>Plan lekcji <span>Mieszko (5b) &amp; Gucio (klasa 1) · rok szkolny 2026/2027</span></h1>
  <div class="note">Po lekcjach siatka co 30 min (14:30–19:30) na <b>zajęcia dodatkowe</b> – do wpisania ręcznie</div>
</header>
<table>
  <thead>
    <tr><th class="time" rowspan="3">Lekcja</th>{head_days}</tr>
    <tr>{head_kids}</tr>
    <tr>{head_end}</tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
<footer>
  <div class="legend">{legend}
    <span class="lg"><i style="background:{COLORS['Logopedia']}"></i>Logopedia <small>(Mieszko, wt. 8:00, od 15.09)</small></span>
    <span class="lg"><i style="background:{COLORS['TUS']}"></i>TUS <small>(Mieszko, pt. 13:40, p. A. Dąbrowska)</small></span>
  </div>
  <div class="free-hint"><i></i>zajęcia dodatkowe – do uzupełnienia</div>
</footer>
</body></html>
"""


def main():
    html_path = HERE / "plan_lekcji.html"
    pdf_path = HERE / "plan_lekcji.pdf"
    html_path.write_text(build_html(), encoding="utf-8")
    chrome = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    subprocess.run(
        [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
         "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True,
    )
    print(f"Zapisano: {html_path}\nZapisano: {pdf_path}")


if __name__ == "__main__":
    main()
