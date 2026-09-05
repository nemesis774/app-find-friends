#!/usr/bin/env python3
"""Wspólny plan tygodnia: Mieszko (5b), Gucio (kl. 1), Gaja (przedszkole).

Plan rysowany jest na ciągłej osi czasu (8:00–21:00) – każdy blok ma wysokość
proporcjonalną do czasu trwania, dzięki czemu zajęcia o nieregularnych godzinach
(15:15, 16:50, 19:40 …) trafiają dokładnie tam, gdzie powinny.
Uruchomienie: python3 generuj_plan.py  ->  plan_lekcji.html + plan_lekcji.pdf
"""
import html
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
DAYS = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek"]
KIDS = [("m", "Mieszko", "5b"), ("g", "Gucio", "kl. 1"), ("j", "Gaja", "przedszkole")]

DAY_START, DAY_END = "8:00", "21:00"

# dzwonki – lekcja nr -> (od, do)
BELLS = {1: ("8:00", "8:45"), 2: ("8:55", "9:40"), 3: ("9:50", "10:35"), 4: ("10:45", "11:30"),
         5: ("11:50", "12:35"), 6: ("12:45", "13:30"), 7: ("13:40", "14:25")}

# ---------------------------------------------------------------- lekcje szkolne
# lista (przedmiot, sala) – pozycja na liście = numer lekcji
MIESZKO = {
    "Poniedziałek": [("Godz. wych.", "s. 12"), ("Język polski", "s. 12"), ("WF", ""),
                     ("Historia", "s. 6"), ("Język angielski", "s. 0"), ("Matematyka", "s. 0")],
    "Wtorek": [("Logopedia", "od 15.09"), ("Informatyka", "s. 7"), ("Język polski", "czyt."),
               ("Język polski", "s. 10"), ("Matematyka", "s. 6")],
    "Środa": [("WF", ""), ("Matematyka", "s. 7"), ("Edukacja zdrowotna", ""),
              ("Historia", "s. 11"), ("Język angielski", "s. 13")],
    "Czwartek": [("Język polski", "s. 13"), ("Geografia", "s. 13"), ("Plastyka", "s. 18a"),
                 ("Matematyka", "s. 15"), ("Technika", "s. 10"), ("Religia", "s. 6")],
    "Piątek": [("Biologia", "s. 10"), ("Język polski", "s. 18a"), ("Muzyka", "czyt."),
               ("Język angielski", "s. 13"), ("WF", ""), ("WF", ""), ("TUS", "p. A. Dąbrowska")],
}
EW = "Edukacja wczesnoszkolna"
GUCIO = {
    "Poniedziałek": [(EW, ""), (EW, ""), (EW, "")],
    "Wtorek": [(EW, ""), (EW, ""), ("WF", ""), (EW, ""), ("Język angielski", "s. 3")],
    "Środa": [(EW, ""), (EW, ""), (EW, ""), (EW, ""), ("Język angielski", "")],
    "Czwartek": [("Informatyka", ""), (EW, ""), (EW, ""), (EW, ""), ("Religia", "s. 4")],
    "Piątek": [(EW, ""), (EW, ""), (EW, ""), ("WF", "")],
}

# ---------------------------------------------------------------- zajęcia dodatkowe
# (dziecko, dzień, od, do, nazwa, uwaga)
EXTRA = [
    ("j", d, "8:00", "15:30", "Przedszkole", "") for d in DAYS
] + [
    ("j", "Wtorek", "17:00", "18:00", "Tańce", ""),
    ("j", "Środa", "16:00", "16:40", "Taekwondo", ""),
    ("j", "Piątek", "16:00", "16:40", "Taekwondo", ""),
    ("j", "Piątek", "15:00", "16:00", "Basen", "co 2 tyg."),
    ("g", "Wtorek", "15:15", "16:00", "Język angielski", ""),
    ("g", "Czwartek", "15:15", "16:00", "Język angielski", ""),
    ("g", "Środa", "16:50", "17:35", "Taekwondo", ""),
    ("g", "Piątek", "16:50", "17:35", "Taekwondo", ""),
    ("m", "Wtorek", "17:10", "18:05", "Język angielski", ""),
    ("m", "Piątek", "17:10", "18:05", "Język angielski", ""),
    ("m", "Środa", "19:40", "20:40", "Taekwondo", ""),
    ("m", "Piątek", "19:40", "21:00", "Taekwondo", ""),
]

COLORS = {
    EW: "#fff1b8", "Język polski": "#d6e8ff", "Matematyka": "#ffd9d9", "Język angielski": "#d4f5dc",
    "WF": "#ffe0c2", "Historia": "#e8dcff", "Informatyka": "#d9f2f7", "Religia": "#f3e9d8",
    "Geografia": "#d9f0e3", "Biologia": "#dff5c8", "Plastyka": "#ffd9f0", "Muzyka": "#f0dcff",
    "Technika": "#e4e4e4", "Edukacja zdrowotna": "#e0f4f0", "Godz. wych.": "#eeeeee",
    "Logopedia": "#f5f5dc", "TUS": "#f5f5dc",
    "Przedszkole": "#fbe4ef", "Taekwondo": "#ffc9c9", "Tańce": "#ffd6f5", "Basen": "#c9ecff",
}
KID_COLORS = {"m": "#dbe7f7", "g": "#fdebc8", "j": "#f6d9e6"}

# ---------------------------------------------------------------- geometria
BODY_H_MM = 165.0                      # wysokość osi czasu na stronie


def mins(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


T0, T1 = mins(DAY_START), mins(DAY_END)
MM_PER_MIN = BODY_H_MM / (T1 - T0)


def pos(start, end):
    top = (mins(start) - T0) * MM_PER_MIN
    height = (mins(end) - mins(start)) * MM_PER_MIN
    return f"top:{top:.2f}mm;height:{height:.2f}mm"


def merged_lessons(lessons):
    """Łączy sąsiednie identyczne przedmioty w jeden blok; zwraca (od, do, przedmiot, sale)."""
    out, i = [], 0
    while i < len(lessons):
        subj, room = lessons[i]
        rooms = [room] if room else []
        j = i + 1
        while j < len(lessons) and lessons[j][0] == subj:
            if lessons[j][1] and lessons[j][1] not in rooms:
                rooms.append(lessons[j][1])
            j += 1
        out.append((BELLS[i + 1][0], BELLS[j][1], subj, " / ".join(rooms)))
        i = j
    return out


def block(start, end, name, sub, kind):
    color = COLORS.get(name, "#f0f0f0")
    label = html.escape(name).replace("wczesnoszkolna", "wczesno&shy;szkolna")
    sub_html = f'<span class="sub">{html.escape(sub)}</span>' if sub else ""
    return (f'<div class="blk {kind}" style="{pos(start, end)};background:{color}">'
            f'<span class="name">{label}</span>{sub_html}</div>')


def column(kid, day):
    parts = []
    lessons = {"m": MIESZKO, "g": GUCIO}.get(kid, {}).get(day, [])
    for start, end, subj, rooms in merged_lessons(lessons):
        parts.append(block(start, end, subj, rooms, "lesson"))
    for k, d, start, end, name, note in EXTRA:
        if k == kid and d == day:
            sub = f"{start}–{end}" + (f" · {note}" if note else "")
            kind = "extra whole" if name == "Przedszkole" else "extra"
            parts.append(block(start, end, name, sub, kind))
    return f'<div class="col {kid}">{"".join(parts)}</div>'


def axis():
    labels = []
    t = T0
    while t <= T1:
        top = (t - T0) * MM_PER_MIN
        cls = "h" if t % 60 == 0 else "hh"
        cls += " first" if t == T0 else (" last" if t == T1 else "")
        labels.append(f'<span class="tick {cls}" style="top:{top:.2f}mm">{t // 60}:{t % 60:02d}</span>')
        t += 30
    return f'<div class="axis">{"".join(labels)}</div>'


def build_html():
    hour_mm = 60 * MM_PER_MIN
    head = "".join(
        f'<div class="day"><div class="dname">{d}</div><div class="kids">'
        + "".join(f'<div class="kid {k}">{n} <small>{c}</small></div>' for k, n, c in KIDS)
        + "</div></div>"
        for d in DAYS
    )
    body = "".join(
        f'<div class="daycols">' + "".join(column(k, d) for k, _, _ in KIDS) + "</div>"
        for d in DAYS
    )
    legend_school = [EW, "Język polski", "Matematyka", "Język angielski", "WF", "Historia",
                     "Informatyka", "Religia", "Edukacja zdrowotna", "Logopedia", "TUS"]
    legend_extra = ["Przedszkole", "Język angielski", "Taekwondo", "Tańce", "Basen"]

    def lg(items, extra=False):
        return "".join(
            f'<span class="lg"><i class="{"ex" if extra else ""}" style="background:{COLORS[s]}"></i>{html.escape(s)}</span>'
            for s in items)

    return f"""<!DOCTYPE html>
<html lang="pl"><head><meta charset="utf-8">
<title>Plan tygodnia – Mieszko, Gucio, Gaja</title>
<style>
@page {{ size: A4 landscape; margin: 5mm 7mm; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{ font-family: "Liberation Sans", "DejaVu Sans", Arial, sans-serif; color: #222; font-size: 8pt; }}
header {{ display: flex; align-items: baseline; justify-content: space-between; margin: 0 0 1.6mm; }}
h1 {{ font-size: 13.5pt; margin: 0; }}
h1 span {{ font-weight: normal; color: #666; font-size: 9.5pt; margin-left: 5px; }}
.note {{ font-size: 7.6pt; color: #555; }}

.grid {{ display: grid; grid-template-columns: 13mm repeat(5, 1fr); border: 1px solid #9a9a9a; }}
.corner {{ background: #2f3e55; border-right: 1px solid #9a9a9a; }}
.day {{ border-left: 1.5px solid #2f3e55; }}
.day:first-of-type {{ border-left: none; }}
.dname {{ background: #2f3e55; color: #fff; font-weight: bold; font-size: 10.5pt; text-align: center; padding: 1.3mm 0; }}
.kids {{ display: grid; grid-template-columns: repeat(3, 1fr); }}
.kid {{ font-weight: bold; font-size: 7.6pt; text-align: center; padding: .9mm 0; border-left: 1px solid #c8c8c8; border-bottom: 1px solid #9a9a9a; line-height: 1.15; }}
.kid:first-child {{ border-left: none; }}
.kid small {{ display: block; font-weight: normal; color: #555; font-size: 6.2pt; }}
.kid.m {{ background: {KID_COLORS['m']}; }} .kid.g {{ background: {KID_COLORS['g']}; }} .kid.j {{ background: {KID_COLORS['j']}; }}

.axis {{ position: relative; height: {BODY_H_MM}mm; border-right: 1px solid #9a9a9a; background: #f1f3f6; }}
.tick {{ position: absolute; right: 1.2mm; transform: translateY(-50%); font-size: 7pt; color: #333; }}
.tick.h {{ font-weight: bold; }}
.tick.first {{ transform: translateY(0); }}
.tick.last {{ transform: translateY(-100%); }}
.tick.hh {{ font-size: 6pt; color: #888; }}
.daycols {{ display: grid; grid-template-columns: repeat(3, 1fr); height: {BODY_H_MM}mm; border-left: 1.5px solid #2f3e55;
  background-image:
    repeating-linear-gradient(to bottom, #b5b5b5 0, #b5b5b5 0.25mm, transparent 0.25mm, transparent {hour_mm:.3f}mm),
    repeating-linear-gradient(to bottom, transparent 0, transparent {hour_mm/2:.3f}mm, #e2e2e2 {hour_mm/2:.3f}mm, #e2e2e2 {hour_mm/2+0.2:.3f}mm, transparent {hour_mm/2+0.2:.3f}mm, transparent {hour_mm:.3f}mm);
}}
.daycols:first-of-type {{ border-left: none; }}
.col {{ position: relative; border-left: 1px solid #d0d0d0; }}
.col:first-child {{ border-left: none; }}

.blk {{ position: absolute; left: 0.35mm; right: 0.35mm; border: 1px solid rgba(0,0,0,.18); border-radius: .7mm;
  display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
  padding: 0 .6mm; overflow: hidden; line-height: 1.1; }}
.blk .name {{ font-weight: bold; font-size: 7.2pt; hyphens: manual; }}
.blk .sub {{ font-size: 5.9pt; color: #444; margin-top: .3mm; }}
.blk.extra {{ border: 1.2px solid rgba(0,0,0,.45); }}
.blk.extra .name {{ font-size: 7.4pt; }}
.blk.whole {{ border-style: dashed; justify-content: flex-start; padding-top: 2.5mm; }}
.blk.whole .name {{ font-size: 8pt; }}

footer {{ display: flex; justify-content: space-between; align-items: flex-start; margin-top: 1.6mm; font-size: 7pt; color: #444; gap: 4mm; }}
.legend {{ display: flex; flex-wrap: wrap; gap: .8mm 3.2mm; align-items: center; }}
.legend b {{ color: #2f3e55; margin-right: .5mm; }}
.lg i {{ display: inline-block; width: 3mm; height: 2.6mm; border: 1px solid #aaa; vertical-align: -0.4mm; margin-right: 1mm; border-radius: .4mm; }}
.lg i.ex {{ border: 1.2px solid #444; }}
.hint {{ white-space: nowrap; text-align: right; }}
</style></head>
<body>
<header>
  <h1>Plan tygodnia <span>Mieszko (5b) · Gucio (klasa 1) · Gaja (przedszkole) · rok szkolny 2026/2027</span></h1>
  <div class="note">Oś czasu 8:00–21:00, linie co 30 min. Białe pola = wolne, można dopisać ręcznie.</div>
</header>
<div class="grid">
  <div class="corner"></div>
  {head}
  {axis()}
  {body}
</div>
<footer>
  <div class="legend"><b>Szkoła:</b>{lg(legend_school)}</div>
  <div class="legend"><b>Zajęcia dodatkowe:</b>{lg(legend_extra, True)}</div>
  <div class="hint">Basen Gai (pt. 15:00–16:00) – co 2 tygodnie<br>Logopedia Mieszka – od 15.09</div>
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
