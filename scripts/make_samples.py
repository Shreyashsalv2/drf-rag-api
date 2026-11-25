"""Build the sample documents (PDF, DOCX, Markdown) from the structured GDPR text in data/gdpr.json.

Usage: python scripts/make_samples.py
Each article starts with a heading line "Article N - Title" so evaluation can tell which article a chunk holds.
"""

import json
import re
from pathlib import Path

import docx
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"

FILES = [
    ("gdpr-ch01-02-principles.pdf", "GDPR Chapters I-II: General provisions and principles", range(1, 12)),
    ("gdpr-ch03-data-subject-rights.pdf", "GDPR Chapter III: Rights of the data subject", range(12, 24)),
    ("gdpr-ch04-controller-processor.docx", "GDPR Chapter IV: Controller and processor", range(24, 44)),
    ("gdpr-ch05-international-transfers.pdf", "GDPR Chapter V: Transfers to third countries", range(44, 51)),
    ("gdpr-ch08-remedies-penalties.md", "GDPR Chapter VIII: Remedies, liability and penalties", range(77, 85)),
]
SOURCE_NOTE = (
    "Text: Regulation (EU) 2016/679 (GDPR), from the Structured GDPR JSON dataset by Guilherme Silveira, "
    "CC BY 4.0, https://github.com/gbasilveira/GDPR-JSON. Not legal advice."
)


def point_lines(node: dict, prefix: str = "") -> list[str]:
    lines = []
    if node.get("content"):
        lines.append(f"{prefix}{node['content'].strip()}")
    for sub in node.get("sub_points") or []:
        lines.extend(point_lines(sub, f"({sub['sub_point_number']}) " if sub.get("sub_point_number") else ""))
    if node.get("additional_content"):
        lines.append(node["additional_content"].strip())
    return lines


def articles() -> dict[int, tuple[str, list[str]]]:
    data = json.loads((ROOT / "data" / "gdpr.json").read_text())
    out = {}
    for chapter in data["enabling terms"]:
        for art in chapter.get("articles") or []:
            number = int(re.search(r"\d+", art["article_number"]).group())
            paragraphs = []
            for p in art.get("paragraphs") or []:
                lines = point_lines(p)
                if lines and p.get("paragraph_number"):
                    lines[0] = f"{p['paragraph_number']}. {lines[0]}"
                paragraphs.append(lines)
            for d in art.get("definitions") or []:
                paragraphs.append([f"({d['definition_number']}) {d['content'].strip()}"])
            out[number] = (art["article_title"].strip(), paragraphs)
    return out


def latin1(text: str) -> str:
    table = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", "€": "EUR", " ": " "}
    text = "".join(table.get(c, c) for c in text)
    return text.encode("latin-1", "replace").decode("latin-1")


def write_pdf(path: Path, title: str, arts: dict, numbers) -> None:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, latin1(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for n in numbers:
        name, paragraphs = arts[n]
        pdf.set_font("Helvetica", "B", 11.5)
        pdf.multi_cell(0, 6.5, latin1(f"Article {n} - {name}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for lines in paragraphs:
            for line in lines:
                pdf.multi_cell(0, 5, latin1(line), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, latin1(SOURCE_NOTE), new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def write_docx(path: Path, title: str, arts: dict, numbers) -> None:
    document = docx.Document()
    document.core_properties.title = title
    document.add_heading(title, level=1)
    for n in numbers:
        name, paragraphs = arts[n]
        document.add_paragraph(f"Article {n} - {name}", style="Heading 2")
        for lines in paragraphs:
            for line in lines:
                document.add_paragraph(line)
    document.add_paragraph(SOURCE_NOTE)
    document.core_properties.author = "GDPR sample"
    document.save(str(path))


def write_md(path: Path, title: str, arts: dict, numbers) -> None:
    parts = [f"# {title}", ""]
    for n in numbers:
        name, paragraphs = arts[n]
        parts += [f"Article {n} - {name}", ""]
        for lines in paragraphs:
            parts += lines + [""]
    parts.append(f"_{SOURCE_NOTE}_")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    arts = articles()
    for filename, title, numbers in FILES:
        path = OUT / filename
        {"pdf": write_pdf, "docx": write_docx, "md": write_md}[path.suffix[1:]](path, title, arts, numbers)
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
