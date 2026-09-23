"""Render the editable coursework Markdown to a paginated Chinese PDF."""

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)


ROOT = Path(__file__).resolve().parents[1]


def font_name():
    font_file = Path('C:/Windows/Fonts/simhei.ttf')
    if font_file.exists():
        pdfmetrics.registerFont(TTFont('CourseChinese', str(font_file)))
        return 'CourseChinese'
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    return 'STSong-Light'


def styles(font):
    return {
        'title': ParagraphStyle('title', fontName=font, fontSize=19,
                                leading=27, alignment=TA_CENTER, spaceAfter=11),
        'byline': ParagraphStyle('byline', fontName=font, fontSize=10.5,
                                 leading=17, alignment=TA_CENTER, spaceAfter=19),
        'heading': ParagraphStyle('heading', fontName=font, fontSize=13,
                                  leading=19, spaceBefore=15, spaceAfter=8,
                                  keepWithNext=True),
        'body': ParagraphStyle('body', fontName=font, fontSize=10.3,
                               leading=17, alignment=TA_JUSTIFY, spaceAfter=8,
                               splitLongWords=False, wordWrap='CJK'),
        'table': ParagraphStyle('table', fontName=font, fontSize=8.2,
                                leading=12, wordWrap='CJK'),
    }


def render_text(value):
    escaped = html.escape(value).replace('**', '').replace('`', '')

    def make_link(match):
        candidate = match.group(0)
        url = candidate.rstrip('.,;。；')
        trailer = candidate[len(url):]
        return f'<link href="{url}" color="#166397">{url}</link>{trailer}'

    return re.sub(r'https?://[^\s<]+', make_link, escaped)


def build(source, output):
    font = font_name()
    style = styles(font)
    lines = Path(source).read_text(encoding='utf-8-sig').splitlines()
    if any('{{' in line or '}}' in line for line in lines):
        raise ValueError('Report still contains unfilled placeholders')
    flow = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [part.strip() for part in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{2,}:?', cell) for cell in cells):
                    rows.append(cells)
                i += 1
            if len({len(row) for row in rows}) != 1:
                raise ValueError('Inconsistent table columns')
            ncols = len(rows[0])
            usable = A4[0] - 44 * mm
            widths = ([usable * 0.19] + [usable * 0.81 / (ncols - 1)] * (ncols - 1)
                      if ncols > 1 else [usable])
            data = [[Paragraph(render_text(cell), style['table']) for cell in row]
                    for row in rows]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign='CENTER')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E7EEF6')),
                ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#C5CED7')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            flow.extend([table, Spacer(1, 10)])
            continue
        if line.startswith('# '):
            flow.append(Paragraph(render_text(line[2:]), style['title']))
        elif line.startswith('## '):
            flow.append(Paragraph(render_text(line[3:]), style['heading']))
        elif i == 2 and '组' in line:
            flow.append(Paragraph(render_text(line), style['byline']))
        else:
            flow.append(Paragraph(render_text(line), style['body']))
        i += 1

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(output), pagesize=A4, leftMargin=22 * mm,
                          rightMargin=22 * mm, topMargin=22 * mm,
                          bottomMargin=20 * mm)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor('#586777'))
        canvas.drawString(22 * mm, 12 * mm, '25 组 · RoTE / ReChorus CPU 复现')
        canvas.drawRightString(A4[0] - 22 * mm, 12 * mm, str(document.page))
        canvas.restoreState()

    doc.addPageTemplates(PageTemplate(id='course', frames=[frame], onPage=footer))
    doc.build(flow)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'report' / '报告.md')
    parser.add_argument('--output', type=Path, default=ROOT / 'report' / '报告.pdf')
    args = parser.parse_args()
    print(build(args.source, args.output))
