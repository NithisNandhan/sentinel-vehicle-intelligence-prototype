"""Render the evidence-bound handoff Markdown as a readable PDF."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Preformatted

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/PROJECT_STATUS_AND_NEXT_STEPS.md'
OUTPUT = ROOT / 'output/pdf/Sentinel_Project_Status_and_Next_Steps.pdf'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

body = ParagraphStyle('Body', fontName='Helvetica', fontSize=11, leading=15, spaceAfter=8, allowWidows=0, allowOrphans=0)
title = ParagraphStyle('Title', parent=body, fontName='Helvetica-Bold', fontSize=23, leading=28, spaceAfter=16)
h1 = ParagraphStyle('Heading1', parent=body, fontName='Helvetica-Bold', fontSize=19, leading=23, spaceAfter=13, keepWithNext=True)
h2 = ParagraphStyle('Heading2', parent=body, fontName='Helvetica-Bold', fontSize=12.5, leading=17, spaceBefore=8, spaceAfter=7, keepWithNext=True)
bullet = ParagraphStyle('Bullet', parent=body, leftIndent=12, firstLineIndent=-9, spaceAfter=5)
cell = ParagraphStyle('Cell', parent=body, fontSize=9.5, leading=12.5, spaceAfter=0)
head = ParagraphStyle('Head', parent=cell, fontName='Helvetica-Bold', textColor=colors.white)
code = ParagraphStyle('Code', parent=body, fontName='Courier', fontSize=8.4, leading=11.5, spaceBefore=3, spaceAfter=9)

def rich(text):
    out = escape(text)
    return re.sub(r'(https://[^\s<>]+)', lambda m: '<link href="'+m.group(1)+'" color="#1F497D">'+m.group(1)+'</link>', out)

def table(rows):
    width=504
    widths=[width*.29, width*.27, width*.44]
    if rows[0][0]=='Layer': widths=[width*.20,width*.38,width*.42]
    if rows[0][0]=='Bundle': widths=[width*.25,width*.12,width*.63]
    if rows[0][0]=='Still diagnostic': widths=[width*.50,width*.25,width*.25]
    if rows[0][0]=='Check': widths=[width*.27,width*.32,width*.41]
    data=[[Paragraph(rich(c), head if i==0 else cell) for c in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17324D')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F5F8')]),
        ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#D9D9D9')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    return t

lines=SOURCE.read_text(encoding='utf-8').splitlines()
story=[]
i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:
        i+=1;continue
    if line=='<!-- pagebreak -->':
        story.append(Spacer(1,12));i+=1;continue
    if line.startswith('# '):
        story.append(Paragraph(rich(line[2:]),title if not story else h1));i+=1;continue
    if line.startswith('## '):
        story.append(Paragraph(rich(line[3:]),h2));i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\-\s]+',c) for c in row): rows.append(row)
            i+=1
        rendered_table=table(rows)
        story.extend([KeepTogether([rendered_table]) if len(rows)<=5 else rendered_table,Spacer(1,9)]);continue
    if line.startswith('```'):
        i+=1;cl=[]
        while i<len(lines) and not lines[i].startswith('```'):
            # Wrap shell lines with PowerShell continuation, not truncation.
            raw=lines[i]
            while len(raw)>83 and ' ' in raw[:83]:
                split=raw.rfind(' ',0,83)
                cl.append(raw[:split]+' `');raw='  '+raw[split+1:]
            cl.append(raw);i+=1
        story.append(Preformatted('\n'.join(cl),code));i+=1;continue
    if line.startswith('- '):
        story.append(Paragraph('- '+rich(line[2:]),bullet));i+=1;continue
    parts=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','- ','|','```','<!--')):
        parts.append(lines[i].strip());i+=1
    story.append(Paragraph(rich(' '.join(parts)),body))

def footer(canvas,doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor('#596674'))
    canvas.setFont('Helvetica',8.5)
    canvas.drawString(54,35,'Sentinel Vehicle Intelligence  |  Status 8 October 2026')
    canvas.drawRightString(558,35,f'Page {doc.page}')
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUTPUT),pagesize=letter,rightMargin=54,leftMargin=54,topMargin=48,bottomMargin=56,
    title='Sentinel Vehicle Intelligence Project Status and Completion Handoff',author='Sentinel project team')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUTPUT)
