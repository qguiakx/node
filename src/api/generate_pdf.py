from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json

router = APIRouter(
    prefix="/pdf",
    tags=["pdf"],
    responses={404: {"description": "Not found"}},
)

# 注册中文字体（确保字体文件存在，此处以 Windows 的微软雅黑为例，也可用其他支持中文的字体）
pdfmetrics.registerFont(TTFont('MicrosoftYaHei', 'msyh.ttc'))  # 或 'simsun.ttc' 等


# 如果使用 Linux，可下载思源黑体等开源字体

def build_pdf(data: dict) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()

    # 自定义支持中文的样式
    title_style = ParagraphStyle(
        'ChineseTitle',
        parent=styles['Title'],
        fontName='MicrosoftYaHei',
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        'ChineseHeading',
        parent=styles['Heading2'],
        fontName='MicrosoftYaHei',
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        'ChineseBody',
        parent=styles['Normal'],
        fontName='MicrosoftYaHei',
        fontSize=10,
        leading=14,
    )

    story = []

    for note in data.get('notes', []):
        # 笔记标题
        story.append(Paragraph(note['title'], title_style))
        story.append(Spacer(1, 6 * mm))

        # 标签（可选）
        if note.get('tags'):
            tags_text = ' | '.join(note['tags'])
            story.append(Paragraph(f"<b>标签：</b>{tags_text}", body_style))
            story.append(Spacer(1, 4 * mm))

        # 来源（可选）
        if note.get('source'):
            story.append(Paragraph(f"<b>来源：</b>{note['source']}", body_style))
            story.append(Spacer(1, 4 * mm))

        # 内容处理：将换行符转换为 <br/> 以便 Paragraph 识别
        content = note['content'].replace('\n', '<br/>')
        story.append(Paragraph(content, body_style))
        story.append(Spacer(1, 8 * mm))

    doc.build(story)
    buffer.seek(0)
    return buffer


@router.post("/generate-pdf")
async def generate_pdf(data: dict):
    pdf_buffer = build_pdf(data)
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=notes.pdf"}
    )