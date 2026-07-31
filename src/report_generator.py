import os
import json
import uuid
import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    ListFlowable,
    ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing


class ReportFooter:
    def __init__(self, report_id: str, timestamp: str):
        self.report_id = report_id
        self.timestamp = timestamp

    def __call__(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setStrokeColor(colors.lightgrey)
        canvas.line(doc.leftMargin, doc.bottomMargin - 10, doc.width + doc.leftMargin, doc.bottomMargin - 10)
        
        footer_text = f"Inspection ID: {self.report_id}  |  Generated: {self.timestamp}  |  Page {doc.page}"
        canvas.drawString(doc.leftMargin, doc.bottomMargin - 25, footer_text)
        canvas.restoreState()


def _create_qr_code(report_id: str) -> Drawing:
    qr_code = qr.QrCodeWidget(f"REPORT-ID:{report_id}")
    bounds = qr_code.getBounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    drawing = Drawing(1.5*inch, 1.5*inch, transform=[1.5*inch/width, 0, 0, 1.5*inch/height, 0, 0])
    drawing.add(qr_code)
    return drawing


def generate_report(
    image_path: str,
    heatmap_path: str,
    pipeline_output: Dict[str, Any],
    output_dir: str = "outputs/reports"
) -> tuple[str, str]:
    """
    Generates a professional industrial inspection PDF report and a JSON report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp_obj = datetime.datetime.now()
    timestamp_str = timestamp_obj.strftime("%Y_%m_%d_%H%M%S")
    display_time = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    report_id = str(uuid.uuid4()).split('-')[0].upper()
    
    pdf_filename = f"inspection_{timestamp_str}.pdf"
    json_filename = f"inspection_{timestamp_str}.json"
    
    pdf_path = os.path.join(output_dir, pdf_filename)
    json_path = os.path.join(output_dir, json_filename)
    
    # --- Generate JSON Report ---
    report_data = {
        "report_id": report_id,
        "timestamp": display_time,
        "pipeline_output": pipeline_output
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)
        
    # --- Generate PDF Report ---
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor("#1f497d"),
        spaceAfter=12
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor("#2e74b5"),
        spaceBefore=15,
        spaceAfter=10
    )
    normal_text = styles['Normal']
    normal_text.fontSize = 11
    
    card_style = ParagraphStyle(
        'CardStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.black,
        leading=16
    )
    
    elements = []
    
    # HEADER
    # Top table with Title and QR Code
    title_p = Paragraph("Industrial Inspection Report", title_style)
    qr_drawing = _create_qr_code(report_id)
    
    header_data = [
        [title_p, qr_drawing]
    ]
    header_table = Table(header_data, colWidths=[doc.width - 1.5*inch, 1.5*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT')
    ]))
    elements.append(header_table)
    
    # Meta Information
    meta_info = (
        f"<b>Generated Date:</b> {display_time}<br/>"
        f"<b>Inspection ID:</b> {report_id}<br/>"
    )
    elements.append(Paragraph(meta_info, normal_text))
    elements.append(Spacer(1, 15))
    
    # Overall Status Badge
    verification = pipeline_output.get("verification", {})
    vision_out = pipeline_output.get("vision_output", {})
    confidence = vision_out.get("confidence", 0.0)
    verified = verification.get("verified", False)
    
    if confidence > 0.85 and verified:
        status_text = "PASS"
        status_color = colors.HexColor("#28a745")
    elif confidence > 0.5:
        status_text = "WARNING"
        status_color = colors.HexColor("#ffc107")
    else:
        status_text = "FAIL"
        status_color = colors.HexColor("#dc3545")
        
    status_p = Paragraph(f"<font color='white'><b>OVERALL STATUS: {status_text}</b></font>", ParagraphStyle(
        'Status', fontSize=14, alignment=1, backColor=status_color, textColor=colors.white, spaceBefore=10, spaceAfter=10,
        borderPadding=8
    ))
    elements.append(status_p)
    elements.append(Spacer(1, 20))
    
    # SECTION 1 & 2: Images
    elements.append(Paragraph("1 & 2. Inspection Images (Original & AI Heatmap)", section_heading))
    
    img_data = []
    
    def _load_image(path, max_width, max_height):
        if not os.path.exists(path):
            return Paragraph(f"[Image not found: {os.path.basename(path)}]", normal_text)
        img = Image(path)
        img._restrictSize(max_width, max_height)
        return img
    
    orig_img = _load_image(image_path, doc.width/2 - 10, 3*inch)
    hm_img = _load_image(heatmap_path, doc.width/2 - 10, 3*inch)
    
    img_data = [
        [orig_img, hm_img],
        [Paragraph("Original Image", ParagraphStyle('c', alignment=1)), 
         Paragraph("GradCAM Heatmap", ParagraphStyle('c', alignment=1))]
    ]
    
    img_table = Table(img_data, colWidths=[doc.width/2, doc.width/2])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 20))
    
    # SECTION 3: Detection Summary
    elements.append(Paragraph("3. Detection Summary", section_heading))
    bbox = vision_out.get("bbox", [])
    bbox_str = ", ".join([f"{x:.2f}" for x in bbox]) if bbox else "N/A"
    
    summary_data = [
        ["Defect Type", vision_out.get("label", "N/A")],
        ["Confidence", f"{confidence*100:.1f}%"],
        ["Verification Score", f"{verification.get('confidence_score', 'N/A')}/100"],
        ["Bounding Box", bbox_str],
        ["Inspection Time", display_time]
    ]
    
    t_summary = Table(summary_data, colWidths=[2*inch, doc.width - 2*inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f4f4f4")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
    ]))
    elements.append(t_summary)
    
    elements.append(PageBreak())
    
    # SECTION 4: AI Findings
    elements.append(Paragraph("4. AI Findings", section_heading))
    findings = pipeline_output.get("findings", {})
    report_dict = pipeline_output.get("report", {})
    
    findings_html = (
        f"<b>Summary:</b> {findings.get('summary', 'N/A')}<br/><br/>"
        f"<b>Impression:</b> {report_dict.get('impression', 'N/A')}<br/><br/>"
        f"<b>Root Cause:</b> {report_dict.get('root_cause', 'N/A')}"
    )
    
    # Card format via a 1x1 table
    card_table = Table([[Paragraph(findings_html, card_style)]], colWidths=[doc.width])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#fafafa")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#cccccc")),
        ('PADDING', (0,0), (0,0), 12)
    ]))
    elements.append(card_table)
    elements.append(Spacer(1, 15))
    
    # SECTION 5: Supporting Evidence
    elements.append(Paragraph("5. Supporting Evidence", section_heading))
    evidence_list = report_dict.get("supporting_evidence", [])
    if evidence_list:
        bullet_items = [ListItem(Paragraph(ev, normal_text), bulletColor=colors.black) for ev in evidence_list]
        elements.append(ListFlowable(bullet_items, bulletType='bullet', start='circle'))
    else:
        elements.append(Paragraph("No supporting evidence provided.", normal_text))
        
    elements.append(Spacer(1, 15))
    
    # SECTION 6: Recommended Actions
    elements.append(Paragraph("6. Recommended Actions", section_heading))
    actions = report_dict.get("recommended_next_steps", [])
    if actions:
        action_data = [["Action Item", "Priority", "Status"]]
        for idx, act in enumerate(actions):
            priority = "High" if idx == 0 else ("Medium" if idx == 1 else "Low")
            action_data.append([Paragraph(act, normal_text), priority, "[  ]"])
            
        t_actions = Table(action_data, colWidths=[doc.width - 2.5*inch, 1.25*inch, 1.25*inch])
        t_actions.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2e74b5")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 8)
        ]))
        elements.append(t_actions)
    else:
        elements.append(Paragraph("No recommended actions.", normal_text))
    
    elements.append(Spacer(1, 20))
    
    # SECTION 7: Verification
    elements.append(Paragraph("7. Verification & Audit", section_heading))
    v_score = verification.get("confidence_score", 0)
    flagged = verification.get("flagged_claims", [])
    
    verif_data = [
        ["Verified by System", "Yes" if verified else "No"],
        ["Confidence Score", f"{v_score}/100"],
        ["Flagged Claims", ", ".join(flagged) if flagged else "None"]
    ]
    t_verif = Table(verif_data, colWidths=[2.5*inch, doc.width - 2.5*inch])
    t_verif.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    elements.append(t_verif)
    
    elements.append(Spacer(1, 20))
    
    # SECTION 8: Overall Risk Gauge
    elements.append(Paragraph("8. Overall Risk Gauge", section_heading))
    
    risk_level = "LOW"
    risk_color = colors.HexColor("#28a745")
    if not verified or confidence < 0.6:
        risk_level = "CRITICAL"
        risk_color = colors.HexColor("#dc3545")
    elif confidence < 0.8:
        risk_level = "HIGH"
        risk_color = colors.HexColor("#ff851b")
    elif confidence < 0.9:
        risk_level = "MEDIUM"
        risk_color = colors.HexColor("#ffc107")
        
    risk_p = Paragraph(f"<font color='white'><b>RISK LEVEL: {risk_level}</b></font>", ParagraphStyle(
        'Risk', fontSize=12, alignment=1, backColor=risk_color, textColor=colors.white, borderPadding=6
    ))
    elements.append(risk_p)
    elements.append(Spacer(1, 40))
    
    # SECTION 9: Inspector Approval
    elements.append(Paragraph("9. Inspector Approval", section_heading))
    
    approval_text = (
        "By signing below, the inspector confirms review of the AI findings and authorizes "
        "any recommended maintenance actions to proceed."
    )
    elements.append(Paragraph(approval_text, normal_text))
    elements.append(Spacer(1, 30))
    
    sig_data = [
        ["Inspector Name:", "___________________________", "Date:", "___________________"],
        ["Signature:", "___________________________", "Comments:", "___________________"]
    ]
    
    t_sig = Table(sig_data, colWidths=[1.2*inch, 2.3*inch, 1*inch, 2*inch])
    t_sig.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('PADDING', (0,0), (-1,-1), 10)
    ]))
    elements.append(t_sig)
    
    # Build PDF
    footer_handler = ReportFooter(report_id, display_time)
    doc.build(elements, onFirstPage=footer_handler, onLaterPages=footer_handler)
    
    return pdf_path, json_path


if __name__ == "__main__":
    # Test execution
    dummy_output = {
        "vision_output": {
            "label": "Gear Tooth Crack",
            "confidence": 0.934,
            "bbox": [0.41, 0.32, 0.22, 0.18],
            "heatmap_overlay_path": "outputs/heatmap.png"
        },
        "findings": {
            "summary": "High intensity crack detected on gear tooth."
        },
        "report": {
            "impression": "Industrial crack detected",
            "root_cause": "Overheating during heat treatment",
            "supporting_evidence": [
                "Heatmap focuses on cracked region",
                "Similarity search matched historical Case C-2331",
                "Literature indicates thermal fatigue"
            ],
            "recommended_next_steps": [
                "Replace affected gear",
                "Inspect remaining gears",
                "Reduce heat treatment temperature",
                "Schedule preventive maintenance"
            ]
        },
        "verification": {
            "verified": True,
            "confidence_score": 92,
            "flagged_claims": []
        }
    }
    # For testing, provide paths to dummy images if they exist, or non-existent paths to test fallback
    gen_pdf, gen_json = generate_report("dummy_original.jpg", "dummy_heatmap.png", dummy_output)
    print(f"Generated PDF: {gen_pdf}")
    print(f"Generated JSON: {gen_json}")
