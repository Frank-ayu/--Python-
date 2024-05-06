import tempfile
import os
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, LongTable, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from io import BytesIO


from MorphologicalCharacteristics import calculate_characters

def generate_related_report(path_original, single_image_path, double_image_path,
                            metrics_value,
                            me_value, diameter_value, area_value, perimeter_value,
                            pred_mask, mask_original,
                            pdf_save_path):
    pdfmetrics.registerFont(TTFont('SimSun', './SimSun.ttf'))  # 默认不支持中文，需要注册字体
    pdfmetrics.registerFont(TTFont('SimSunBd', './SimSun-Bold.ttf'))
    stylesheet = getSampleStyleSheet()  # 获取样式集

    # 获取reportlab自带样式
    Normal = stylesheet['Normal']
    BodyText = stylesheet['BodyText']
    Italic = stylesheet['Italic']
    Title = stylesheet['Title']
    Heading1 = stylesheet['Heading1']
    Heading2 = stylesheet['Heading2']
    Heading3 = stylesheet['Heading3']
    Heading4 = stylesheet['Heading4']
    Heading5 = stylesheet['Heading5']
    Heading6 = stylesheet['Heading6']
    Bullet = stylesheet['Bullet']
    Definition = stylesheet['Definition']
    Code = stylesheet['Code']

    # 自带样式不支持中文，需要设置中文字体，但有些样式会丢失，如斜体Italic。有待后续发现完全兼容的中文字体
    Normal.fontName = 'SimSun'
    Italic.fontName = 'SimSun'
    BodyText.fontName = 'SimSun'
    Title.fontName = 'SimSunBd'
    Heading1.fontName = 'SimSun'
    Heading2.fontName = 'SimSun'
    Heading3.fontName = 'SimSun'
    Heading4.fontName = 'SimSun'
    Heading5.fontName = 'SimSun'
    Heading6.fontName = 'SimSun'
    Bullet.fontName = 'SimSun'
    Definition.fontName = 'SimSun'
    Code.fontName = 'SimSun'

    # 创建一个自定义的段落样式 添加自定义样式
    stylesheet.add(
        ParagraphStyle(name='body',
                       fontName="SimSun",
                       fontSize=10,
                       textColor='black',
                       leading=20,  # 行间距
                       spaceBefore=0,  # 段前间距
                       spaceAfter=20,  # 段后间距
                       leftIndent=0,  # 左缩进
                       rightIndent=0,  # 右缩进
                       firstLineIndent=20,  # 首行缩进，每个汉字为10
                       alignment=TA_JUSTIFY,  # 对齐方式

                       # bulletFontSize=15,       #bullet为项目符号相关的设置
                       # bulletIndent=-50,
                       # bulletAnchor='start',
                       # bulletFontName='Symbol'
                       )
    )
    body = stylesheet['body']
    story = []
    # content1 = "<para align='center'><img src='../processed_data/reguge/test/image/T0001.png' width=100 height=100 valign='top'/><br/><br/><br/><br/></para>"
    content1 = "<para align='center'>Original image size 512*512<br/><img src=" + path_original + " width=100 height=100 valign='top'/><br/><br/><br/><br/></para>"
    # content2 = "<para align='center'><img src='../trainingrecords/pred_on_gui_refuge/refuge_unet_esa_grid_Lovasz/T0001-single-images.png' width=100 height=100 valign='top'/><br/><br/><br/><br/></para>"
    content2 = "<para align='center'>Red: Cup Green: Disc<br/><img src=" + single_image_path + " width=100 height=100 valign='top'/><br/><br/><br/><br/></para>"
    # content3 = "<para align='center'><img src='../trainingrecords/pred_on_gui_refuge/refuge_unet_esa_grid_Lovasz/T0001-double-images.png' width=200 height=100 valign='top'/><br/><br/><br/><br/></para>"
    content3 = "<para align='center'><img src=" + double_image_path + " width=200 height=100 valign='top'/><br/><br/><br/><br/></para>"

    content4 = ""

    content5 = ""

    Dice_Cup = str(metrics_value[0][1])[0:6]
    Dice_Disc = str(metrics_value[0][2])[0:6]
    IOU_Cup = str(metrics_value[1][1])[0:6]
    IOU_Disc = str(metrics_value[1][2])[0:6]
    PC_Cup = str(metrics_value[2][1])[0:6]
    PC_Disc = str(metrics_value[2][2])[0:6]
    F1_Cup  = str(metrics_value[3][1])[0:6]
    F1_Disc = str(metrics_value[3][2])[0:6]
    table_data_metrics = [['', 'Optic_Cup', 'Optic_Disc'],
                  ['Dice', Dice_Cup, Dice_Disc],
                  ['IOU', IOU_Cup, IOU_Disc],
                  ['Precision', PC_Cup, PC_Disc],
                  ['F1', F1_Cup, F1_Disc]
                  ]

    table_style_metrics = [
                    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),  # 字体
                    ('FONTSIZE', (0, 0), (-1, 0), 12),  # 第一行的字体大小
                    ('FONTSIZE', (0, 1), (-1, -1), 10),  # 第二行到最后一行的字体大小
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 所有表格左右中间对齐
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
                    ('GRID', (0, 0), (-1, -1), 0.1, colors.black),  # 设置表格框线为灰色，线宽为0.1
                    ]
    diameter_cup, area_cup, perimeter_cup, diameter_disc, area_disc, perimeter_disc = calculate_characters(pred_mask, mask_original)
    disc_diameter = str(diameter_disc)[0:8]
    disc_area = str(area_disc)
    disc_perimeter = str(perimeter_disc)[0:8]

    cup_diameter = str(diameter_cup)[0:8]
    cup_area = str(area_cup)
    cup_perimeter = str(perimeter_cup)[0:8]

    table_data_characteristics = [[''],
                                  ['Optic_Disc'],
                                  ['Optic_Cup']]

    if diameter_value:
        table_data_characteristics[0].append('Diameter')
        table_data_characteristics[1].append(disc_diameter)
        table_data_characteristics[2].append(cup_diameter)

    if area_value:
        table_data_characteristics[0].append('Area')
        table_data_characteristics[1].append(disc_area)
        table_data_characteristics[2].append(cup_area)

    if perimeter_value:
        table_data_characteristics[0].append('Perimeter')
        table_data_characteristics[1].append(disc_perimeter)
        table_data_characteristics[2].append(cup_perimeter)


    table_style_characteristics = [
                    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),  # 字体
                    ('FONTSIZE', (0, 0), (-1, 0), 12),  # 第一行的字体大小
                    ('FONTSIZE', (0, 1), (-1, -1), 10),  # 第二行到最后一行的字体大小
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 所有表格左右中间对齐
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
                    ('GRID', (0, 0), (-1, -1), 0.1, colors.black),  # 设置表格框线为灰色，线宽为0.1
                    ]
    table_metrics = Table(data=table_data_metrics, style=table_style_metrics, colWidths=90)
    # 以下删除
    table_characteristics = Table(data=table_data_characteristics, style=table_style_characteristics,
                                           colWidths=60)
    # 如果要控制间距 可以
    # story = [p1, Spacer(1, 12), p2]
    story.append(Paragraph("Report", Title))
    story.append(Paragraph("1.Origianl Image", Heading4))
    story.append(Paragraph(content1, body))
    story.append(Spacer(1, 2))
    story.append(Paragraph("2.Predict Result", Heading4))
    story.append(Paragraph(content2, body))
    story.append(Spacer(1, 2))
    story.append(Paragraph("3.GroundTruth And Predict Mask", Heading4))
    story.append(Paragraph(content3, body))
    story.append(Spacer(1, 2))
    if me_value:
        story.append(Paragraph("4.Table of Metrics", Heading4))
        story.append(Paragraph(content4, body))
        story.append(table_metrics)
    if diameter_value or area_value or perimeter_value:
        if me_value:
            story.append(Paragraph("5.Table of Characteristics", Heading4))
            story.append(Paragraph(content5, body))
            story.append(table_characteristics)
        else:
            story.append(Paragraph("4.Table of Characteristics", Heading5))
            story.append(Paragraph(content5, body))
            story.append(table_characteristics)

    image_name = single_image_path.split('/')[-1].split('-')[0]
    print("image_name:" + image_name)
    print("pdf_save_path:" + pdf_save_path)
    file_name = image_name + "-report"
    pdf_path = pdf_save_path + '/' + file_name + ".pdf"
    # 如果文件已存在则删除
    if os.path.exists(pdf_path):
        print("file exist, remove!")
        os.remove(pdf_path)

    doc = SimpleDocTemplate(pdf_path)
    doc.build(story)





