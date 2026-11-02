from pathlib import Path
import cairosvg

W, H = 1800, 1120
out = Path('/mnt/data')
svg = []
svg.append('''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="160mm" height="99.56mm" viewBox="0 0 1800 1120">
<defs>
  <marker id="arrowNavy" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#36566D"/></marker>
  <marker id="arrowTeal" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#2A8C82"/></marker>
  <marker id="arrowGray" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8A9AA5"/></marker>
  <filter id="softShadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#17324D" flood-opacity="0.08"/></filter>
  <style>
    text { font-family: 'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',sans-serif; fill:#20394B; }
    .title { font-size:34px; font-weight:700; letter-spacing:1px; }
    .subtitle { font-size:17px; fill:#657987; }
    .panelTitle { font-size:22px; font-weight:700; }
    .cardTitle { font-size:18px; font-weight:700; }
    .miniTitle { font-size:16px; font-weight:700; }
    .supportTitle { font-size:15px; font-weight:700; }
    .body { font-size:16px; fill:#435968; }
    .small { font-size:14px; fill:#667B88; }
    .tiny { font-size:13px; fill:#6F828D; }
    .white { fill:#FFFFFF; }
    .num { font-size:17px; font-weight:700; fill:#FFFFFF; }
  </style>
</defs>
<rect width="1800" height="1120" fill="#FFFFFF"/>
''')

def rect(x,y,w,h,fill='#FFFFFF',stroke='#D8E2E8',rx=16,sw=1.5,shadow=False):
    f = ' filter="url(#softShadow)"' if shadow else ''
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{f}/>')

def text(x,y,s,cls='body',anchor='start',fill=None,weight=None):
    attrs = f'class="{cls}" text-anchor="{anchor}"'
    if fill: attrs += f' fill="{fill}"'
    if weight: attrs += f' font-weight="{weight}"'
    svg.append(f'<text x="{x}" y="{y}" {attrs}>{s}</text>')

def multiline(x,y,lines,cls='body',line_h=25,anchor='start',fill=None):
    svg.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"'+(f' fill="{fill}"' if fill else '')+'>')
    for i,s in enumerate(lines):
        svg.append(f'<tspan x="{x}" dy="{0 if i==0 else line_h}">{s}</tspan>')
    svg.append('</text>')

def line(x1,y1,x2,y2,color='#36566D',width=3,dash=None,marker='arrowNavy'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = f' marker-end="url(#{marker})"' if marker else ''
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{d}{m}/>')

# Header
text(60,57,'项目总体研究框架','title')
text(62,91,'面向公众的装配式内装知识服务与交互智能体关键技术研究','subtitle')
svg.append('<line x1="60" y1="116" x2="1740" y2="116" stroke="#E6EDF1" stroke-width="2"/>')

# Main panel dimensions and headers
# left: challenges
rect(55,150,330,610,'#F7FAFC','#DCE6EB',18,1.5,True)
rect(55,150,330,62,'#EAF1F5','#EAF1F5',18,0)
text(80,190,'现实需求与关键瓶颈','panelTitle')
# problem cards
problems = [
    ('01','语言落差',['专业术语、标准条文难以','转化为居民听得懂的表达。'],'#477B9B'),
    ('02','适用性落差',['地域、时间、房屋类型与','改造条件影响政策和标准适用。'],'#477B9B'),
    ('03','可信度落差',['检索到相关内容不等于','证据充分，生成答案可能失真。'],'#477B9B'),
]
py = 235
for no, title, desc, col in problems:
    rect(78,py,284,142,'#FFFFFF','#DFE8ED',13,1.2)
    svg.append(f'<circle cx="108" cy="{py+36}" r="18" fill="{col}"/>')
    text(108,py+42,no,'num','middle')
    text(140,py+43,title,'cardTitle')
    multiline(98,py+83,desc,'body',27)
    py += 158

# center: research core
rect(425,150,925,610,'#FBFDFE','#D6E3E9',18,1.5,True)
rect(425,150,925,62,'#EAF4F3','#EAF4F3',18,0)
text(452,190,'课题总体目标与四项核心研究','panelTitle')
# goal banner
rect(455,232,865,82,'#254E68','#254E68',14,0)
text(887,266,'构建“可理解、可交互、可信赖”的装配式内装公众知识服务系统','white','middle',weight='700')
text(887,292,'形成多源知识治理、语义转换、可信判断与多轮交互相融合的方法体系','white','middle')
# cards 2x2
cards = [
    (455,338,'01','多源知识结构化治理与时空适用性管理',[
        '建立标准、政策、案例等多源知识元数据体系',
        '判定地域、时间、对象、版本及来源权威性']),
    (900,338,'02','公众生活语言与专业术语双向语义转换',[
        '识别咨询意图，归一口语表达并抽取关键条件',
        '将规范条文转化为通俗表达，保证语义不失真']),
    (455,542,'03','证据约束下的适用性判断与可信拒答',[
        '分层判断依据存在性、适用性与证据充分性',
        '形成有据回答、证据不足提示与无据拒答机制']),
    (900,542,'04','面向公众的多轮交互智能体与效果评价',[
        '实现上下文保持、主动澄清、任务引导与工具调用',
        '通过理解度、任务完成率和满意度验证服务效果']),
]
for x,y,no,title,desc in cards:
    rect(x,y,420,180,'#FFFFFF','#D9E5EA',13,1.4)
    svg.append(f'<circle cx="{x+31}" cy="{y+34}" r="17" fill="#2A8C82"/>')
    text(x+31,y+40,no,'num','middle')
    # title may be long, wrap to 2 lines in two cases
    if no in ('01','02'):
        # narrower font for header
        text(x+60,y+40,title,'cardTitle')
    else:
        text(x+60,y+40,title,'cardTitle')
    svg.append(f'<line x1="{x+22}" y1="{y+61}" x2="{x+398}" y2="{y+61}" stroke="#E7EEF1" stroke-width="1.3"/>')
    multiline(x+24,y+95,desc,'body',30)

# arrows from problem -> research, research -> outcomes
line(388,450,416,450,'#8A9AA5',3,None,'arrowGray')
line(1353,450,1380,450,'#8A9AA5',3,None,'arrowGray')

# right: outputs
rect(1390,150,355,610,'#F7FAFC','#DCE6EB',18,1.5,True)
rect(1390,150,355,62,'#EAF1F5','#EAF1F5',18,0)
text(1417,190,'预期成果与应用','panelTitle')
outs = [
    ('A','装配式内装公众知识库','多源、结构化、可追溯、可更新'),
    ('B','公众交互智能体原型','支持多轮咨询、主动澄清与可信回答'),
    ('C','知识服务与评测方法','形成可复现的指标、题集和评价规范'),
    ('D','试点应用与研究成果','用户评价、研究报告及相关软件成果'),
]
oy=235
for letter,title,desc in outs:
    rect(1415,oy,305,112,'#FFFFFF','#DFE8ED',13,1.2)
    rect(1432,oy+28,38,38,'#DDEDEA','#DDEDEA',10,0)
    text(1451,oy+54,letter,'cardTitle','middle',fill='#2A8179')
    text(1485,oy+43,title,'cardTitle')
    text(1485,oy+75,desc,'small')
    oy+=128

# Bottom bands: foundation, supports, evaluation loop
rect(55,795,520,238,'#F5F8FA','#D8E2E8',16,1.4)
text(82,835,'前期工程基础（已完成）','panelTitle')
text(82,868,'为本课题提供可复用的工程底座','small')
# 3 columns mini chips
foundation = [
    ('文档解析与入库','多格式解析、章节识别、OCR'),
    ('检索与答案生成','混合检索、重排、引用生成'),
    ('评测与系统运行','可复现评测、Web/API、660项测试'),
]
fx=80
for title,desc in foundation:
    rect(fx,895,150,108,'#FFFFFF','#DFE8ED',10,1.0)
    multiline(fx+12,929,[title],'miniTitle',22)
    multiline(fx+12,961,desc.split('、'),'tiny',20)
    fx += 162

rect(600,795,785,238,'#F5F8FA','#D8E2E8',16,1.4)
text(628,835,'三项技术支撑','panelTitle')
text(628,868,'保障核心研究可实现、可验证、可交付','small')
supports = [
    ('01','工程文档结构解析与切块',['章节树还原','OCR与结构化抽取']),
    ('02','混合检索与跨通道融合',['向量检索 + BM25','融合重排与候选优化']),
    ('03','评测体系与服务化交付',['指标评测与消融验证','接口、权限与部署交付']),
]
sx=625
for no,title,desc in supports:
    rect(sx,895,232,108,'#FFFFFF','#DFE8ED',10,1.0)
    svg.append(f'<circle cx="{sx+22}" cy="{918}" r="13" fill="#547C93"/>')
    text(sx+22,923,no,'tiny','middle',fill='#FFFFFF',weight='700')
    text(sx+43,924,title,'supportTitle')
    multiline(sx+16,958,desc,'tiny',20)
    sx += 246

rect(1410,795,335,238,'#F5F8FA','#D8E2E8',16,1.4)
text(1438,835,'评测与反馈闭环','panelTitle')
multiline(1438,875,['检索与引用质量','证据判断与拒答效果','多轮任务完成与公众体验'],'body',29)
# feedback loop arrow at bottom
line(1690,977,1690,1005,'#2A8C82',2,None,'arrowTeal')
text(1438,1015,'评价结果反馈至知识、算法与交互策略','tiny')

# Dashed connectors: foundation + support underpin main research
line(315,785,315,770,'#8A9AA5',2,'5 6','arrowGray')
line(990,785,990,770,'#8A9AA5',2,'5 6','arrowGray')
# Loop from feedback to core
svg.append('<path d="M1410 918 C1370 918,1360 780,1310 765" fill="none" stroke="#2A8C82" stroke-width="2.5" stroke-dasharray="7 6" marker-end="url(#arrowTeal)"/>')

# Main flow captions
text(220,746,'现实问题牵引','small','middle')
text(887,746,'核心研究形成服务能力','small','middle')
text(1567,746,'成果转化与示范应用','small','middle')

# Footer logic band
svg.append('<line x1="60" y1="1060" x2="1740" y2="1060" stroke="#E6EDF1" stroke-width="1.5"/>')
text(900,1092,'研究逻辑：问题牵引  →  核心技术研究  →  系统集成  →  公众试点验证  →  反馈优化','small','middle')
svg.append('</svg>')

svg_text='\n'.join(svg)
svg_path=out/'figure1_research_framework.svg'
svg_path.write_text(svg_text,encoding='utf-8')
png_path=out/'figure1_research_framework.png'
cairosvg.svg2png(bytestring=svg_text.encode('utf-8'),write_to=str(png_path),output_width=2700,output_height=1680)
print(svg_path)
print(png_path)
