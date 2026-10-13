"""Regenerate Korean portfolio SVGs from published measurements, using only Python.

Usage: python render_visuals.py --input visual-data.json --output figures
Input accepts {"projects": [...]}, a project list, or one project object.
The script reads no models, original datasets, network or source-path fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

W = 1200
FONT = "'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"


def attrs(**values):
    return ' '.join(f'{key.replace("_", "-")}="{escape(str(value), {chr(34): "&quot;"})}"' for key, value in values.items() if value is not None)


class SVG:
    def __init__(self, project, title, description, height=800):
        self.project = project
        self.palette = project['theme']
        self.height = height
        self.dark = project['site_id'] == 'project-03'
        self.ink = self.palette['ink']
        self.muted = '#B2C4CC' if self.dark else '#64696B'
        self.line_color = '#2B414C' if self.dark else '#DDDDE0'
        self.panel_color = '#182A35' if self.dark else '#FFFFFF'
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{escape(title)}</title>', f'<desc id="desc">{escape(description)}</desc>', f'<style>text{{font-family:{FONT};font-variant-numeric:tabular-nums}} .axis{{fill:{self.muted}}}</style>']
        self.rect(0, 0, W, height, self.palette['background'])

    def text(self, x, y, value, size=26, color=None, weight=400, anchor='start', **extra):
        self.parts.append(f'<text {attrs(x=x, y=y, fill=color or self.ink, font_size=size, font_weight=weight, text_anchor=anchor, **extra)}>{escape(str(value))}</text>')

    def lines(self, x, y, values, size=25, line_height=38, **extra):
        for i, value in enumerate(values):
            self.text(x, y + i * line_height, value, size=size, **extra)

    def rect(self, x, y, width, height, fill, rx=0, stroke=None, stroke_width=1, opacity=None):
        self.parts.append(f'<rect {attrs(x=x, y=y, width=width, height=height, fill=fill, rx=rx, stroke=stroke, stroke_width=stroke_width, opacity=opacity)}/>')

    def line(self, x1, y1, x2, y2, color=None, width=2, dash=None, opacity=None):
        self.parts.append(f'<line {attrs(x1=x1, y1=y1, x2=x2, y2=y2, stroke=color or self.line_color, stroke_width=width, stroke_dasharray=dash, opacity=opacity)}/>')

    def circle(self, x, y, r, fill, stroke=None, width=1, opacity=None):
        self.parts.append(f'<circle {attrs(cx=x, cy=y, r=r, fill=fill, stroke=stroke, stroke_width=width, opacity=opacity)}/>')

    def path(self, points, color, width=3, dash=None, fill='none', opacity=None):
        d = ' '.join(('M' if i == 0 else 'L') + f'{x:.3f},{y:.3f}' for i, (x, y) in enumerate(points))
        self.parts.append(f'<path {attrs(d=d, fill=fill, stroke=color, stroke_width=width, stroke_dasharray=dash, stroke_linecap='round', stroke_linejoin='round', opacity=opacity)}/>')

    def panel(self, x, y, width, height, fill=None):
        self.rect(x, y, width, height, fill or self.panel_color, rx=20, stroke=self.line_color)

    def pill(self, x, y, width, value, color, size=23):
        self.rect(x, y - 31, width, 44, color, rx=12)
        self.text(x + width / 2, y, value, size=size, color='#FFFFFF', weight=700, anchor='middle')

    def header(self, title, subtitle):
        self.text(48, 64, title, size=36, weight=750)
        self.text(48, 104, subtitle, size=22, color=self.muted)

    def footer(self, lines):
        y = self.height - 31 - max(0, len(lines) - 1) * 29
        for line in lines:
            self.text(48, y, line, size=20, color=self.muted)
            y += 29

    def legend(self, entries, x=50, y=151, size=23):
        for label, color, width in entries:
            self.rect(x, y - 16, 18, 18, color, rx=4)
            self.text(x + 29, y, label, size=size)
            x += width

    def arrow(self, x1, y1, x2, y2, color, width=4):
        self.line(x1, y1, x2, y2, color, width)
        angle = math.atan2(y2-y1, x2-x1)
        points = [(x2-17*math.cos(angle-.45), y2-17*math.sin(angle-.45)), (x2,y2), (x2-17*math.cos(angle+.45),y2-17*math.sin(angle+.45))]
        self.path(points, color, width)

    def finish(self, input_hash):
        self.parts.append('<metadata>' + escape(json.dumps({'project_id': self.project['site_id'], 'visual_data_sha256': input_hash, 'generated_from_published_measurements': True}, ensure_ascii=False)) + '</metadata>')
        return '\n'.join(self.parts) + '\n</svg>\n'


def get_metric(p, key):
    return next(item for item in p['metrics'] if item['metric'] == key)


def percent_grid(svg, x0, x1, top, bottom, ticks=(0,20,40,60,80,100), max_value=100):
    for tick in ticks:
        x = x0 + tick / max_value * (x1-x0)
        svg.line(x, top, x, bottom, width=1)
        svg.text(x, bottom+33, str(tick), size=21, color=svg.muted, anchor='middle')


def paper_thumbnail(p):
    s=SVG(p, '근거 기반 논문 질의응답', '논문 원문을 검색하고 답변 문장과 근거 위치를 연결하는 구현 흐름입니다. 논문 30편을 평가했으며 답변 정확도를 뜻하는 그림은 아닙니다.', 760)
    pal=p['theme']
    s.text(56, 80, '논문 질의응답과 원문 근거 추적', 48, weight=750)
    s.text(56, 131, '논문 질의응답과 원문 추적', 32, color=s.muted)
    s.panel(57, 185, 330, 390)
    s.rect(86, 211, 268, 338, '#FBFAF5', 8, pal['secondary'], 2)
    s.text(112, 259, '논문 원문', 35, weight=750)
    for y,width in [(295,200),(321,186),(347,202),(412,192),(438,204),(464,160),(500,183)]:
        s.line(111,y,111+width,y,'#C3C9C3',4)
    s.rect(105, 363, 217, 27, '#D2E5DC', 4)
    s.text(114,385,'문장과 페이지',22,color=pal['primary'],weight=700)
    s.arrow(386, 350, 511, 350, pal['primary'], 6)
    s.text(451, 316, '검색', 32, anchor='middle', weight=700)
    s.panel(532, 208, 606, 168)
    s.text(562, 260, '찾은 근거', 37, weight=750, color=pal['primary'])
    s.text(562, 309, '문장  →  페이지  →  출처', 32)
    s.arrow(829, 387, 829, 425, pal['secondary'], 5)
    s.panel(532, 445, 606, 154)
    s.text(562, 496, '답변과 원문 대조', 37, weight=750)
    s.text(562, 547, '한국어 질문과 영어 논문을 연결', 31, color=s.muted)
    s.rect(55,635,1088,75,pal['ink'],16)
    s.text(84,685,'30편 평가',38,color='#FFFFFF',weight=750)
    s.text(391,683,'원문 문장과 답변을 함께 확인',32,color='#FFFFFF')
    return s


def paper_results(p):
    s=SVG(p,'논문 질의응답의 자동 진단 결과','평가 논문 20편에서 같은 의미 그룹을 초기 및 최종 구현으로 비교했습니다. 응답 여부와 어휘 표지, 인용 위치 진단이며 답변 정확도가 아닙니다.',900)
    pal=p['theme'];s.header('답변과 원문의 자동 대조 결과','평가 논문 20편의 같은 질문 비교. 아래 수치는 답변 정확도가 아닌 자동 진단입니다.')
    s.legend([('초기 개발 버전',pal['before'],250),('최종 개발 버전',pal['after'],260)])
    s.pill(811,154,340,'출력 형식 준수 240 / 240',pal['ink'],21)
    x0=385;x1=995
    percent_grid(s,x0,x1,193,755)
    for i,m in enumerate(p['metrics'][1:6]):
        y=219+i*108
        labels={'answerable_retained':['답변 가능한 질문에','응답'],'all_fact_aliases':['필수 정보의 어휘 표지를','모두 포함'],'acceptable_source_page_hit':['정답 근거 위치 중','하나를 인용'],'unsupported_refused':['근거가 없는 요청을','거절'],'false_premise_refused':['잘못된 전제의 요청을','거절']}
        s.lines(48,y+13,labels[m['metric']],size=24,line_height=34)
        for j,key in enumerate(['before','after']):
            value=m[key];denom=m['denominator']; bar_y=y+j*39
            s.rect(x0,bar_y-21,max(2,value/denom*(x1-x0)),27,pal[key],5)
            s.text(1038,bar_y+2,f'{value} / {denom}',25,weight=700)
    s.text(1120,789,'비율 (%)',21,anchor='end',color=s.muted)
    s.footer(['영어와 한국어가 모두 조건을 만족해야 의미 그룹이 통과합니다.', '어휘 표지와 인용 위치는 사실 정확성을 보증하지 않습니다. 전체 답변 품질 향상은 확인하지 못했습니다.'])
    return s


def trajectory_curve(s,p,x,y,width,height,show_axis=True):
    points=p['trajectory']['points']; xs=[a['x_km'] for a in points];zs=[a['altitude_km'] for a in points]
    maxx=max(xs)*1.14; maxz=max(zs)*1.14
    tx=lambda value:x+value/maxx*width
    tz=lambda value:y+height-value/maxz*height
    if show_axis:
        for fraction in [0,.25,.5,.75,1]:
            yy=y+height-fraction*height;s.line(x,yy,x+width,yy,width=1)
            s.text(x-15,yy+8,f'{fraction*maxz:.0f}',22,anchor='end',color=s.muted)
        for fraction in [0,.25,.5,.75,1]:
            xx=x+fraction*width;s.line(xx,y,xx,y+height,width=1)
            s.text(xx,y+height+36,f'{fraction*maxx:.1f}',22,anchor='middle',color=s.muted)
    before=[a for a in points if a['time_s'] <= p['trajectory']['separation_time_s']]
    after=[a for a in points if a['time_s'] >= p['trajectory']['separation_time_s']]
    sep=(tx(p['trajectory']['separation_x_km']),tz(p['trajectory']['separation_altitude_km']))
    s.path([(tx(a['x_km']),tz(a['altitude_km'])) for a in before]+[sep],p['theme']['secondary'],5)
    s.path([sep]+[(tx(a['x_km']),tz(a['altitude_km'])) for a in after],p['theme']['primary'],5)
    s.circle(*sep,8,p['theme']['secondary'],s.ink,2)
    launch=(tx(points[0]['x_km']),tz(points[0]['altitude_km']))
    s.circle(*launch,9,s.ink)
    return {'separation':sep,'launch':launch,'max_x':maxx,'max_z':maxz}


def dm_thumbnail(p):
    s=SVG(p,'DM100 로켓 발사와 회수','보존된 무풍 대표 비행의 실제 계산 궤적을 보여 줍니다. 19개 상태 변수를 사용하는 2차원 개념 시뮬레이터입니다.',760)
    pal=p['theme'];s.text(54,79,'DM100',60,weight=750)
    s.text(54,145,'로켓 발사와 회수',35,color=s.muted)
    s.text(54,205,'19개 상태로 계산한',32)
    s.text(54,249,'발사부터 첫 접지까지',32)
    s.rect(53,280,310,108,'#233D4B',16)
    s.text(76,324,'100톤 로켓',39,weight=700)
    s.text(76,367,'2차원 개념 시뮬레이션',29,color=s.muted)
    g=trajectory_curve(s,p,475,165,630,407,True)
    sx,sy=g['separation'];s.text(sx-14,sy-26,'단 분리',32,color=pal['secondary'],anchor='end',weight=750)
    s.text(995,211,'귀환',33,color=pal['primary'],weight=700)
    s.text(1105,653,'수평 거리 (km)',24,color=s.muted,anchor='end')
    s.text(445,143,'고도 (km)',27,color=s.muted)
    s.text(492,548,'발사와 착륙',28,weight=700)
    s.rect(53,665,1090,57,'#233D4B',14)
    s.text(78,705,'저장된 비행 기록의 궤적',31,color=s.muted)
    s.text(1120,705,'실물 비행 성능과 구분',29,anchor='end',color=s.muted)
    return s


def dm_trajectory(p):
    s=SVG(p,'저장된 대표 비행의 궤적','보존된 무풍 대표 비행의 수평 거리와 고도입니다. 새 센서 조건 평가의 통계가 아니며 실제 로켓 궤적을 측정한 것이 아닙니다.',820)
    s.header('저장된 대표 비행의 궤적','무풍 조건의 시뮬레이션 기록을 시각화했습니다. 실제 비행 계측이나 새 센서 평가 통계가 아닙니다.')
    s.legend([('상승',p['theme']['secondary'],150),('1단 귀환',p['theme']['primary'],210)],y=153)
    g=trajectory_curve(s,p,120,228,970,420,True)
    sx,sy=g['separation'];s.text(sx+19,sy+10,'단 분리',26,color=p['theme']['secondary'],weight=700)
    s.text(48,193,'고도 (km)',24,color=s.muted)
    s.text(1098,730,'수평 거리 (km)',24,anchor='end',color=s.muted)
    s.footer(['총질량 100톤의 2차원 개념 모델입니다. 2단 궤도 진입이나 실제 기체 안전 인증은 포함하지 않습니다.'])
    return s


def dm_validation(p):
    s=SVG(p,'목적이 다른 세 평가 집단의 통과 결과','기존 조건 회귀 검증 286개, 새로운 복합 조건 96개와 센서 확장 조건 48개는 서로 다른 집단입니다. 모든 실행이 선언한 합성 모델의 기준을 통과했습니다.',620)
    s.header('평가 집단별 기준 충족 결과','같은 100%라도 집단의 조건과 분모가 다릅니다. 서로 합쳐 하나의 최종 성공률로 표시하지 않습니다.')
    labels=[['기존 조건','회귀 검증'],['새로운','복합 조건'],['센서 오차와','자세 추정기']]
    colors=['#8FA6B1',p['theme']['secondary'],p['theme']['primary']]
    for i,g in enumerate(p['groups']):
        x=48+i*376;s.panel(x,157,352,337)
        s.lines(x+26,209,labels[i],size=28,line_height=39,weight=700)
        s.text(x+176,358,f"{g['passed']}/{g['count']}",60,color=colors[i],weight=750,anchor='middle')
        s.text(x+176,403,'기준 통과 / 실행',23,color=s.muted,anchor='middle')
        s.rect(x+26,442,300,10,'#263D49',5)
        s.rect(x+26,442,300*g['passed']/g['count'],10,colors[i],5)
        s.text(x+325,475,f"{g['passed']/g['count']*100:.0f}%",23,anchor='end',color=colors[i],weight=700)
    s.footer(['모든 수치는 선언된 합성 조건에서의 관측 결과입니다. 연속 입력 전체의 성공이나 실제 기체 안전성을 보장하지 않습니다.'])
    return s


def dm_propellant(p):
    fuel=p['paired_fuel'];s=SVG(p,'동일 센서 조건의 RCS 추진제 소모량','같은 입력과 시드를 사용하고 두 처리 방식이 모두 통과한 48쌍을 비교했습니다. 기존 처리 중앙값은 120.977kg이며 자세 추정 적용 중앙값은 99.122kg입니다.',850)
    pal=p['theme'];s.header('동일 조건의 RCS 추진제 소모량 비교','같은 입력과 시드의 48쌍. 모든 점은 두 방식이 모두 성공한 비행 한 쌍입니다.')
    x0,y0,side=120,182,478;limit=180
    for tick in [0,30,60,90,120,150,180]:
        xx=x0+tick/limit*side;yy=y0+side-tick/limit*side
        s.line(xx,y0,xx,y0+side,width=1);s.line(x0,yy,x0+side,yy,width=1)
        s.text(xx,y0+side+31,str(tick),21,anchor='middle',color=s.muted)
        s.text(x0-14,yy+7,str(tick),21,anchor='end',color=s.muted)
    s.line(x0,y0+side,x0+side,y0,'#7E949E',2,'8 7')
    for before,after in zip(fuel['raw_used_kg'],fuel['filtered_used_kg']):
        color=pal['primary'] if after<before else pal['warning']
        s.circle(x0+before/limit*side,y0+side-after/limit*side,6,color,opacity=.9)
    s.text(48,151,'자세 추정 적용 (kg)',23,color=pal['primary'])
    s.text(x0+side/2,y0+side+78,'기존 처리 (kg)',24,anchor='middle',color=pal['secondary'])
    s.panel(675,174,472,522)
    s.text(702,221,'RCS 소모량 중앙값',29,weight=700)
    s.text(704,275,'기존 처리',23,color=s.muted)
    s.text(704,329,f"{fuel['raw_median_kg']:.3f} kg",42,color=pal['secondary'],weight=750)
    s.text(704,389,'자세 추정 적용',23,color=s.muted)
    s.text(704,443,f"{fuel['filtered_median_kg']:.3f} kg",42,color=pal['primary'],weight=750)
    s.line(704,471,1117,471)
    s.text(704,537,f"{fuel['median_consumption_reduction_pct']:.2f}% 감소",45,color=pal['primary'],weight=750)
    s.text(704,583,f"48쌍 중 {fuel['lower_consumption_pairs']}쌍에서 소모량 감소",24)
    s.lines(704,632,['대각선 아래: 추정 적용 후 감소','대각선 위: 추정 적용 후 증가'],size=22,line_height=30,color=s.muted)
    s.footer(['18.07%는 두 중앙값의 비율입니다. 각 비행의 절감률 평균이나 실물 로켓의 절감률이 아닙니다.', '선언한 합성 센서 범위의 비교입니다. 모든 착륙 지표가 개선된 것은 아닙니다.'])
    return s


def rag_thumbnail(p):
    s=SVG(p,'한국어 검색과 질의응답','한국어 질문에서 근거 문단을 검색하고 원문에서 답을 추출합니다. 같은 501문항의 정답 일치율은 62.28%에서 73.25%로 변했습니다.',760)
    pal=p['theme'];s.text(56,83,'한국어 근거 검색과 답변 추출',49,weight=750)
    s.text(56,135,'한국어 검색과 추출형 질의응답',32,color=s.muted)
    s.rect(54,185,1090,100,pal['ink'],18)
    s.text(81,244,'질문',35,color=pal['secondary'],weight=700)
    s.text(215,244,'필요한 정보를 원문에서 찾아 답하다',32,color='#FFFFFF')
    s.panel(56,324,482,258)
    s.text(82,373,'근거 검색',35,weight=700)
    for y,width in [(420,405),(458,370),(496,390)]:
        s.rect(83,y-14,width,15,'#DDE5EF',6)
    s.rect(80,519,406,29,'#F0DC9B',6)
    s.text(92,542,'정답 구간을 포함한 문단',23,weight=700)
    s.arrow(558,446,625,446,pal['primary'],6)
    s.panel(649,324,496,258)
    s.text(677,373,'답 추출',35,weight=700)
    s.text(677,438,'원문에서 답을 선택',32,color=pal['primary'],weight=700)
    s.text(677,496,'근거가 부족하면 보류',30,color=s.muted)
    s.line(677,538,1110,538,pal['secondary'],5)
    m=get_metric(p,'exact_match')
    s.text(55,642,'같은 501문항의 정답 일치율',31,color=s.muted)
    s.text(55,713,f"{m['before']:.2f}%",49,color=pal['before'],weight=700)
    s.arrow(313,696,365,696,pal['primary'],5)
    s.text(394,713,f"{m['after']:.2f}%",49,color=pal['primary'],weight=750)
    s.text(1123,707,'정답 55개 증가',34,anchor='end',weight=700)
    return s


def rag_results(p):
    s=SVG(p,'한국어 검색과 질의응답의 대응 비교','같은 답변 가능한 501문항에서 초기 및 최종 모델의 정답 일치율, 한국어 문자 F1과 상위5개 검색 재현율을 비교했습니다.',690)
    pal=p['theme'];s.header('검색 성능과 답변 품질의 비교 결과','최초 개발 모델과 최종 개발 모델을 같은 답변 가능한 501문항에서 비교했습니다.')
    s.legend([('최초 개발 모델',pal['before'],245),('최종 개발 모델',pal['after'],245)],y=156)
    x0,x1=338,969;percent_grid(s,x0,x1,197,536)
    labels={'exact_match':['정답 일치율','EM'],'character_f1':['한국어 문자 F1'],'recall_at_5':['상위 5개 근거 검색','재현율']}
    for i,key in enumerate(['exact_match','character_f1','recall_at_5']):
        m=get_metric(p,key);y=225+i*110
        s.lines(48,y+10,labels[key],size=26,line_height=35,weight=700)
        for j,role in enumerate(['before','after']):
            yy=y+j*40;v=m[role]
            s.rect(x0,yy-22,v/100*(x1-x0),28,pal[role],5)
            s.text(1003,yy+2,f'{v:.2f}%',27,weight=700)
    s.text(1138,570,'지표 (%)',21,color=s.muted,anchor='end')
    s.footer(['정답 일치율: 312 / 501 → 367 / 501. 검색 재현율: 정답 구간 전체를 포함하는 근거를 찾은 비율입니다.', '공개 KorQuAD 검증 자료의 프로젝트 평가입니다. 공식 비공개 시험 성적이나 의미상 정확도는 아닙니다.'])
    return s


def rag_tradeoffs(p):
    s=SVG(p,'답변 개선과 함께 증가한 실행 비용과 잘못된 답변','같은 GPU 실행 경로의 전체581문항 평균 처리 시간이 증가했고 별도 합성 무응답80문항의 잘못된 답변도1개에서4개로 증가했습니다. 두 패널의 단위와 분모는 다릅니다.',710)
    pal=p['theme'];s.header('답변 개선에 따른 실행 비용과 오답 변화','정답은 늘었지만 처리 시간과 근거 없는 질문의 잘못된 답변도 증가했습니다.')
    specs=[(48,get_metric(p,'mean_latency_ms'),'평균 GPU 처리 시간','전체 581문항 / 단위 ms',240),(636,get_metric(p,'false_answer'),'근거 없는 질문의 오답','별도 합성 80문항 / 단위 건',5)]
    for x,m,title,subtitle,maxv in specs:
        s.panel(x,153,516,438);s.text(x+25,199,title,28,weight=700);s.text(x+25,233,subtitle,22,color=s.muted)
        for i,role in enumerate(['before','after']):
            yy=299+i*126;v=m[role]
            s.text(x+28,yy,'최초 개발' if role=='before' else '최종 개발',23,color=s.muted)
            s.text(x+480,yy,(f'{v:.2f} ms' if m['unit']=='ms' else f'{int(v)} / 80'),31,weight=700,anchor='end',color=pal[role])
            s.rect(x+28,yy+30,452,30,'#EBEDF0',7)
            s.rect(x+28,yy+30,v/maxv*452,30,pal[role],7)
        s.text(x+28,550,(f"{m['after']/m['before']:.2f}배의 처리 시간" if m['unit']=='ms' else '잘못된 답변 3건 증가'),27,weight=700,color=pal['warning'])
    s.footer(['GPU 처리 시간은 모델 로딩을 제외합니다. 사용자 CPU의 실행 시간과 구분합니다.', '합성 무응답 문항은 통제 표본이며 실제 모든 질문의 오류율로 일반화할 수 없습니다.'])
    return s


def kd_thumbnail(p):
    s=SVG(p,'지식 증류를 통한 모델 경량화','큰 교사 모델의 확률 분포를 작은 학생에 전달합니다. 대표 학생의 참조 정확도는87.06%이며 파라미터가 약98.5% 적습니다.',760)
    pal=p['theme'];s.text(56,84,'지식 증류를 통한 모델 경량화',47,weight=750)
    s.text(56,136,'교통 표지판 인식 모델 경량화',32,color=s.muted)
    s.panel(55,192,470,380)
    s.text(84,246,'교사 모델',37,color=pal['teacher'],weight=750)
    for row in range(3):
        for col in range(5):
            s.circle(118+col*79,316+row*64,20,pal['teacher'],opacity=.9)
    s.text(84,531,'2,313,067개 파라미터',31,weight=700)
    s.arrow(545,366,672,366,pal['teacher'],6)
    s.text(608,326,'증류',33,anchor='middle',weight=700)
    s.panel(694,243,451,278)
    s.text(723,298,'대표 학생',37,color=pal['student'],weight=750)
    for x in [776,871,966]:s.circle(x,360,20,pal['student'])
    s.text(723,466,'34,406개 파라미터',32,weight=700)
    s.rect(55,617,1090,105,pal['student'],18)
    s.text(84,661,'파라미터 약 98.5% 감소',35,color='#FFFFFF',weight=750)
    s.text(84,704,'대표 학생 T5-a05 / FP32',29,color='#E9EFDF')
    s.text(1109,687,'87.06%',64,color='#FFFFFF',weight=750,anchor='end')
    return s


def kd_results(p):
    s=SVG(p,'대표 배포와 보존한 실험 결과','같은5850개 참조 이미지에서 교사83.54%, 정답만 학습한 학생73.56%, 대표 증류 학생87.06%와 새 실험의 고정 학생85.69%를 비교합니다. 대표 배포와 연구 선택의 역할을 구분합니다.',765)
    pal=p['theme'];s.header('대표 배포 모델과 확장 실험의 결과','같은 참조 이미지 5,850장. 대표 배포는 T5-a05이며 9개 조건의 검증 선택은 T5-a01입니다.')
    experimental=next(row for row in p['grid'] if row['id']==p['experimental_selected_candidate'])
    experimental_accuracy=experimental['reference_accuracy_by_seed_pct'][experimental['seeds'].index(p['experimental_selected_seed'])]
    experimental_count=p['models'][0]['count']
    rows=[(p['models'][0],pal['teacher'],'MobileNetV2'),(p['models'][1],pal['baseline'],'같은 학생 구조 / 정답만 학습'),(p['models'][2],pal['student'],'T5-a05 / 대표 배포 유지'),({'accuracy_pct':experimental_accuracy,'correct':round(experimental_accuracy/100*experimental_count),'count':experimental_count,'label_ko':'새 실험의 고정 학생'},pal['experimental'],'T5-a01 / 실험 선택 보존')]
    x0,x1=390,951;percent_grid(s,x0,x1,171,625)
    for i,(row,color,detail) in enumerate(rows):
        y=212+i*113
        s.text(48,y,row['label_ko'],27,weight=700)
        s.text(48,y+37,detail,20,color=s.muted)
        s.rect(x0,y-28,row['accuracy_pct']/100*(x1-x0),46,color,8)
        s.text(982,y+4,f"{row['accuracy_pct']:.2f}%",31,color=color,weight=750)
        s.text(982,y+44,f"{row['correct']:,} / {row['count']:,}",22,color=s.muted)
    s.text(1138,660,'정확도 (%)',22,anchor='end',color=s.muted)
    s.footer(['참조 결과는 이미 관측됐습니다. 기존 배포 유지는 후속 운영 결정이며 실험의 선택 기록은 바꾸지 않았습니다.'])
    return s


def kd_costs(p):
    s=SVG(p,'대표 모델의 구조와 파일 크기 및 CPU 비용','기존 대표 T5-a05 모델의 파라미터, 파일 바이트와 동일CPU의 추론 지연 중앙값을 비교합니다. 새T5-a01의 측정값을 대신 사용하지 않습니다.',1050)
    pal=p['theme'];s.header('대표 모델의 경량화 효과와 CPU 실행 비용','기존 대표 T5-a05의 측정. CPU 스레드 1개, 배치 1, 검증 이미지 128장과 반복 3회입니다.')
    teacher=p['models'][0];student=p['models'][2];fp=p['fp32']
    groups=[('파라미터 수','개',[("교사",teacher['parameters'],pal['teacher']),('대표 학생',student['parameters'],pal['student'])],1-student['parameters']/teacher['parameters'],'percent'),('파일 크기','KiB',[("교사 Keras",teacher['keras_bytes']/1024,pal['teacher']),('학생 Keras',student['keras_bytes']/1024,pal['student']),('학생 FP32 TFLite',fp['bytes']/1024,pal['experimental'])],1-student['keras_bytes']/teacher['keras_bytes'],'percent'),('CPU 추론 지연 중앙값','ms',[("교사 Keras",teacher['cpu_keras_p50_ms'],pal['teacher']),('학생 Keras',student['cpu_keras_p50_ms'],pal['student']),('학생 FP32 TFLite',fp['cpu_p50_ms'],pal['experimental'])],teacher['cpu_keras_p50_ms']/student['cpu_keras_p50_ms'],'speed')]
    for gi,(title,unit,rows,improve,kind) in enumerate(groups):
        y=154+gi*261;s.panel(48,y,1104,237)
        s.text(74,y+41,title,28,weight=700)
        s.text(1117,y+41,(f'{improve*100:.1f}% 감소' if kind=='percent' else f'Keras {improve:.2f}배'),28,color=pal['student'],weight=750,anchor='end')
        maximum=rows[0][1];x0=329;length=573
        for i,(label,v,color) in enumerate(rows):
            yy=y+92+i*54;s.text(74,yy+5,label,23)
            s.rect(x0,yy-22,length,29,'#ECE8EC',5)
            s.rect(x0,yy-22,max(2,v/maximum*length),29,color,5)
            value=f'{int(v):,} 개' if unit=='개' else (f'{v:,.1f} KiB' if unit=='KiB' else f'{v:.4f} ms')
            s.text(1117,yy+3,value,23,anchor='end',weight=700)
    s.footer(['파일 감소율은 같은 Keras 형식끼리 비교했습니다. FP32 TFLite는 별도 배포 형식입니다.', '한 Windows CPU 호스트의 순수 추론 측정이며 모델 로딩과 이미지 디코딩은 제외합니다. 다른 기기의 속도는 다릅니다.'])
    return s


def kd_grid(p):
    s=SVG(p,'전체 9개 조건의 검증 및 참조 평가 반복 실험','온도 3, 5, 10과 정답 손실 비중 0.1, 0.3, 0.5를 각각 3개 시드로 학습했습니다. 검증 평균으로 T5-a01을 선택했으며 대표 배포는 기존 T5-a05를 유지합니다. 평균과 표본 표준편차는 독립 확인 신뢰구간이 아닙니다.',950)
    pal=p['theme'];s.header('9개 조건의 반복 실험 결과','각 칸은 시드 378, 379, 380의 평균입니다. 아래의 ± 값은 표본 표준편차이며 신뢰구간이 아닙니다.')
    for side,base,title,metric,sdkey in [(0,121,'검증 정확도', 'validation_accuracy_mean_pct','validation_accuracy_sample_sd_pp'),(1,688,'참조 정확도','reference_accuracy_mean_pct','reference_accuracy_sample_sd_pp')]:
        s.text(base+210,176,title,30,anchor='middle',weight=750)
        s.text(base+210,212,'평균 % / 표준편차 %p',21,color=s.muted,anchor='middle')
        for ci,a in enumerate([.1,.3,.5]):s.text(base+ci*145+67,253,f'α = {a:.1f}',23,anchor='middle')
        for ri,t in enumerate([3,5,10]):
            yy=282+ri*143;s.text(base-18,yy+61,f'T={t}',21,color=s.muted,anchor='end')
            for ci,a in enumerate([.1,.3,.5]):
                row=next(r for r in p['grid'] if r['temperature']==t and r['alpha']==a)
                xx=base+ci*145; border=pal['experimental'] if row['validation_selected'] else (pal['student'] if row['representative_deployment'] else '#DED8DF')
                s.rect(xx,yy,133,127,'#FFFFFF',13,border,4 if (row['validation_selected'] or row['representative_deployment']) else 1)
                s.text(xx+66.5,yy+54,f"{row[metric]:.2f}",32,weight=750,anchor='middle')
                s.text(xx+66.5,yy+91,f"± {row[sdkey]:.2f}",23,color=s.muted,anchor='middle')
    s.rect(119,750,20,20,pal['experimental'],4);s.text(151,770,'T5-a01: 검증 평균으로 선택',24,weight=700)
    s.rect(675,750,20,20,pal['student'],4);s.text(707,770,'T5-a05: 대표 배포 유지',24,weight=700)
    s.lines(48,829,['T는 증류 온도이고 α는 정답 손실의 비중입니다. 조건 선택에는 참조 평가 결과를 사용하지 않았습니다.', '대표 배포 유지 결정은 이미 관측된 결과와 공개 모델의 연속성을 고려한 후속 결정입니다.'],size=21,line_height=32,color=s.muted)
    s.footer(['참조 평가: 같은 이미지 5,850장. 반복 개발 시드의 비교이며 독립적인 최종 확인 실험은 아닙니다.'])
    return s


def kd_errors(p):
    s=SVG(p,'대표 학생의 교사 오류 수정과 순수 증류 대조','대표 T5-a05 학생은 교사 오답 473개를 수정하고 교사 정답 267개를 놓쳐 정답이 206개 늘었습니다. 순수 증류 대조도 교사를 넘었지만 각 원인의 인과 기여는 확정하지 않았습니다.',1050)
    pal=p['theme'];s.header('대표 학생의 오류 수정과 학습 목적 비교','대표 T5-a05의 같은 참조 이미지 5,850장. 오류 수정과 새 오류를 함께 계산했습니다.')
    specs=[(48,'교사 오답을 수정',p['teacher_errors_repaired'],pal['student']),(424,'교사 정답을 놓침',p['teacher_correct_regressed'],pal['warning']),(800,'정답의 순수 증가',p['net_correct_gain'],pal['teacher'])]
    for x,label,value,color in specs:
        s.panel(x,152,352,197);s.text(x+24,197,label,27,weight=700)
        s.text(x+24,285,('+' if label=='정답의 순수 증가' else '')+str(value),65,color=color,weight=750)
        s.text(x+316,285,'개',25,anchor='end',color=s.muted)
    s.text(48,408,f"정확도 차이: +{p['net_correct_gain']/5850*100:.2f}%p",32,weight=750,color=pal['student'])
    s.line(48,441,1152,441)
    s.text(48,494,'같은 학생 구조의 학습 목적 비교',31,weight=750)
    s.text(48,535,'각 목적의 3회 학습 평균과 표본 표준편차. 교사 정확도 83.54%는 고정 모델 한 개의 결과입니다.',22,color=s.muted)
    x0,x1=380,963;percent_grid(s,x0,x1,567,811)
    colors=[pal['baseline'],pal['experimental'],pal['student']]
    for i,row in enumerate(p['controls']):
        yy=606+i*77;s.text(48,yy,row['label_ko'],25,weight=700)
        s.rect(x0,yy-27,row['mean_accuracy_pct']/100*(x1-x0),40,colors[i],7)
        s.text(990,yy+1,f"{row['mean_accuracy_pct']:.2f}%",28,weight=750)
        s.text(990,yy+32,f"± {row['sample_sd_pp']:.2f}%p",21,color=s.muted)
    s.text(1138,850,'정확도 (%)',21,color=s.muted,anchor='end')
    s.lines(48,889,['순수 증류도 3회 모두 교사를 넘었습니다. 정답 손실을 추가한 것만으로 현상을 설명할 수 없습니다.', '교사의 부분 미세 조정과 고정 BN, 학생 전체 학습과 BN 갱신, 최적화 경로 차이는 설명 후보입니다.'],size=21,line_height=34,color=s.muted)
    s.footer(['각 요인의 인과 기여는 확정하지 않았습니다. 압축이 정확도를 높이거나 학생이 항상 교사보다 낫다는 결론은 아닙니다.'])
    return s


def nlu_thumbnail(p):
    s=SVG(p,'문체와 문맥을 이용한 동일 저자 판별','영어 글 두 편의 문체 특징과 문맥을 독립적인 모델로 비교합니다. 고정 평가 4,708쌍에서 선택된 RoBERTa-Large CE의 Macro F1은 0.8084입니다.',760)
    pal=p['theme'];s.text(54,81,'문체와 문맥을 이용한 저자 판별',47,weight=750)
    s.text(54,134,'문체와 문맥을 이용한 동일 저자 판별',31,color=s.muted)
    for x,label,widths in [(56,'글 A',[260,226,254,203]),(435,'글 B',[225,261,206,251])]:
        s.panel(x,186,332,292);s.text(x+28,239,label,38,weight=750)
        for i,width in enumerate(widths):s.line(x+29,292+i*36,x+29+width,292+i*36,'#CEBAB6',5)
        s.rect(x+29,440,144,10,pal['primary'] if label=='글 A' else pal['secondary'],4)
    s.panel(838,186,306,292)
    s.text(863,243,'비교 모델',35,weight=750)
    s.text(863,310,'문체 앙상블',30,color=pal['secondary'],weight=700)
    s.text(863,376,'RoBERTa-Large',27,color=pal['primary'],weight=700)
    s.arrow(783,333,827,333,pal['primary'],4)
    s.rect(56,527,1088,184,pal['primary'],18)
    s.text(84,578,'배포용 Large CE',35,color='#FFFFFF',weight=750)
    s.text(84,633,'문서가 분리된 4,708쌍 평가',30,color='#F6E8E9')
    s.text(84,684,'Macro F1',29,color='#F6E8E9')
    ce=next(x for x in p['models'] if x['id']=='ce')
    s.text(1108,656,f"{ce['macro_f1']:.4f}",82,color='#FFFFFF',weight=750,anchor='end')
    return s


def nlu_results(p):
    s=SVG(p,'동일 저자 판별의 고정 파이프라인 비교','문서가 분리된 같은 4,708쌍에서 문체 앙상블과 RoBERTa-Large CE 및 직접 구현한 비대칭 focal을 비교합니다. 배포용 Large는 평가 전에 선택된 CE입니다.',745)
    pal=p['theme'];s.header('동일 조건에서 비교한 세 모델의 판별 성능','문서가 분리된 최종 4,708쌍. 입력 순서와 확률 보정을 포함한 고정 파이프라인의 결과입니다.')
    x0,x1=372,935;percent_grid(s,x0,x1,181,544,ticks=(0,.2,.4,.6,.8,1),max_value=1)
    labels={'stylometric':['문체 앙상블','XGBoost + LightGBM'],'ce':['RoBERTa-Large CE','평가 전 선택한 배포용 Large'],'custom_asl':['Large 비대칭 focal','직접 구현 / 보존한 비교 모델']}
    for i,row in enumerate(p['models']):
        y=228+i*122;color=pal[{'stylometric':'stylometric','ce':'ce','custom_asl':'custom_focal'}[row['id']]]
        s.text(48,y,labels[row['id']][0],25,weight=700);s.text(48,y+35,labels[row['id']][1],21,color=s.muted)
        s.rect(x0,y-27,row['macro_f1']*(x1-x0),42,color,7)
        s.text(966,y+4,f"{row['macro_f1']:.4f}",33,color=color,weight=750)
        s.text(966,y+37,f"정답 {row['correct']:,}쌍",21,color=s.muted)
    s.text(1138,581,'Macro F1',22,color=s.muted,anchor='end')
    s.footer(['CE는 모델 선택 집합에서 정한 배포 모델입니다. 최종 평가에서 focal의 점수가 높아도 선택을 바꾸지 않았습니다.', '직접 구현한 비대칭 이진 focal은 최초 프로젝트에서 ASL로 불렀으며 발표된 다중 라벨 ASL과 다릅니다.', '최초 개발 F1 0.8348과 평가 조건이 다르므로 통제된 전후 개선량으로 비교하지 않습니다.'])
    return s


def nlu_calibration(p):
    s=SVG(p,'같은 입력 순서에서의 확률 보정 전후 ECE','고정 입력 순서를 유지하고 같은 최종 4,708쌍에서 온도 보정 전후의 동일 저자 확률 ECE를 비교했습니다. 세 모델 모두 ECE가 감소했으며 작을수록 좋습니다.',765)
    pal=p['theme'];s.header('확률 보정 전후의 예측 확률 오차','같은 최종 4,708쌍과 고정 입력 순서에서 온도 보정 전후를 비교했습니다. ECE는 작을수록 좋습니다.')
    s.legend([('보정 전',pal['before'],185),('보정 후',pal['after'],185)],y=156)
    x0,x1=375,955;maximum=.07
    percent_grid(s,x0,x1,190,561,ticks=(0,.01,.02,.03,.04,.05,.06,.07),max_value=maximum)
    for i,row in enumerate(p['models']):
        y=228+i*116;labels={'stylometric':'문체 앙상블','ce':'RoBERTa-Large CE','custom_asl':'Large 비대칭 focal'}
        s.text(48,y+13,labels[row['id']],26,weight=700)
        for j,role in enumerate(['before','after']):
            yy=y+j*40;v=row['ece_'+role];s.rect(x0,yy-22,v/maximum*(x1-x0),28,pal[role],5)
            s.text(989,yy+3,f'{v:.4f}',28,weight=700)
    s.text(1138,625,'동일 저자 확률 ECE',22,color=s.muted,anchor='end')
    s.footer(['ECE는 동일 너비의 10개 확률 구간에서 평균 확률과 실제 동일 저자 비율의 차이를 가중한 값입니다.', '보정값은 별도 1,569쌍에서 맞추고 최종 평가 전에 고정했습니다. 최종 평가에서 다시 맞추지 않았습니다.', '확률 보정은 정답이나 F1이 늘어나는 것을 보장하지 않습니다.'])
    return s


RENDERERS = {
    'project-01': [('paper-qna-thumbnail.svg', paper_thumbnail), ('paper-qna-results.svg', paper_results)],
    'project-03': [('dm100-thumbnail.svg', dm_thumbnail), ('dm100-validation.svg', dm_validation), ('dm100-propellant.svg', dm_propellant), ('dm100-trajectory.svg', dm_trajectory)],
    'project-04': [('korean-rag-thumbnail.svg', rag_thumbnail), ('korean-rag-results.svg', rag_results), ('korean-rag-tradeoffs.svg', rag_tradeoffs)],
    'project-02': [('kd-thumbnail.svg', kd_thumbnail), ('kd-results.svg', kd_results), ('kd-costs.svg', kd_costs), ('kd-grid.svg', kd_grid), ('kd-errors.svg', kd_errors)],
    'project-05': [('nlu-thumbnail.svg', nlu_thumbnail), ('nlu-results.svg', nlu_results), ('nlu-calibration.svg', nlu_calibration)],
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.input.read_bytes(); input_hash=hashlib.sha256(raw).hexdigest(); data=json.loads(raw.decode('utf-8-sig'))
    projects=data['projects'] if isinstance(data,dict) and 'projects' in data else (data if isinstance(data,list) else [data])
    ids=[p['site_id'] for p in projects]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate project IDs')
    if not all(identifier in RENDERERS for identifier in ids): raise ValueError('Unknown project ID')
    args.output.mkdir(parents=True,exist_ok=True)
    files=[]
    for project in projects:
        for filename,renderer in RENDERERS[project['site_id']]:
            target=args.output/filename
            rendered=renderer(project).finish(input_hash)
            if any(char in rendered for char in ['·','ㆍ','•']): raise ValueError('Middle-dot characters are forbidden')
            target.write_text(rendered,encoding='utf-8',newline='\n')
            files.append({'filename':filename,'project_id':project['site_id'],'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    receipt={'schema_version':1,'status':'generated_from_published_measurements','visual_data_sha256':input_hash,'project_ids':ids,'files':files,'network_or_model_execution_performed':False,'source_paths_read':False}
    (args.output/'visual-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'generated','projects':len(projects),'svg_files':len(files),'receipt':str(args.output/'visual-receipt.json')},ensure_ascii=False))


if __name__ == '__main__':
    main()
