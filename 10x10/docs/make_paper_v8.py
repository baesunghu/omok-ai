"""Paper v8 (2026-10-26, user): v7 (여명 + 메기, ep100) + everything after it up to the NOON model ep530.
Reads Downloads/국문_한글_HWP_202608_v7.docx, writes Downloads/국문_한글_HWP_202608_v8.docx (v7 untouched).
New text is put into copies of v7's own paragraphs and tables, so the look stays the same."""
import copy
import os

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

DL = r"C:\Users\My PC\Downloads"
SRC = os.path.join(DL, "국문_한글_HWP_202608_v7.docx")
DST = os.path.join(DL, "국문_한글_HWP_202608_v8.docx")

d = docx.Document(SRC)
B = d.element.body


def blocks():
    return list(B.iterchildren())


def para_at(i):
    return Paragraph(blocks()[i], d)


def find_para(prefix):
    for el in blocks():
        if el.tag.endswith("}p") and Paragraph(el, d).text.strip().startswith(prefix):
            return el
    raise KeyError(prefix)


def set_text(p_el, text):
    """Keep the first run's formatting, drop the other runs."""
    p = Paragraph(p_el, d)
    runs = p.runs
    if not runs:
        p.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)


def clone_para(template_el, text):
    el = copy.deepcopy(template_el)
    for child in list(el):   # drop pictures/fields that may sit in the template
        if child.tag.endswith("}r") and child.findall(".//{*}drawing"):
            el.remove(child)
    set_text(el, text)
    return el


def clone_table(template_el, rows):
    """rows: list of lists of strings; the template's first data row is reused for every body row."""
    el = copy.deepcopy(template_el)
    t = Table(el, d)
    trs = [r._tr for r in t.rows]
    header, body_tpl = trs[0], trs[1]
    for tr in trs[1:]:
        el.remove(tr)
    tbl_rows = [header] + [copy.deepcopy(body_tpl) for _ in rows[1:]]
    for tr in tbl_rows[1:]:
        el.append(tr)
    t = Table(el, d)
    for r, vals in zip(t.rows, rows):
        for c, v in zip(r.cells, vals):
            for p in c.paragraphs[1:]:
                p._element.getparent().remove(p._element)
            set_text(c.paragraphs[0]._element, v)
    return el


# --- templates from v7 -------------------------------------------------------------------------------------------
T_HEAD = find_para("8.5 실전 운영")            # sub-section title
T_BODY = find_para("2026년 10월 17~18일 학교 축전")   # body text
T_CAP = find_para("<표 14>")                    # centred caption
T_BULLET = find_para("∙AlphaGo Zero")           # bullet line
tables = [el for el in blocks() if el.tag.endswith("}tbl")]
T5 = next(el for el in tables if Table(el, d).rows[0].cells[0].text == "모델")        # 5 columns
T4 = next(el for el in tables if Table(el, d).rows[0].cells[0].text == "문제")        # 4 columns
T3 = next(el for el in tables if Table(el, d).rows[0].cells[0].text == "구간" and len(Table(el, d).columns) == 3)
EMPTY = copy.deepcopy(find_para("8.5 실전 운영").getprevious())   # an empty line


def H(t): return clone_para(T_HEAD, t)
def P(t): return clone_para(T_BODY, t)
def C(t): return clone_para(T_CAP, t)
def L(t): return clone_para(T_BULLET, t)
def E(): return copy.deepcopy(EMPTY)


def insert_after(anchor, els):
    for el in els:
        anchor.addnext(el)
        anchor = el
    return anchor


# --- 1. ABSTRACT -------------------------------------------------------------------------------------------------
set_text(find_para("Self-play reinforcement learning such as AlphaZero"),
         "  Self-play reinforcement learning such as AlphaZero has achieved superhuman play, but it assumes large-scale "
         "computing resources. This study trains an 11x11 Gomoku AI on a single laptop (i5-8300H, GTX 1050) and reports "
         "the main difficulty we met in that setting: overconfidence of the value network. Because self-play pits a network "
         "against itself, both sides share the same misjudgments, so positions the network wrongly believes it is winning "
         "are rarely punished. We propose CATFISH (Crippled Anchors To Fight Inbred Self-play Habits), a league of opponents "
         "made by copying the current network every epoch and disabling one of its judgments (value output set to zero or "
         "halved, policy flattened). Only the main network's moves and the final results of these games are used for "
         "training, so the method needs no extra training and no external data. A first model, NAPS-625 (DAWN), reached a "
         "Rapfi equal-strength point of 959 nodes. A cold-start run (v3) with Gumbel AlphaZero, auxiliary targets and CATFISH "
         "passed DAWN within six hours, and its checkpoint ep530, about 69 hours after a random start, became the first "
         "model to meet our NOON criterion on 2026-10-25: all 10 required problems and 15 of 18 hard problems taken from "
         "games the program had lost (25 of 28 overall). Measured with 200 games against Rapfi, its equal-strength point is "
         "about 52,000 nodes (68% interval 40,700-65,000), roughly 50 times DAWN and 10 times the previous best, with a third "
         "of the parameters (2.7 M). A problem that had blocked every earlier model was traced to an over-pessimistic value "
         "network and was first solved by the half-value CATFISH copy. Because no controlled run without CATFISH has been "
         "made yet, these results are a case report rather than proof of the method's effect.")

# --- 2. research-question answer --------------------------------------------------------------------------------------
set_text(find_para("첫째 질문에는 ‘그렇다’라는 방향의 근거를 얻었다."),
         "첫째 질문에는 ‘그렇다’라는 방향의 근거를 얻었다. 둘째 질문에는 메기 투입 뒤 가치 손실의 하락, 과신 문제의 해결, "
         "그리고 메기를 넣은 v3가 약 3일 만에 정오 기준(필수 10문제 전부 + 어려운 문제 15/18)을 처음으로 채운 정황 근거를 "
         "얻었으나, 메기를 뺀 대조 실험을 아직 하지 않아 효과를 확정하지는 못하였다. 두 결론 모두 표본이 작아 예비적이다.")

# --- 3. stale sentence in 8.4 ----------------------------------------------------------------------------------------
set_text(find_para("100에폭은 10문제 중 9문제로 계보 최고이며"),
         "100에폭은 10문제 중 9문제로 당시 계보 최고였으며, 2단계 4/5는 이전 후보 ep880과 같은 공동 최고였다. 특히 p17은 "
         "판세를 +0.8 이상으로 확신하다가 이길 수를 놓치는 과신의 전형으로, 메기 투입 뒤 3/3으로 풀렸다. 남은 p18은 판세를 "
         "−0.6으로 보아 불리함은 알지만 정답 수순을 찾지 못하였다. 이 문제의 원인과 해결은 8.7절에서 다룬다.")

# --- 4. new sections 8.6-8.10 after 8.5 --------------------------------------------------------------------------------
new = [E(), E(),
       H("8.6 v3: 이후 경과 (ep100 ~ ep500)"),
       P("100에폭 이후에도 학습을 이어 가며 몇 가지를 바꾸었다. 130에폭부터 쉬운 상대였던 정책 0 메기를 빼고 정책 출력을 "
         "절반으로 줄인 메기를 넣었고, 140에폭부터 메기 대국의 비율을 25%에서 12.5%로 낮추고 정밀 탐색을 200회에서 400회로 "
         "늘렸다. 자가대국의 다양성을 위해 일부 판을 무작위 돌 몇 개에서 시작하게 하고, 처음 네 수는 확률적으로 고르게 하였다. "
         "169에폭에는 처음부터 고정이던 학습률을 1e-3에서 3e-4로 낮추었다. 또 10월 23일, 그때까지의 측정이 연속 4목 탐색(VCF) "
         "없이 이루어졌음을 발견하여 200에폭부터는 학습과 같은 VCF 켬 조건으로 측정하였다. 주요 체크포인트의 결과는 표 15와 같다."),
       C("<표 15> v3 주요 체크포인트 (대등국면 단위: Rapfi 노드)"),
       clone_table(T5, [["체크포인트", "시작 후", "대등국면", "문제 채점", "비고"],
                        ["ep100", "약 8시간", "5,214", "9/10", "메기 투입 21에폭째"],
                        ["ep170", "약 17시간", "12,021", "8/10", "95% 하한도 이전 최고 초과"],
                        ["ep200", "약 21시간", "2만 이상", "9/10", "VCF 켬 첫 측정"],
                        ["ep300", "약 1.5일", "42,067*", "24/28 (졸업)", "p18 첫 정식 통과"],
                        ["ep400", "약 2일", "66,296*", "24/28", "3단계 15/18, 1단계 1문제 실패"],
                        ["ep500", "약 2.7일", "맞춤 실패", "24/28", "3단계 14/18"],
                        ["ep530", "약 2.9일", "51,696*", "25/28 (정오)", "8.8절"]]),
       P("* 10월 25일 같은 조건(5단계 × 40판)으로 다시 잰 값이다(8.9절). ep300은 처음 측정에서 83,823으로 나왔으나, 단계당 "
         "20판과 개시 칸 8개로 잰 과대평가였다. 졸업의 정의는 10월 23일에 ‘1·2단계 10문제 전부 통과’로 넓혔으며, 이 기준으로 "
         "v3의 졸업은 ep300이다."),
       E(),
       H("8.7 p18의 원인: 가치망의 과한 비관"),
       P("2단계 p18은 계보의 어떤 모델도 정식 채점에서 풀지 못한 수문장이었다. 먼저 문제가 공정한지 확인하기 위해 Rapfi끼리 "
         "두게 하였다. 공격 100만 노드를 상대로 수비 Rapfi는 100만, 30만, 3만 노드 모두에서 40수를 버텼으므로 지킬 수 있는 "
         "문제이다. 우리 모델은 첫 수를 맞히고도 그 뒤에 무너졌다. 루트에 승리 수순 탐색(VCT)을 붙여 보았으나 수를 바꾼 적이 "
         "없어 수읽기 부족이 주원인은 아니었다. 250에폭의 메기 복사본에게 같은 문제를 풀게 하자 원본은 1/3(시작 판세 −0.45), "
         "가치망 0 메기는 2/3(−0.1), 가치망 절반 메기는 3/3(−0.25)이었다. 원인은 과신의 반대인 가치망의 과한 비관이었고, "
         "판단을 절반으로 줄인 메기가 먼저 풀었다. 이후 300에폭 본 모델이 정식 채점에서 처음 통과하였다."),
       E(),
       H("8.8 정오: ep530 (2026. 10. 25.)"),
       P("정오(NOON)는 10월 23일에 정한 기준으로, 졸업(1·2단계 10문제 전부 통과)에 더해 3단계 18문제 중 15문제 이상을 통과하는 "
         "것이다. 중간 단계는 두지 않는다. 400에폭이 3단계 15/18에 처음 닿았으나 1단계 문제 하나(구석 대각선)를 풀지 못하여 "
         "정오로 인정하지 않았다. 530에폭은 2026년 10월 25일 14시 14분 29초에 채점을 마치며 조건을 모두 채운 첫 모델이 되었다"
         "(표 16). 무작위 신경망에서 시작한 지 약 69시간 만이며, 정오 선언은 10월 26일 13시 7분에 하였다."),
       C("<표 16> 정오 기준 채점: ep530 (문제당 3회 중 2회 이상 통과)"),
       clone_table(T4, [["단계", "문제 수", "통과", "비고"],
                        ["1단계 (졸업 문제)", "5", "5", "구석 대각선 3/3 포함"],
                        ["2단계", "5", "5", "p18 포함"],
                        ["3단계", "18", "15", "실패: 27수, 4수, 14수"],
                        ["합계", "28", "25", "역대 최고, 정오 충족"]]),
       E(),
       H("8.9 같은 조건의 재측정과 측정의 포화"),
       P("체크포인트마다 측정 조건이 달라 비교가 어려웠으므로, ep300, ep400, ep530을 같은 조건(Rapfi 3,560~288,000노드의 "
         "5단계, 단계당 40판, 개시 칸 25개, 모두 200판)으로 다시 재었다(표 17). 대등국면은 세 가지로 계산하였다. 기존 방식은 "
         "승 1, 무 0.5로 점수를 매겨 직선을 맞추고, Davidson 방식은 무승부와 흑백 차이를 함께 모델링한다. 색별 성취 방식은 "
         "아래의 흑 필승 관찰에 근거하여 흑은 승리만, 백은 무승부 이상을 성취로 센다."),
       C("<표 17> 같은 조건 재측정 (단계별 점수와 대등국면, 단위: 노드)"),
       clone_table(T4, [["항목", "ep300", "ep400", "ep530"],
                        ["3,560 노드 점수", "76%", "79%", "76%"],
                        ["10,680 노드", "62%", "55%", "66%"],
                        ["32,000 노드", "48%", "49%", "50%"],
                        ["96,000 노드", "41%", "49%", "39%"],
                        ["288,000 노드", "35%", "44%", "40%"],
                        ["기존 방식", "43,190", "68,109", "52,900"],
                        ["Davidson (68% 범위)", "42,067 (33,926~53,017)", "66,296 (50,344~94,241)", "51,696 (40,701~64,959)"],
                        ["색별 성취", "38,430", "96,000", "42,114"]]),
       P("세 방식 모두 ep530을 약 4만~5만 노드로 보았다. 이는 첫 졸업 모델 여명(959)의 약 50배, 이전 계보 최고 NAPS-960(4,432)의 "
         "약 10배이다. Rapfi 기준으로는 ep400이 조금 더 높았으나 68% 범위가 겹치며, 두 모델의 직접 대결 72판에서 ep530의 점수는 "
         "49.3%(25승 26패 21무)로 동률이었다."),
       P("고노드 구간에서는 모든 모델의 점수가 40% 안팎에서 평평해졌다. 원인을 보기 위해 Rapfi끼리 같은 노드로 25판씩 두게 "
         "하였다(표 18). 3.2만 노드 이상에서 백은 75판 중 한 판도 이기지 못하였고, 28.8만 노드에서는 80%가 무승부였다. 11×11 "
         "자유 오목에서도 흑에게 필승 수순이 있다는 강한 추측과 맞는 결과이며, 이 전제에서는 백의 무승부가 흑의 승리와 같은 "
         "성취이다. 노드를 늘려도 무승부만 늘어나므로, 기존의 대등국면 자는 이 구간에서 실력 차이를 담지 못한다(측정의 포화)."),
       C("<표 18> Rapfi끼리 둔 기준선 (같은 노드, 개시 칸 25개)"),
       clone_table(T4, [["Rapfi 노드", "흑 승", "백 승", "무승부"],
                        ["10,680", "13", "2", "10"],
                        ["32,000", "19", "0", "6"],
                        ["96,000", "13", "0", "12"],
                        ["288,000", "5", "0", "20"]]),
       E(),
       H("8.10 축제 모델과 모델의 크기"),
       P("10월 28일 축제에는 정오 모델 ep530을 내보내기로 하였다. Rapfi 기준 실력이 ep400과 같고, 사람에게서 나온 1·2단계 "
         "문제를 모두 푸는 쪽이 관람객과의 대국에 더 안전하다고 판단하였다. ep530의 파라미터는 약 271만 개로 지난 축제 모델 "
         "NAPS-880(약 860만 개)의 약 3분의 1이지만, Rapfi 대등국면은 약 20배이다. 작은 신경망은 같은 GPU에서 더 많은 자가대국을 "
         "두게 해 주었고(v3는 약 3일에 7만 4천 판 이상, NAPS 계열은 약 3주에 약 1만 8천 판), 웹 대국판의 탐색 2,000회 응답도 "
         "약 1~1.5초로 빨라졌다. 축제 서버도 채점과 같은 VCF 엔진으로 바꾸었다."),
       ]
insert_after(find_para("2026년 10월 17~18일 학교 축전"), new)

# --- 5. conclusion, limits, follow-up ---------------------------------------------------------------------------------
set_text(find_para("둘째, 판단 일부를 끈 자기 복사본(메기)이 자가대국의 과신을 줄이고"),
         "둘째, 판단 일부를 끈 자기 복사본(메기)이 자가대국의 과신을 줄이고 학습을 도울 수 있는가. 가치망 제거 실험에서 가치망을 "
         "끈 모델이 원본이 0/3이던 최난도 문제를 3/3으로 수비하였고, 이를 연습 상대로 바꾼 것이 메기 리그이다. 메기는 간단하고"
         "(출력층 하나 변경), 값싸고(추가 학습 없음), 범용적이며, 난도가 저절로 조절되고, 외부 데이터를 쓰지 않는다. 처음부터 다시 "
         "학습한 v3에 메기를 넣은 뒤 오르던 가치 손실이 내려갔고(0.191 → 0.133), 과신 문제 p17이 풀렸으며, 6시간 만에 여명을 "
         "넘어섰다. 계보 내내 막혀 있던 p18은 가치망의 과한 비관이 원인이었고 가치망 절반 메기가 먼저 풀었다. 그리고 530에폭은 "
         "무작위에서 약 69시간 만에 정오 기준(필수 10문제 전부, 어려운 문제 15/18, 전체 25/28)을 처음 채웠다. 같은 조건 200판으로 "
         "잰 Rapfi 대등국면은 약 5만 2천 노드로 여명의 약 50배, 이전 최고의 약 10배이며, 파라미터는 이전 모델의 약 3분의 1이다.")
set_text(find_para("두 결과 모두 ‘그렇다’는 방향의 근거이다."),
         "두 결과 모두 ‘그렇다’는 방향의 근거이다. 2016년 이세돌의 78수가 AI의 과신을 깬 사람의 한 수였다면, 메기는 AI가 "
         "스스로 과신을 깨는 방법으로서 이 연구의 78수였다. 다만 둘째 질문은 메기를 뺀 대조 실험 없이 얻은 사례 보고이므로, "
         "효과를 확정하려면 아래 후속 연구가 필요하다.")
lim = find_para("여섯째, 메기의 효과는 대조 실험으로 확인되지 않았다.")
insert_after(lim, [E(), P(
    "열째, 대등국면 자는 고노드 구간에서 포화한다. 흑이 크게 유리하여 노드를 늘려도 무승부만 늘어나므로 기존 방식의 대등국면은 "
    "이 구간에서 불안정하다(8.9절). 색별 성취 방식은 11×11에서도 흑 필승이라는 추측에 기대며, Rapfi끼리 둔 기준선도 노드당 "
    "25판뿐이다. 열한째, 정오 판정은 3단계 18문제 중 15문제라는 경계에 걸려 있어 문제 하나의 결과에 좌우된다. 같은 조건에서 "
    "한 판의 결과가 GPU 연산의 미세한 차이로 바뀐 사례(3단계 6수, 가채점 2/3과 본 채점 3/3)도 있었다.")])
set_text(find_para("첫째, 메기를 뺀 대조 실험이다."),
         "첫째, 메기를 뺀 대조 실험이다. 같은 v3 설정에서 메기 없음, 79에폭부터 메기, 처음부터 메기의 세 조건을 학습하고, 같은 "
         "에폭에서 래더와 문제 채점, 고정 검증 국면의 가치 오차와 과신 정도를 비교한다. 둘째, 메기 종류별 효과를 따로 잰다. 셋째, "
         "포화하지 않는 자를 만든다. 색별 성취 방식의 기준선을 노드별로 늘리고, 직접 대결과 함께 쓰는 방법을 검토한다. 넷째, 15×15 "
         "오목과 다른 AlphaZero 계열 게임으로 확장한다. 다섯째, 10월 28일 축제에서 정오 모델 ep530과 관람객의 대국을 모아 사람의 "
         "실력과 Rapfi와 AI를 같은 자 위에 놓는다.")

# --- 6. appendix: measurement conditions + timeline ----------------------------------------------------------------
appA = find_para("상대: Rapfi, 자유 오목(rule 0), 탐색 노드 수 제한.")
insert_after(appA, [P("v3의 정밀 재측정(2026-10-25~26): Rapfi 3,560·10,680·32,000·96,000·288,000 노드의 5단계, 단계당 40판(모두 200판), "
                      "개시 칸 중앙 5×5의 25개, 우리 모델은 탐색 1,100회와 VCF(깊이 10, 노드 3,000)를 켜고 잡음 없이 두었다. "
                      "Davidson 방식은 판 단위 재표집(400회)으로 범위를 구하였다.")])
last = find_para("2026년 10월 22~23일 웹 검색으로")
insert_after(last, [E(), P("  H. v3 연표"), E(),
                    P("10/22 14:52 v3 시작(무작위 신경망) · 20:59 메기 투입(79에폭) · 23:52 ep100 대등국면 5,214. "
                      "10/23 ep170 12,021 · ep200 2만 이상(VCF 켬) · 측정의 VCF 누락 발견과 정정 · 졸업 정의 확대. "
                      "10/24 ep300 졸업(24/28, p18 첫 통과) · ep400 3단계 15/18이나 1단계 실패로 정오 아님. "
                      "10/25 ep500 24/28 · 14:14:29 ep530 정오 충족(25/28) · Rapfi 정밀 재측정. "
                      "10/26 ep530 대 ep400 직접 대결 72판 49.3% · 13:07 정오 선언 · 축제 모델 ep530 결정.")])

d.save(DST)
print("saved", DST)
