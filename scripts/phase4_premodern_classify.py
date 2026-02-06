"""Phase 4: 전근대 원전 기준 현토 재분류 (개선판)

존칭 변이형, 축약형 등을 포함한 확장 분류
"""
import pandas as pd
from collections import defaultdict
from pathlib import Path
import json
import re

# 전근대 원전 기준 분류 (확장판 - 수정)
# 주의: 나/니/며 = 이나/이니/이며의 축약 (NOT 하나/하니/하며)
PREMODERN_TAXONOMY = {
    # === 구두점/오류 (제외) ===
    "제외_구두점": {
        "description": "구두점 - 현토 아님",
        "source": "오류",
        "patterns": [r"^ㆍ$", r"^,$", r"^\."]
    },
    
    # === 종결 어미 (단사) ===
    "단사_미절": {
        "description": "약하게 끊음 (微絶)",
        "source": "임규직 《구두해법》",
        "patterns": [r"^(이)?라$"]
    },
    "단사_쾌절": {
        "description": "단호하게 결정하여 끊음 (?絶)",
        "source": "임규직 《구두해법》",
        "patterns": [r".*니라$", r".*시니라$"]
    },
    "단사_범론이단": {
        "description": "보편적 진리를 진술하여 단정함",
        "source": "임규직 《구두해법》",
        "patterns": [r".*하나니라$"]
    },
    "단사_기사지단": {
        "description": "정령, 조서 등 공적 기록",
        "source": "임규직 《구두해법》",
        "patterns": [r"^하다$", r".*하시다$"]
    },
    "단사_서술지단": {
        "description": "사건의 전말이나 행적을 서술함",
        "source": "임규직 《구두해법》",
        "patterns": [r".*러라$", r".*더라$"]
    },
    "단사_유사이단": {
        "description": "감탄이나 여운을 남김 (游辭以斷) - 탄사",
        "source": "임규직 《구두해법》 / 박문호 《이두해》 30번",
        "patterns": [r".*로다$", r".*놋다$", r".*도다$"]
    },
    
    # === 주체/객체 구분 ===
    "주체_한대": {
        "description": "주체의 행위, 동일 주어 유지",
        "source": "이삼환 《구두지남》 구두거요 / 임규직 2번",
        # 완대/이완대 제거 (의미 다름 - 미분류로)
        "patterns": [r"^한대$", r"^하신대$", r".*한대$", r"^로대$", r"^이로대$",
                    r"^온대$", r"^이온대$"]
    },
    "객체_어늘": {
        "description": "객체의 행위, 주어 전환",
        "source": "이삼환 《구두지남》 구두거요 / 임규직 2번",
        # -하거- 축약형, 존칭형 어시늘 추가
        "patterns": [r".*어늘$", r".*거늘$", r".*커늘$", r".*커든$", r".*커나$", r".*커니와$",
                    r".*시늘$", r"^어시늘$"]
    },
    
    # === 시제/양태 ===
    "과거_러니": {
        "description": "과거/회상 (往昔事)",
        "source": "이삼환 《구두지남》 / 박문호 13번",
        # 랬다 추가
        "patterns": [r".*러니$", r".*더니$", r".*러시니$", r".*더시니$", 
                    r"^시니$", r".*터니$", r".*러시다$", r"^랬다$"]
    },
    "미래_리니": {
        "description": "미래/필연 (未來事, 必然事)",
        "source": "이삼환 《구두지남》 / 박문호 15번",
        "patterns": [r".*리니$"]
    },
    "미래_리라": {
        "description": "미래 추측/의지",
        "source": "이삼환 《구두지남》",
        # 하리다, 하리 추가
        "patterns": [r".*리라$", r"^하리다$", r".*하리다$", r"^리다$", r"^하리$"]
    },
    "진행_할새": {
        "description": "한창 진행 중임 (方將)",
        "source": "임규직 11번 / 박문호 12번",
        # 새라, ㄹ새니 등 변이형 포함
        "patterns": [r".*할새$", r".*ㄹ새$", r"^라가$", r"^새$", r"^새라$", 
                    r".*실새$", r".*ㄹ새니$", r".*쇠새$"]
    },
    
    "의문_설명": {
        "description": "설명의문 (何, 豈 등 아래)",
        "source": "이삼환 《구두지남》 / 박문호 31번",
        # 잇고는 설명의문, 리요/리시오는 리오 변형
        "patterns": [r"^리오$", r"^고$", r"^오$", r".*리오$", r".*잇고$",
                    r"^리요$", r".*리요$", r"^리시오$"]
    },
    "의문_판정": {
        "description": "판정의문/반어",
        "source": "이삼환 《구두지남》 / 박문호 31번",
        # 릿가, 닛가, 닛고 등 변이형 포함
        "patterns": [r"^잇가$", r".*잇가$", r"^인저$", r".*ㄴ저$", 
                    r"^가$", r"^아$", r".*릿가$", r".*닛가$", r".*닛고$",
                    r".*리까$", r".*닛까$", r".*이릿고$"]
    },
    
    # === 연결 어미 ===
    "일의상승_하야": {
        "description": "하나의 뜻이 이어짐 / 인과관계",
        "source": "임규직 1번 / 박문호 1번",
        # 여는 감탄사(오호라의 라 같은 것)이므로 제외
        "patterns": [r"^하야$", r"^하여$", r"^하샤$", r"^하사$"]
    },
    "대우_하고하며": {
        "description": "짝을 이루는 말 / 나열 (하고/하며 계열)",
        "source": "임규직 3번 / 박문호 2번",
        # 코 = 하고의 축약형
        "patterns": [r"^하고$", r"^하며$", r".*하시고$", r".*하시며$", r"^코$"]
    },
    "대우_이오이며": {
        "description": "대우 (이오/이며 계열) - 며, 요는 축약형",
        "source": "임규직 3번",
        "patterns": [r"^이오$", r"^요$", r"^이며$", r"^며$"]
    },
    "승상_하니": {
        "description": "위 구절을 잇는 말 (하니 계열)",
        "source": "임규직 6번",
        "patterns": [r"^하니$", r".*하시니$"]
    },
    "승상_이니": {
        "description": "위 구절을 잇는 말 (이니 계열) - 니는 축약형",
        "source": "임규직 6번",
        # 로니 추가 (~하였으니)
        "patterns": [r"^이니$", r"^니$", r"^로니$", r"^이로니$"]
    },
    "범론_하나니": {
        "description": "두루 진술하는 말",
        "source": "임규직 7번 / 이삼환",
        "patterns": [r".*하나니$", r".*이나니$"]
    },
    
    # === 인용 ===
    "인용": {
        "description": "옛말이나 타인의 말 인용",
        "source": "임규직 14번 / 이삼환",
        "patterns": [r".*라하니$", r".*라하고$", r".*라하다$"]
    },
    
    "자칭_하노라": {
        "description": "자신을 일컫는 말",
        "source": "임규직 8번 / 박문호 26번",
        # 노라, 호라, 이노라, 아호라, 로라 추가
        "patterns": [r".*하노라$", r".*호니$", r".*하노니$", r".*호리라$",
                    r"^노라$", r"^호라$", r"^이노라$", r"^아호라$",
                    r"^로라$", r".*로라$"]
    },
    
    # === 존칭 ===
    "존칭": {
        "description": "존칭/겸양 표현",
        "source": "임규직 전체 / 박문호 32번",
        # 오니, 께, 시고 추가
        "patterns": [r".*이다$", r".*니이다$", r".*소서$", r".*노이다$", r".*시니이다$",
                    r"^오니$", r"^이오니$", r"^께$", r"^시고$"]
    },
    
    "상반_호되": {
        "description": "서로 반대되는 내용",
        "source": "임규직 9번 / 박문호 21번",
        # 하되 계열 포함, 허되/오호되/아호되 등 변이형
        "patterns": [r"^호되$", r".*로되$", r".*하되$", r".*시되$",
                    r"^허되$", r".*오호되$", r".*아호되$"]
    },
    
    # 인용선도 범주 제거: 복합형(라호대 등)은 복합으로, 단순형(호대, 컨대)은 미분류로
    
    # === 점층 ===
    "점층": {
        "description": "점층/심화 (?, ? 위) - 하물며",
        "source": "이삼환 / 박문호 18번",
        # 원전: 이어든, 이온, 이어니 (어든은 가정과 중복→일독양용으로 처리)
        "patterns": [r"^온$", r"^이온$", r"^어니$", r"^이어니$"]
    },
    
    # === 양보/반어 ===
    "양보_이나이라도": {
        "description": "양보/반어 (雖 아래) - 이나/이라도 계열, 나는 축약형",
        "source": "이삼환 / 박문호 17번",
        # 들 추가 (~한들)
        "patterns": [r"^이나$", r"^나$", r".*라도$", r".*어니와$", r"^들$", r".*들$"]
    },
    
    # === 조사/직하 ===
    "직하_조사": {
        "description": "곧장 내려오는 조사",
        "source": "임규직 5번",
        "patterns": [r"^이$", r"^은$", r"^는$", r"^의$", r"^를$", r"^을$"]
    },
    "처소_에": {
        "description": "처소/대상 (於, 于, 乎 아래)",
        "source": "이삼환 / 박문호 19번",
        # 에서 추가
        "patterns": [r"^에$", r"^애$", r"^에서$"]
    },
    "나열_와과": {
        "description": "낱낱이 셈 / 일일이 거론",
        "source": "임규직 12번 / 박문호 11번",
        "patterns": [r"^와$", r"^과$"]
    },
    
    # === 기타 ===
    "가정": {
        "description": "가정/조건 (若, 如 아래)",
        "source": "이삼환 / 박문호 14번",
        # 인댄 계열: ㄴ댄(인댄), 컨댄, 댄, 신댄 등
        # 하면 계열, 어시든/커시든 등 존칭 변이형
        # 런들/ㄴ들 추가 (가정 조건)
        "patterns": [r"^면$", r".*이면$", r".*어든$", r".*하시면$",
                    r".*ㄴ댄$", r".*컨댄$", r"^댄$", r".*하면$", r".*시면$",
                    r"^신댄$", r".*어시든$", r".*커시든$",
                    r"^라면$", r".*런들$", r"^ㄴ들$", r"^한들$", r".*ㄴ대$",
                    r"^이신대$", r"^인대$", r"^완대$"]
    },
    "수단_으로": {
        "description": "수단/기점 (以, 使 아래)",
        "source": "이삼환 / 박문호 23번",
        "patterns": [r"^로$", r"^으로$"]
    },
    "청원_하라": {
        "description": "금지/청원",
        "source": "임규직 13번 / 박문호 27번",
        "patterns": [r"^하라$", r".*어다$"]
    },
    "개괄_히": {
        "description": "대략적으로 묶음 (?括)",
        "source": "박문호 29번",
        "patterns": [r"^히$"]
    },
    "유역_도": {
        "description": "또한/역시 (猶亦)",
        "source": "박문호 20번",
        "patterns": [r"^도$"]
    },
    
    
    # === 추가 분류 (미분류 해소) ===
    # 인용선도_컨대 삭제됨 - 컨대는 미분류로
    "양보_언만": {
        "description": "양보 표현 (~건만, ~언정)",
        "source": "구두 전통",
        "patterns": [r"^언만$", r"^언정$", r"^언마는$", r"^이언마는$"]
    },
    "감탄": {
        "description": "감탄/호격 표현",
        "source": "구두 전통",
        "patterns": [r"^여$", r"^저$", r"^ㄴ저$", r"^인져$", r"^ㄴ져$", r"^져$", 
                    r"^시여$", r"^온여$", r"^야$"]
    },
    "필수조건_라야": {
        "description": "필수 조건 (~라야, ~이라야)",
        "source": "구두 전통",
        "patterns": [r"^라야$", r"^이라야$", r"^에야$", r"^라사$", r"^이라사$",
                    r"^오야$", r"^오사$", r"^로사$"]
    },
    "지속_타가": {
        "description": "동작 지속 후 전환 (~다가)",
        "source": "구두 전통",
        "patterns": [r"^타가$", r"^하다가$"]
    },
    # 로니 = 이유/승상, 로라 = 자칭 계열이므로 별도 범주 불필요
    # 온은 점층으로 이동됨
    "이유_ㄹ새라": {
        "description": "이유 표현 (~ㄹ새, ~일새)",
        "source": "구두 전통",
        "patterns": [r"^ㄹ새라$", r"^일새라$", r"^ㄹ새요$", r"^일새요$", r"^일세$"]
    },
}

# 일독양용 (一讀兩用): 반어로 두 가지 용법으로 쓰이는 마커
# 박문호 33번: 이어늘, 호대, 이어든
DUAL_USAGE_MARKERS = {
    "이어늘": ["객체_어늘", "반어"],
    "어늘": ["객체_어늘", "반어"],
    "호대": ["주체_한대", "반어"],
    "이어든": ["점층", "반어"],
    "어든": ["점층", "반어"],
    # 추가 발견 시 여기에 추가
}


# 복합형 마커 판별용 기본 어미 목록
BASIC_ENDINGS = [
    '하여', '하야', '하고', '하며', '하니', '하다', '하라', '하사', '하시',
    '라', '니라', '리라', '러라', '더라', '로다',
    '리오', '잇가', '잇고', 'ㄴ저',
    '어늘', '거늘', '커늘', '커든',
    '면', '어든', 'ㄴ댄', '컨댄',
    '이나', '나', '도', '는', '은',
    '에', '로', '와', '과',
]

# 축약형 복합 마커 (2개 어미가 축약된 형태)
# 에+는→엔, 로+는→론, 에+도→에도, 로+도→로도, 에+는→에는 등
CONTRACTED_COMPOUNDS = [
    '엔',    # 에 + 는
    '론',    # 로 + 는
    '에도',  # 에 + 도
    '로도',  # 로 + 도
    '에는',  # 에 + 는
    '로는',  # 로 + 는
    '하여는', '하여도', '하고는', '하고도',  # 연결+조사
    '리오마는', '리오만',  # 의문+양보
    '하시니이다',  # 존칭 복합
    '나니',  # 하나니의 하(ㆍ) 탈락
    # 미분류에서 발견된 복합형
    '하리다', '하오리다',  # ~하리다 계열
    '코도', '코서', '코야',  # 코(하고) 계열
    '하곤', '하와',  # 하+조사
    '이리다', '로소니', '이러늘',  # 기타 복합
    '하나', '어나', '거나',  # ~나 계열
    '하대', '한데',  # ~대 계열
    '토록',  # ~도록
    '하는', '하신', '하산',  # 관형사형 복합
    '아는',  # 아+는
    '니와',  # 니+와
    '시나', '이시나',  # 시+나
    '란', '으란', '를안',  # 관형+조사
    '호대',  # 한대 + 1인칭
    # 미분류 해소 (2차)
    '로니', '이로니',  # 1인칭 + 이니
    '로라',  # 1인칭 + 라
    '어시늘',  # 어 + 시(존칭) + 늘
    '어니',  # 어 + 니
    '오도',  # 오(1인칭) + 도
    '니다', '시니다', '하시니다',  # 니 + 다
    '온데', '이온데',  # 온 + 데
    '하리이까', '리이까',  # 리 + 이 + 까
    '라한', '라니',  # 라 + 한/니
    '이어시니',  # 이 + 어 + 시 + 니
    'ㄹ새니다', '일새이니다',  # ㄹ새 + 니 + 다
    # 미분류 해소 (3차)
    '시니이까',  # 시 + 니 + 이 + 까 (존칭+의문)
    '이사되',  # 이 + 사(시) + 되 (존칭+상반)
]

# 복합 판별용 핵심 어미 (이것이 포함되고 길이가 3 이상이면 복합 가능성 높음)
COMPOUND_CORE_ENDINGS = [
    '하고', '하니', '하다', '하며', '하여', '하야', '하라', '하사',
    '라고', '라하', '라호', '라면',
    '니이다', '리이다', '소서', '오니', '오라',
    '호되', '호대', '호라',
]

def is_compound_marker(marker):
    """복합형 마커 판별: 2개 이상의 기본 어미 포함 또는 축약형"""
    if pd.isna(marker):
        return False
    m = str(marker)
    
    # 축약형 복합 마커 체크
    if m in CONTRACTED_COMPOUNDS:
        return True
    for cc in CONTRACTED_COMPOUNDS:
        if m.endswith(cc) or cc in m:
            return True
    
    # 핵심 복합 어미 포함 체크 (길이 3 이상, 단 정확히 일치하는 경우는 제외)
    if len(m) >= 3:
        for ce in COMPOUND_CORE_ENDINGS:
            # 핵심 어미 자체는 복합 아님 (예: '하여' 자체는 복합 아님)
            if m == ce:
                continue
            if ce in m:
                return True
    
    # 2글자 이하는 복합 아님 (축약형 제외)
    if len(m) <= 2:
        return False
    
    # 기본 어미 2개 이상 포함 체크
    count = 0
    for e in BASIC_ENDINGS:
        if e in m:
            count += 1
    return count >= 2


def get_base_categories(marker):
    """복합 마커가 어떤 기본 분류들에 해당하는지 반환"""
    categories = []
    for category, info in PREMODERN_TAXONOMY.items():
        for pattern in info["patterns"]:
            if re.search(pattern.replace('^', '').replace('$', ''), marker):
                categories.append(category)
                break
    return categories


def classify_marker(marker):
    """마커를 전근대 분류 체계에 따라 분류
    
    Returns:
        tuple: (분류명, 기본분류_태그리스트)
        - 단일 분류: ("일의상승_하야", [])
        - 복합 분류: ("복합", ["일의상승_하야", "인용"])
    """
    if pd.isna(marker):
        return ("미분류_결측", [])
    
    marker = str(marker)
    
    # 1. 복합형 먼저 체크 (단, 핵심 어미 자체는 제외)
    if is_compound_marker(marker):
        # 복합의 경우 어떤 기본 분류에 해당하는지 태그
        base_cats = get_base_categories(marker)
        return ("복합", base_cats)
    
    # 2. 기본 분류 체크
    for category, info in PREMODERN_TAXONOMY.items():
        for pattern in info["patterns"]:
            if re.match(pattern, marker):
                return (category, [])
    
    return ("미분류", [])


def run_classification():
    """전체 마커 분류 실행"""
    df = pd.read_csv("datasets/phrase_normalized.csv")
    
    # 마커별 빈도
    marker_counts = df['marker_final'].value_counts().to_dict()
    
    # 분류 실행
    classified = defaultdict(list)
    compound_tags = {}  # 복합 마커의 기본 분류 태그
    
    for marker, count in marker_counts.items():
        category, base_cats = classify_marker(marker)
        classified[category].append((marker, count))
        if category == "복합" and base_cats:
            compound_tags[marker] = base_cats
    
    # 정렬 (빈도순)
    for cat in classified:
        classified[cat].sort(key=lambda x: -x[1])
    
    return classified, marker_counts, compound_tags


def save_detailed_report(classified, compound_tags=None):
    """상세 보고서 저장"""
    if compound_tags is None:
        compound_tags = {}
    output_dir = Path("reports/phase4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 마크다운 보고서
    lines = ["# 전근대 원전 기준 현토 분류 결과 (확장판)\n"]
    lines.append("**분류 기준**: 이삼환 《구두지남》, 임규직 《구두해법》, 박문호 《이두해》\n\n")
    lines.append("※ 존칭 변이형(-시-), 축약형 등 포함\n\n")
    lines.append("---\n\n")
    
    # 분류별 요약
    lines.append("## 분류별 요약\n\n")
    lines.append("| 분류 | 고유 마커 | 총 빈도 |\n")
    lines.append("|------|----------|--------|\n")
    
    summary = []
    for cat in sorted(classified.keys()):
        markers = classified[cat]
        total = sum(c for _, c in markers)
        summary.append((cat, len(markers), total))
        lines.append(f"| {cat} | {len(markers)} | {total:,} |\n")
    
    lines.append("\n---\n\n")
    
    # 분류별 상세
    lines.append("## 분류별 마커 상세\n\n")
    
    for cat in sorted(classified.keys()):
        markers = classified[cat]
        total = sum(c for _, c in markers)
        
        if cat in PREMODERN_TAXONOMY:
            info = PREMODERN_TAXONOMY[cat]
            lines.append(f"### {cat}\n\n")
            lines.append(f"**설명**: {info['description']}  \n")
            lines.append(f"**출처**: {info['source']}\n\n")
        else:
            lines.append(f"### {cat}\n\n")
        
        lines.append(f"**고유 마커**: {len(markers)}개, **총 빈도**: {total:,}\n\n")
        
        lines.append("| 마커 | 이형태 | 빈도 |\n")
        lines.append("|------|--------|------|\n")
        all_markers_in_cat = set(m for m, _ in markers)
        for marker, count in markers:
            # 이-/으- 받침 이형태만 (실제 존재하는 것만)
            variants = []
            # 이- 이형태
            if f"이{marker}" in all_markers_in_cat:
                variants.append(f"이{marker}")
            if marker.startswith("이") and len(marker) > 1 and marker[1:] in all_markers_in_cat:
                variants.append(marker[1:])
            # 으- 이형태
            if f"으{marker}" in all_markers_in_cat:
                variants.append(f"으{marker}")
            if marker.startswith("으") and len(marker) > 1 and marker[1:] in all_markers_in_cat:
                variants.append(marker[1:])
            
            variant_str = ", ".join(variants) if variants else "-"
            lines.append(f"| `{marker}` | {variant_str} | {count:,} |\n")
        lines.append("\n")
    
    # 일독양용 섹션 추가
    lines.append("---\n\n")
    lines.append("## 일독양용 (一讀兩用) 마커\n\n")
    lines.append("**설명**: 반어로 두 가지 용법으로 쓰이는 마커 (박문호 33번)\n\n")
    lines.append("| 마커 | 주용법 | 부차용법 |\n")
    lines.append("|------|--------|----------|\n")
    for marker, usages in DUAL_USAGE_MARKERS.items():
        lines.append(f"| `{marker}` | {usages[0]} | {usages[1]} |\n")
    lines.append("\n")
    
    with open(output_dir / "CLASSIFIED_MARKERS.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    # JSON 저장
    json_data = {}
    for cat, markers in classified.items():
        marker_list = []
        for m, c in markers:
            entry = {"marker": m, "count": c}
            if cat == "복합" and m in compound_tags:
                entry["base_categories"] = compound_tags[m]
            marker_list.append(entry)
        
        json_data[cat] = {
            "description": PREMODERN_TAXONOMY.get(cat, {}).get("description", "미분류"),
            "source": PREMODERN_TAXONOMY.get(cat, {}).get("source", ""),
            "markers": marker_list,
            "total_count": sum(c for _, c in markers)
        }
    
    with open(output_dir / "classified_markers.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # 복합 태그 별도 저장
    if compound_tags:
        with open(output_dir / "compound_tags.json", "w", encoding="utf-8") as f:
            json.dump(compound_tags, f, ensure_ascii=False, indent=2)
    
    print(f"저장 완료: {output_dir}")
    return summary


def main():
    print("=" * 70)
    print("전근대 원전 기준 현토 분류 (확장판)")
    print("=" * 70)
    
    classified, marker_counts, compound_tags = run_classification()
    
    print(f"\n총 고유 마커: {len(marker_counts):,}\n")
    
    for cat in sorted(classified.keys()):
        markers = classified[cat]
        total = sum(c for _, c in markers)
        print(f"\n[{cat}] 고유: {len(markers)}, 총: {total:,}")
        
        for m, c in markers[:5]:
            print(f"    {m:20} : {c:>6,}")
        if len(markers) > 5:
            print(f"    ... 외 {len(markers) - 5}개")
    
    summary = save_detailed_report(classified, compound_tags)
    print("\n" + "=" * 70)
    print("보고서 생성 완료")


if __name__ == "__main__":
    main()
