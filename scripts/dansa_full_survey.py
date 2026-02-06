"""
단사(斷辭) 전수조사 - 병렬 처리 버전
=====================================================
10개 동시 요청으로 속도 향상
"""

import pandas as pd
import numpy as np
import re
import json
import asyncio
import aiohttp
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from openai import AsyncOpenAI
import os

# ============================================================
# 설정
# ============================================================
BATCH_SIZE = 20
CONCURRENT_REQUESTS = 10  # 동시 요청 수
DATA_FILE = "datasets/sentence_full.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LEVEL1_CSV = RESULTS_DIR / "dansa_level1_judgments.csv"
LEVEL2_CSV = RESULTS_DIR / "dansa_level2_judgments.csv"
STATS_JSON = RESULTS_DIR / "dansa_full_survey.json"

client = AsyncOpenAI()

# ============================================================
# 마커 추출 및 정규화
# ============================================================
def extract_marker_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(r'[\uAC00-\uD7AF]+$', text.strip())
    return match.group() if match else ""

def normalize_marker(marker: str) -> str:
    if not marker:
        return ""
    if marker.endswith('로다'):
        return '로다'
    elif marker.endswith('니라'):
        return '니라'
    elif marker.endswith('더라'):
        return '더라'
    elif marker.endswith('러라'):
        return '러라'
    elif marker.endswith('하다'):
        return '하다'
    elif marker.endswith('라') and not marker.endswith('니라') and not marker.endswith('더라') and not marker.endswith('러라'):
        return '라'
    return marker

# ============================================================
# 비동기 LLM 분석
# ============================================================
async def analyze_batch_level1_async(translations: list[str]) -> list[bool]:
    """Level 1: 감탄/여운 분석 (비동기)"""
    texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(translations)])
    
    prompt = f"""다음 번역문들이 "감탄이나 여운을 남기는" 뉘앙스를 담고 있는지 판단해주세요.

"감탄이나 여운"이란:
- 감정적 고양 (탄복, 감탄, 찬탄, 탄식)
- 시적 여운, 열린 마무리
- 정서적 반응을 유발하는 표현

해당되면 O, 아니면 X로 답해주세요.
{texts}
각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""
    
    try:
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            
            
        )
        result_text = response.choices[0].message.content
        
        results = []
        for i in range(1, len(translations) + 1):
            if f"{i}. O" in result_text or f"{i}.O" in result_text:
                results.append(True)
            else:
                results.append(False)
        return results
    except Exception as e:
        print(f"LLM 오류: {e}")
        return [False] * len(translations)

async def analyze_batch_level2_async(translations: list[str]) -> list[bool]:
    """Level 2: 단호한 종결 분석 (비동기)"""
    texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(translations)])
    
    prompt = f"""다음 번역문들이 "단호하게 결론짓는" 뉘앙스를 가지는지 판단해주세요.
"단호한 종결"이란:
- 확정적 단언, 최종 결론
- 더 이상 논의가 필요 없는 완결된 진술
- 강한 확신을 담은 결정적 서술

"약한 종결"이란:
- 확신이 약한 추측, 추정
- 단순 사실 나열, 경과 보고

"단호한 종결"에 해당하면 O, "약한 종결"에 해당하면 X로 답해주세요.
{texts}
각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""
    
    try:
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            
            
        )
        result_text = response.choices[0].message.content
        
        results = []
        for i in range(1, len(translations) + 1):
            if f"{i}. O" in result_text or f"{i}.O" in result_text:
                results.append(True)
            else:
                results.append(False)
        return results
    except Exception as e:
        print(f"LLM 오류: {e}")
        return [False] * len(translations)

def append_to_csv(df: pd.DataFrame, filepath: Path):
    write_header = not filepath.exists()
    df.to_csv(filepath, mode='a', header=write_header, index=False, encoding='utf-8-sig')

async def process_batches_parallel(data_df, translations_col, analyze_func, marker_type, level, csv_path, desc):
    """배치들을 병렬로 처리"""
    translations = data_df[translations_col].tolist()
    total_batches = (len(translations) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # 배치 준비
    batches = []
    for i in range(0, len(translations), BATCH_SIZE):
        batch_trans = translations[i:i+BATCH_SIZE]
        batch_df = data_df.iloc[i:i+BATCH_SIZE].copy()
        batches.append((i, batch_trans, batch_df))
    
    # 세마포어로 동시 요청 제한
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    async def process_one_batch(batch_info):
        i, batch_trans, batch_df = batch_info
        async with semaphore:
            judgments = await analyze_func(batch_trans)
            batch_df = batch_df.copy()
            batch_df['llm_judgment'] = judgments[:len(batch_df)]
            batch_df['marker_type'] = marker_type
            batch_df['analysis_level'] = level
            
            # 즉시 저장
            append_to_csv(batch_df[['book', '문단식별자', '문장식별자', '원문', '번역문', 
                                     'marker', 'llm_judgment', 'marker_type', 'analysis_level']], 
                          csv_path)
            return len(batch_df)
    
    # 진행률 표시와 함께 실행
    pbar = tqdm(total=len(batches), desc=desc)
    
    tasks = [process_one_batch(b) for b in batches]
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
    
    pbar.close()

async def main_async():
    print("=" * 60)
    print("단사(斷辭) 전수조사 - 병렬 처리 버전")
    print(f"동시 요청: {CONCURRENT_REQUESTS}개")
    print("분석 대상: 번역문")
    print("=" * 60)
    
    # 데이터 로딩
    print("데이터 로딩 중...")
    df = pd.read_csv(DATA_FILE)
    df['marker_raw'] = df['원문'].apply(extract_marker_from_text)
    df['marker'] = df['marker_raw'].apply(normalize_marker)
    
    print(f"총 데이터: {len(df):,}건\n")
    
    stats_results = []
    
    # ============================================================
    # Level 1: 로다 전수조사
    # ============================================================
    print("=" * 60)
    print("Level 1: 유사이단 '~로다' 전수조사")
    print("=" * 60)
    
    if LEVEL1_CSV.exists():
        existing_l1 = pd.read_csv(LEVEL1_CSV)
        processed_ids = set(existing_l1['문장식별자'].tolist())
        print(f"기존 결과: {len(existing_l1):,}건 (재개)")
    else:
        processed_ids = set()
    
    roda = df[df['marker'] == '로다'].copy()
    control_ra = df[df['marker'] == '라'].sample(n=len(roda), random_state=42).copy()
    
    print(f"로다 전체: {len(roda)}건, 대조군: {len(control_ra)}건")
    
    roda_todo = roda[~roda['문장식별자'].isin(processed_ids)]
    control_todo = control_ra[~control_ra['문장식별자'].isin(processed_ids)]
    
    print(f"미처리: 로다 {len(roda_todo)}건, 대조군 {len(control_todo)}건")
    
    if len(roda_todo) > 0:
        await process_batches_parallel(roda_todo, '번역문', analyze_batch_level1_async, 
                                        '로다', 'Level1', LEVEL1_CSV, "로다 분석")
    
    if len(control_todo) > 0:
        await process_batches_parallel(control_todo, '번역문', analyze_batch_level1_async,
                                        '라(대조군)', 'Level1', LEVEL1_CSV, "대조군 분석")
    
    # Level 1 통계
    l1_df = pd.read_csv(LEVEL1_CSV)
    roda_results = l1_df[l1_df['marker_type'] == '로다']
    control_results = l1_df[l1_df['marker_type'] == '라(대조군)']
    
    roda_positive = int(roda_results['llm_judgment'].sum())
    control_positive = int(control_results['llm_judgment'].sum())
    
    contingency = [[roda_positive, len(roda_results) - roda_positive],
                   [control_positive, len(control_results) - control_positive]]
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    
    print(f"\n[Level 1] 로다: {roda_positive}/{len(roda_results)} ({roda_positive/len(roda_results)*100:.1f}%)")
    print(f"         대조군: {control_positive}/{len(control_results)} ({control_positive/len(control_results)*100:.1f}%)")
    print(f"         p-value: {p_value:.2e}")
    
    stats_results.append({
        "level": "Level 1", "marker": "로다",
        "n_target": int(len(roda_results)), "n_control": int(len(control_results)),
        "target_positive": roda_positive, "control_positive": control_positive,
        "target_rate": float(roda_positive/len(roda_results)*100),
        "control_rate": float(control_positive/len(control_results)*100),
        "chi2": float(chi2), "p_value": float(p_value), "reject_h0": bool(p_value < 0.05)
    })
    
    # ============================================================
    # Level 2: 니라 vs 라
    # ============================================================
    print("\n" + "=" * 60)
    print("Level 2: '~니라' vs '~라' 전수조사")
    print("=" * 60)
    
    if LEVEL2_CSV.exists():
        existing_l2 = pd.read_csv(LEVEL2_CSV)
        processed_ids_l2 = set(existing_l2['문장식별자'].tolist())
        print(f"기존 결과: {len(existing_l2):,}건 (재개)")
    else:
        processed_ids_l2 = set()
    
    nira = df[df['marker'] == '니라'].copy()
    ra = df[df['marker'] == '라'].copy()
    min_count = min(len(nira), len(ra))
    nira = nira.sample(n=min_count, random_state=42)
    ra = ra.sample(n=min_count, random_state=42)
    
    print(f"니라: {len(nira)}건, 라: {len(ra)}건")
    
    nira_todo = nira[~nira['문장식별자'].isin(processed_ids_l2)]
    ra_todo = ra[~ra['문장식별자'].isin(processed_ids_l2)]
    
    print(f"미처리: 니라 {len(nira_todo)}건, 라 {len(ra_todo)}건")
    
    if len(nira_todo) > 0:
        await process_batches_parallel(nira_todo, '번역문', analyze_batch_level2_async,
                                        '니라', 'Level2', LEVEL2_CSV, "니라 분석")
    
    if len(ra_todo) > 0:
        await process_batches_parallel(ra_todo, '번역문', analyze_batch_level2_async,
                                        '라', 'Level2', LEVEL2_CSV, "라 분석")
    
    # Level 2 통계
    l2_df = pd.read_csv(LEVEL2_CSV)
    nira_results = l2_df[l2_df['marker_type'] == '니라']
    ra_results = l2_df[l2_df['marker_type'] == '라']
    
    nira_positive = int(nira_results['llm_judgment'].sum())
    ra_positive = int(ra_results['llm_judgment'].sum())
    
    contingency2 = [[nira_positive, len(nira_results) - nira_positive],
                    [ra_positive, len(ra_results) - ra_positive]]
    chi2_2, p_value_2 = stats.chi2_contingency(contingency2)[:2]
    
    print(f"\n[Level 2] 니라: {nira_positive}/{len(nira_results)} ({nira_positive/len(nira_results)*100:.1f}%)")
    print(f"         라: {ra_positive}/{len(ra_results)} ({ra_positive/len(ra_results)*100:.1f}%)")
    print(f"         p-value: {p_value_2:.2e}")
    
    stats_results.append({
        "level": "Level 2", "marker": "니라 vs 라",
        "n_nira": int(len(nira_results)), "n_ra": int(len(ra_results)),
        "nira_positive": nira_positive, "ra_positive": ra_positive,
        "nira_rate": float(nira_positive/len(nira_results)*100),
        "ra_rate": float(ra_positive/len(ra_results)*100),
        "chi2": float(chi2_2), "p_value": float(p_value_2), "reject_h0": bool(p_value_2 < 0.05)
    })
    
    with open(STATS_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main_async())
