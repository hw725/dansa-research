#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""현토 음운 패턴 분석 스크립트

현토 마커의 음절 구조, 종성/모음 패턴, 운율적 특성을 분석합니다.
낭송(朗誦)을 위한 리듬 장치로서의 현토 기능을 탐색합니다.

출력:
- phonetic_profile.csv: 마커별 음운 프로파일
- phonetic_analysis.md: 분석 리포트
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# 한글 자모 분해
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 모음 분류
BRIGHT_VOWELS = {'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ'}  # 양성 모음
DARK_VOWELS = {'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ'}  # 음성 모음
NEUTRAL_VOWELS = {'ㅡ', 'ㅢ', 'ㅣ'}  # 중성 모음


def decompose_hangul(char: str) -> Tuple[str, str, str]:
    """한글 1음절을 초성, 중성, 종성으로 분해"""
    if not char or len(char) != 1:
        return ('', '', '')
    
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):
        return ('', '', '')
    
    code -= 0xAC00
    cho = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    
    return (CHOSUNG[cho], JUNGSUNG[jung], JONGSUNG[jong])


def analyze_marker_phonetics(marker: str) -> Dict:
    """마커의 음운 특성 분석"""
    syllables = list(marker)
    n_syllables = len(syllables)
    
    if n_syllables == 0:
        return {
            "syllable_count": 0,
            "has_jongsung": False,
            "final_jongsung": "",
            "final_vowel": "",
            "vowel_harmony": "unknown",
            "bright_ratio": 0.0,
        }
    
    # 각 음절 분해
    decomposed = [decompose_hangul(s) for s in syllables]
    
    # 마지막 음절 분석
    final = decomposed[-1]
    final_jongsung = final[2] if final[2] else ""
    final_vowel = final[1]
    
    # 모음 조화 분석
    vowels = [d[1] for d in decomposed if d[1]]
    bright_count = sum(1 for v in vowels if v in BRIGHT_VOWELS)
    dark_count = sum(1 for v in vowels if v in DARK_VOWELS)
    
    if bright_count > dark_count:
        vowel_harmony = "bright"
    elif dark_count > bright_count:
        vowel_harmony = "dark"
    else:
        vowel_harmony = "neutral"
    
    bright_ratio = bright_count / len(vowels) if vowels else 0.0
    
    return {
        "syllable_count": n_syllables,
        "has_jongsung": bool(final_jongsung),
        "final_jongsung": final_jongsung,
        "final_vowel": final_vowel,
        "vowel_harmony": vowel_harmony,
        "bright_ratio": bright_ratio,
    }


def extract_all_markers(df: pd.DataFrame) -> Counter:
    """데이터프레임에서 모든 마커 추출 및 빈도 계산"""
    # 클러스터 데이터에서 top_markers 컬럼이 있으면 활용
    marker_counts = Counter()
    
    if "top_markers" in df.columns:
        # top_markers 형식: "라:8327; 는:4935; ..."
        for markers_str in df["top_markers"].dropna():
            for item in str(markers_str).split(";"):
                item = item.strip()
                if ":" in item:
                    marker, count = item.split(":")
                    marker_counts[marker.strip()] += int(count.strip())
    else:
        # src_left, src_right에서 직접 추출
        # 주요 마커 패턴
        major_markers = [
            "하니라", "시니라", "니라", "이라", "하나니라",
            "하니", "하여", "하고", "하야", "하다",
            "라", "는", "은", "을", "를", "에", "로",
            "면", "니", "요", "이니", "이요",
            "잇고", "잇가", "러니", "리오", "리라",
            "되", "대", "며", "고"
        ]
        
        for _, row in df.iterrows():
            text = str(row.get("src_left", "")) + str(row.get("src_right", ""))
            for marker in major_markers:
                if marker in text:
                    marker_counts[marker] += 1
    
    return marker_counts


def analyze_phonetic_distribution(
    marker_counts: Counter,
    min_freq: int = 100
) -> pd.DataFrame:
    """마커별 음운 분포 분석"""
    
    results = []
    for marker, freq in marker_counts.items():
        if freq < min_freq:
            continue
        
        phonetics = analyze_marker_phonetics(marker)
        phonetics["marker"] = marker
        phonetics["frequency"] = freq
        results.append(phonetics)
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("frequency", ascending=False)
    
    return df


def compute_genre_phonetic_stats(
    df: pd.DataFrame,
    marker_phonetics: pd.DataFrame
) -> Dict[str, Dict]:
    """장르별 음운 통계 계산"""
    # 클러스터 데이터 활용
    
    genre_stats = {}
    
    if "book_name" not in df.columns:
        return genre_stats
    
    for book in df["book_name"].unique():
        book_data = df[df["book_name"] == book]
        
        # 해당 도서에서 마커 추출
        book_markers = Counter()
        if "top_markers" in df.columns:
            for markers_str in book_data["top_markers"].dropna():
                for item in str(markers_str).split(";"):
                    item = item.strip()
                    if ":" in item:
                        marker, count = item.split(":")
                        book_markers[marker.strip()] += int(count.strip())
        
        if not book_markers:
            continue
        
        # 음운 통계 계산
        total_freq = sum(book_markers.values())
        bright_total = 0
        jongsung_total = 0
        syllable_total = 0
        
        for marker, freq in book_markers.items():
            phonetics = analyze_marker_phonetics(marker)
            bright_total += phonetics["bright_ratio"] * freq
            jongsung_total += freq if phonetics["has_jongsung"] else 0
            syllable_total += phonetics["syllable_count"] * freq
        
        genre_stats[book] = {
            "total_markers": total_freq,
            "avg_bright_ratio": bright_total / total_freq if total_freq > 0 else 0,
            "jongsung_ratio": jongsung_total / total_freq if total_freq > 0 else 0,
            "avg_syllables": syllable_total / total_freq if total_freq > 0 else 0,
        }
    
    return genre_stats


def write_phonetic_report(
    out_dir: Path,
    df_phonetics: pd.DataFrame,
    genre_stats: Dict[str, Dict],
    marker_counts: Counter,
    analysis_type: str
) -> None:
    """음운 분석 리포트 작성"""
    
    lines = [
        f"# {analysis_type} 현토 음운 패턴 분석 리포트",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**분석 마커 수**: {len(df_phonetics)}개",
        "",
        "---",
        "",
        "## 1. 음절 수 분포",
        "",
    ]
    
    if len(df_phonetics) > 0:
        syllable_dist = df_phonetics.groupby("syllable_count")["frequency"].sum()
        lines.append("| 음절 수 | 빈도 | 비율 |")
        lines.append("|:---:|---:|---:|")
        total = syllable_dist.sum()
        for syl, freq in syllable_dist.sort_index().items():
            lines.append(f"| {syl} | {freq:,} | {freq/total*100:.1f}% |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 2. 종성 유무 분포",
        "",
    ])
    
    if len(df_phonetics) > 0:
        jong_dist = df_phonetics.groupby("has_jongsung")["frequency"].sum()
        lines.append("| 종성 | 빈도 | 비율 |")
        lines.append("|:---:|---:|---:|")
        total = jong_dist.sum()
        for has_jong, freq in jong_dist.items():
            jong_label = "있음" if has_jong else "없음"
            lines.append(f"| {jong_label} | {freq:,} | {freq/total*100:.1f}% |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 3. 모음 조화 분포",
        "",
    ])
    
    if len(df_phonetics) > 0:
        harmony_dist = df_phonetics.groupby("vowel_harmony")["frequency"].sum()
        lines.append("| 모음 조화 | 빈도 | 비율 |")
        lines.append("|:---:|---:|---:|")
        total = harmony_dist.sum()
        for harmony, freq in harmony_dist.items():
            lines.append(f"| {harmony} | {freq:,} | {freq/total*100:.1f}% |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 4. 마지막 종성 분포",
        "",
    ])
    
    if len(df_phonetics) > 0:
        final_jong = df_phonetics[df_phonetics["final_jongsung"] != ""]
        jong_counts = final_jong.groupby("final_jongsung")["frequency"].sum().sort_values(ascending=False)
        
        lines.append("| 종성 | 빈도 | 대표 마커 |")
        lines.append("|:---:|---:|:---|")
        for jong, freq in jong_counts.head(10).items():
            examples = df_phonetics[df_phonetics["final_jongsung"] == jong].nlargest(3, "frequency")["marker"].tolist()
            lines.append(f"| {jong} | {freq:,} | {', '.join(examples)} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 5. 마지막 모음 분포",
        "",
    ])
    
    if len(df_phonetics) > 0:
        vowel_counts = df_phonetics.groupby("final_vowel")["frequency"].sum().sort_values(ascending=False)
        
        lines.append("| 모음 | 빈도 | 대표 마커 |")
        lines.append("|:---:|---:|:---|")
        for vowel, freq in vowel_counts.head(10).items():
            examples = df_phonetics[df_phonetics["final_vowel"] == vowel].nlargest(3, "frequency")["marker"].tolist()
            lines.append(f"| {vowel} | {freq:,} | {', '.join(examples)} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 6. 장르별 음운 통계",
        "",
    ])
    
    if genre_stats:
        lines.append("| 도서 | 평균 양성모음 비율 | 종성 비율 | 평균 음절 수 |")
        lines.append("|:---|---:|---:|---:|")
        
        sorted_genres = sorted(genre_stats.items(), key=lambda x: x[1]["avg_bright_ratio"], reverse=True)
        for book, stats in sorted_genres[:15]:
            lines.append(
                f"| {book} | {stats['avg_bright_ratio']*100:.1f}% | "
                f"{stats['jongsung_ratio']*100:.1f}% | {stats['avg_syllables']:.2f} |"
            )
    
    lines.extend([
        "",
        "---",
        "",
        "## 7. 핵심 발견",
        "",
        "### 7.1 음운적 경향성",
        "",
    ])
    
    if len(df_phonetics) > 0:
        # 양성 모음 우세 마커
        bright_markers = df_phonetics[df_phonetics["vowel_harmony"] == "bright"].nlargest(10, "frequency")
        if len(bright_markers) > 0:
            lines.append("**양성 모음 우세 마커 (밝은 느낌):**")
            for _, row in bright_markers.iterrows():
                lines.append(f"- {row['marker']} (빈도: {row['frequency']:,})")
            lines.append("")
        
        # 음성 모음 우세 마커
        dark_markers = df_phonetics[df_phonetics["vowel_harmony"] == "dark"].nlargest(10, "frequency")
        if len(dark_markers) > 0:
            lines.append("**음성 모음 우세 마커 (어두운 느낌):**")
            for _, row in dark_markers.iterrows():
                lines.append(f"- {row['marker']} (빈도: {row['frequency']:,})")
            lines.append("")
    
    lines.extend([
        "",
        "### 7.2 낭송 리듬 가설",
        "",
        "현토의 음절 수와 종성 유무가 낭송 시 리듬감에 기여했을 가능성:",
        "",
        "- 1음절 마커: 빠른 리듬, 간결한 구분",
        "- 2음절 이상: 완결감, 강조 효과",
        "- 종성 있음: 명확한 끊김, 문장 종결",
        "- 종성 없음: 연결감, 흐름 지속",
        "",
        "---",
        "",
        "**분석 완료**",
    ])
    
    (out_dir / f"phonetic_analysis_{analysis_type.lower()}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="현토 음운 패턴 분석")
    p.add_argument("--input-pa", type=str, help="PA 클러스터 CSV")
    p.add_argument("--input-sa", type=str, help="SA 클러스터 CSV")
    p.add_argument("--out-dir", type=str, required=True, help="출력 디렉토리")
    p.add_argument("--min-freq", type=int, default=100, help="최소 빈도 임계값")
    
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드 및 마커 수집
    all_markers = Counter()
    
    if args.input_pa:
        print(f"[1/4] PA 데이터 로딩: {args.input_pa}")
        df_pa = pd.read_csv(args.input_pa, encoding="utf-8-sig")
        pa_markers = extract_all_markers(df_pa)
        all_markers.update(pa_markers)
        print(f"  -> PA 마커 {len(pa_markers)}종")
    
    if args.input_sa:
        print(f"[2/4] SA 데이터 로딩: {args.input_sa}")
        df_sa = pd.read_csv(args.input_sa, encoding="utf-8-sig")
        sa_markers = extract_all_markers(df_sa)
        all_markers.update(sa_markers)
        print(f"  -> SA 마커 {len(sa_markers)}종")
    
    print(f"[3/4] 음운 분석 수행...")
    df_phonetics = analyze_phonetic_distribution(all_markers, min_freq=args.min_freq)
    
    # 프로파일 저장
    df_phonetics.to_csv(
        out_dir / "phonetic_profile.csv",
        index=False, encoding="utf-8-sig"
    )
    
    # 장르별 통계 (PA 또는 SA 데이터 활용)
    genre_stats = {}
    if args.input_pa:
        genre_stats = compute_genre_phonetic_stats(df_pa, df_phonetics)
    elif args.input_sa:
        genre_stats = compute_genre_phonetic_stats(df_sa, df_phonetics)
    
    print(f"[4/4] 리포트 작성...")
    write_phonetic_report(out_dir, df_phonetics, genre_stats, all_markers, "PA+SA")
    
    print(f"완료: {out_dir}")
    
    # 요약 출력
    print("\n=== 요약 ===")
    print(f"분석 마커 수: {len(df_phonetics)}")
    if len(df_phonetics) > 0:
        print(f"평균 음절 수: {df_phonetics['syllable_count'].mean():.2f}")
        print(f"종성 있는 마커 비율: {df_phonetics['has_jongsung'].mean()*100:.1f}%")
        harmony_dist = df_phonetics["vowel_harmony"].value_counts()
        print(f"모음 조화 분포: {harmony_dist.to_dict()}")


if __name__ == "__main__":
    main()
