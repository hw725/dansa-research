#!/usr/bin/env python3
"""SA 경계 태거 테스트 스크립트"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.sa_boundary_tagger_loader import get_sa_boundary_tagger

def main():
    print("=" * 60)
    print("🔍 SA 경계 태거 테스트")
    print("=" * 60)
    
    tagger = get_sa_boundary_tagger()
    
    # 테스트 케이스
    test_cases = [
        "이것은 테스트 문장입니다.",
        "첫 번째 구절입니다. 두 번째 구절입니다. 세 번째 구절입니다.",
        "천하에 다툼이 없어진 지 오래되니, 이에 맹수들을 몰아내고 천지 만물을 화육하게 하셨다.",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n🔹 테스트 {i}: {text[:50]}...")
        segments = tagger.segment_text(text, threshold=0.5)
        print(f"   결과 ({len(segments)}개): {segments}")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
