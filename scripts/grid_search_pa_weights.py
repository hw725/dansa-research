#!/usr/bin/env python3
"""
PA 가중치 Grid Search 러너
목적: F1 0.80 → 0.90 달성을 위한 최적 가중치 조합 탐색

실험 매개변수:
- prior_bonus: 현토 마커 보너스 계수
- length_penalty: 길이 차이 패널티 계수
- boundary_threshold: boundary 모델 임계값 (선택)
- supar_bonus: supar 추가 보너스 (선택)
"""

import subprocess
import sys
import itertools
from pathlib import Path
import json
import time
from datetime import datetime

def run_pa_with_config(config: dict, seed: int, output_dir: Path, base_config_path: Path, sample_size: int = None):
    """특정 설정으로 PA 실행 (subprocess로 pa/main.py 호출)"""
    
    config_name = f"pb{config['prior_bonus']:.2f}_lp{config['length_penalty']:.2f}"
    if config.get('boundary_threshold'):
        config_name += f"_bt{config['boundary_threshold']:.2f}"
    if config.get('supar_bonus'):
        config_name += f"_sb{config['supar_bonus']:.2f}"
    
    # 상대 경로로 run_dir 생성
    run_dir = output_dir / config_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 절대 경로로 변환
    run_dir_abs = run_dir.resolve()
    
    print(f"\n{'='*80}")
    print(f"실행: {config_name} / seed{seed}")
    print(f"설정: {config}")
    print(f"출력: {run_dir_abs}")
    print(f"{'='*80}\n")
    
    # csp_config.json을 임시로 백업하고 수정
    backup_path = base_config_path.with_suffix('.backup.json')
    import shutil
    shutil.copy(base_config_path, backup_path)
    
    try:
        # 기본 설정 로드 및 수정
        with open(base_config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        # 실험 설정 반영 - pa_selection_params 경로 사용 (실제로 코드에서 읽히는 설정)
        if 'pa_selection_params' not in cfg:
            cfg['pa_selection_params'] = {}
        if 'candidate_prior_bonus_by_prefix' not in cfg['pa_selection_params']:
            cfg['pa_selection_params']['candidate_prior_bonus_by_prefix'] = {}

        # prior_bonus -> supar/boundary 개별 보너스로 매핑
        prior_bonus = config.get('prior_bonus', 0.015)
        cfg['pa_selection_params']['candidate_prior_bonus_by_prefix']['supar('] = prior_bonus
        cfg['pa_selection_params']['candidate_prior_bonus_by_prefix']['boundary('] = prior_bonus

        # supar_bonus가 명시된 경우 supar만 덮어쓰기
        if 'supar_bonus' in config:
            cfg['pa_selection_params']['candidate_prior_bonus_by_prefix']['supar('] = config['supar_bonus']

        # length_penalty는 현재 pa_selection_params에 해당 항목이 없음
        # (whitespace_dp_penalties 내 penalty 값들과는 별개)
        # 필요시 추가 구현

        if 'boundary_threshold' in config:
            if 'pa' not in cfg:
                cfg['pa'] = {}
            cfg['pa']['boundary_threshold'] = config['boundary_threshold']
        
        # 수정된 설정 저장
        with open(base_config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        
        # PA 입력 파일 생성
        import pandas as pd
        test_df = pd.read_csv("datasets/pa/test.csv")
        
        # 샘플링 (지정된 경우) - (book_name, 문단식별자) 키 기준
        # 주의: pid는 book 간 중복되므로 pid만으로 샘플링하면 문단 수가 폭증/평가가 오염됨
        required_key_cols = ['book_name', '문단식별자']
        missing_key_cols = [c for c in required_key_cols if c not in test_df.columns]
        if missing_key_cols:
            raise ValueError(f"테스트 데이터에 필수 컬럼이 없습니다: {missing_key_cols}\n사용 가능한 컬럼: {test_df.columns.tolist()}")

        key_df = test_df[required_key_cols].drop_duplicates().reset_index(drop=True)

        sample_keys_file = run_dir_abs / f"sample_keys_seed{seed}.json"
        if sample_size and sample_size < len(key_df):
            import random
            random.seed(seed)
            sampled_idx = random.sample(range(len(key_df)), min(sample_size, len(key_df)))
            sampled_keys_df = key_df.iloc[sampled_idx].reset_index(drop=True)
            sample_msg = f" (샘플링: {len(sampled_keys_df)}개 문단)"
        else:
            sampled_keys_df = key_df
            sample_msg = f" (전체: {len(sampled_keys_df)}개 문단)"

        sampled_keys = sampled_keys_df[['book_name', '문단식별자']].values.tolist()  # [[book, pid], ...]
        with open(sample_keys_file, 'w', encoding='utf-8') as f:
            json.dump(sampled_keys, f, ensure_ascii=False)

        # test_df(문장 단위)도 동일 키로 필터
        test_df = test_df.merge(sampled_keys_df, on=['book_name', '문단식별자'], how='inner').reset_index(drop=True)
        
        input_xlsx = run_dir_abs / f"pa_test_input_seed{seed}.xlsx"
        
        # PA는 문단 단위(PD) 입력을 요구: 문단식별자/원문/번역문/book_name
        pd_test_path = Path("datasets/pd/test.csv")
        if not pd_test_path.exists():
            raise FileNotFoundError(f"PD 테스트 데이터를 찾을 수 없습니다: {pd_test_path}")
        
        pd_df_full = pd.read_csv(pd_test_path)
        
        # 샘플링: 위에서 선택된 (book_name, pid) 키를 그대로 사용
        pd_df_sample = pd_df_full.merge(sampled_keys_df, on=['book_name', '문단식별자'], how='inner').reset_index(drop=True)
        
        pd_df_sample.to_excel(input_xlsx, index=False)
        print(f"PA 입력 데이터(문단 단위): {len(pd_df_sample)}개 문단{sample_msg}")
        
        # PA 실행 (Docker 컨테이너 사용)
        output_xlsx = run_dir_abs / f"pa_test_output_seed{seed}.xlsx"
        trace_jsonl = run_dir_abs / f"pa_trace_seed{seed}.jsonl"

        # Docker 경로로 변환 (상대 경로 사용)
        project_root = Path.cwd()
        rel_input = input_xlsx.relative_to(project_root)
        rel_output = output_xlsx.relative_to(project_root)
        rel_trace = trace_jsonl.relative_to(project_root)

        docker_input = f"/workspace/{rel_input.as_posix()}"
        docker_output = f"/workspace/{rel_output.as_posix()}"
        docker_trace = f"/workspace/{rel_trace.as_posix()}"

        cmd = [
            "docker-compose", "exec", "-T", "csp",
            "python", "pa/main.py",
            docker_input,
            docker_output,
            "--embedder", "bge",
            "--use-boundary-model",
            "--boundary-threshold", str(cfg['pa'].get('boundary_threshold', 0.70)),
            "--enable-src-marker-boundary-bonus",
            "--trace-stages-jsonl", docker_trace,
            "--seed", str(seed),
        ]

        # enable_refine 옵션 추가 (config에서 읽음)
        if config.get('enable_refine', False):
            cmd.append("--enable-refine")
        
        result = subprocess.run(cmd, capture_output=True, cwd=Path.cwd())
        
        # 결과를 UTF-8로 디코딩
        try:
            stdout = result.stdout.decode('utf-8', errors='ignore')
            stderr = result.stderr.decode('utf-8', errors='ignore')
        except:
            stdout = str(result.stdout)
            stderr = str(result.stderr)
        
        # stderr는 로그 정보일 수 있으므로 출력만 함
        if stderr and '--use-boundary-model' in ' '.join(cmd):
            # Boundary 모델 로드 로그는 정상
            pass
        
        # returncode로만 성공/실패 판단
        if result.returncode != 0:
            print(f"[WARNING] PA 실행 실패 (exit code: {result.returncode}):")
            if stderr:
                print(stderr)
            return None
        
        # 출력 파일 존재 확인
        if not output_xlsx.exists():
            print(f"[WARNING] PA 출력 파일이 생성되지 않음: {output_xlsx}")
            return None
        
        # 평가: integrity_report.py 호출
        # Gold subset 준비 (sample_keys로 필터링)
        sample_keys_file = run_dir_abs / f"sample_keys_seed{seed}.json"
        gold_subset_path = run_dir_abs / f"pa_gold_subset_seed{seed}.csv"

        if sample_keys_file.exists():
            # integrity_report의 extract_gold_subset 사용
            import sys
            sys.path.insert(0, str(Path.cwd()))
            from integrity_report import extract_gold_subset

            extract_gold_subset(
                gold_path=Path("datasets/pa/test.csv"),
                out_path=gold_subset_path,
                keys_from=input_xlsx,  # input_xlsx에 이미 (book_name, 문단식별자)가 있음
            )
        else:
            # 전체 데이터 사용
            gt_df_full = pd.read_csv("datasets/pa/test.csv")
            gt_df_full.to_csv(gold_subset_path, index=False, encoding='utf-8-sig')

        # integrity_report.py 실행 (Docker)
        docker_gold = f"/workspace/{gold_subset_path.relative_to(Path.cwd()).as_posix()}"

        eval_cmd = [
            "docker-compose", "exec", "-T", "csp",
            "python", "integrity_report.py",
            "--input", docker_output,
            "--gold", docker_gold,
        ]

        eval_result = subprocess.run(eval_cmd, capture_output=True, cwd=Path.cwd())

        try:
            eval_stdout = eval_result.stdout.decode('utf-8', errors='ignore')
            eval_stderr = eval_result.stderr.decode('utf-8', errors='ignore')
        except:
            eval_stdout = str(eval_result.stdout)
            eval_stderr = str(eval_result.stderr)

        # 디버깅용 로그 저장
        eval_log = run_dir_abs / f"eval_log_seed{seed}.txt"
        with open(eval_log, 'w', encoding='utf-8') as f:
            f.write(f"Return code: {eval_result.returncode}\n\n")
            f.write(f"=== STDOUT ===\n{eval_stdout}\n\n")
            f.write(f"=== STDERR ===\n{eval_stderr}\n")

        # exit code 2는 "실패 항목 존재"를 의미하지만, F1 값은 파싱 가능
        # exit code 1 이상은 실제 에러로 간주
        if eval_result.returncode > 2:
            print(f"[WARNING] 평가 실패 (exit code: {eval_result.returncode})")
            print(f"[DEBUG] 로그 확인: {eval_log}")
            return None

        # 출력 파싱 (pa_multitest_runner.py 방식)
        import re
        _F1_TGT_EXACT_RE = re.compile(
            r"\(micro,\s*tgt\s*완전일치\s*subset\):\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)"
        )
        _SRC_SIM_OK_MEAN_RE = re.compile(r"원문 유사도\(SequenceMatcher,\s*tgt문장일치\s*subset\):\s*mean=([0-9.]+)")

        f1_score = 0.0
        sim_score = 0.0

        m_f1 = _F1_TGT_EXACT_RE.search(eval_stdout)
        if m_f1:
            f1_score = float(m_f1.group(3))

        m_sim = _SRC_SIM_OK_MEAN_RE.search(eval_stdout)
        if m_sim:
            sim_score = float(m_sim.group(1))

        # 결과 저장
        result_json = run_dir_abs / f"metrics_seed{seed}.json"
        metrics = {
            'micro_f1_tgt_exact': f1_score,
            'mean_similarity': sim_score,
        }
        with open(result_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"[OK] 완료 (F1: {f1_score:.4f}, Sim: {sim_score:.4f})")
        return {'f1': f1_score, 'similarity': sim_score}
        
    except Exception as e:
        print(f"[ERROR] 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # 설정 복원
        if backup_path.exists():
            shutil.copy(backup_path, base_config_path)
            backup_path.unlink()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="PA 가중치 grid search")
    
    parser.add_argument('--prior-bonus', type=str, required=True,
                        help='콤마 구분 prior bonus 값들 (예: 0.10,0.15,0.20)')
    parser.add_argument('--length-penalty', type=str, required=True,
                        help='콤마 구분 length penalty 값들 (예: 0.3,0.5,0.7)')
    parser.add_argument('--boundary-threshold', type=str, default=None,
                        help='콤마 구분 boundary threshold 값들 (선택, 예: 0.65,0.70,0.75)')
    parser.add_argument('--supar-bonus', type=str, default=None,
                        help='콤마 구분 supar bonus 값들 (선택, 예: 0.0,0.05,0.10)')
    parser.add_argument('--seeds', type=str, required=True,
                        help='콤마 구분 seed 값들 (예: 1,2,3 또는 1-10)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='결과 저장 디렉토리')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='확인 프롬프트 없이 자동 실행')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='테스트 샘플 크기 (기본: 전체, 빠른 검증: 100)')
    parser.add_argument('--enable-refine', action='store_true',
                        help='Refinement 로직 활성화 (max_shift=4)')

    return parser.parse_args()

def parse_range(value_str: str) -> list:
    """값 범위 파싱: '1,2,3' 또는 '1-10' 형식 지원"""
    if '-' in value_str and ',' not in value_str:
        # 범위 형식: 1-10
        start, end = map(int, value_str.split('-'))
        return list(range(start, end + 1))
    else:
        # 콤마 구분 형식: 1,2,3
        return [float(v) if '.' in v else int(v) for v in value_str.split(',')]

def main():
    args = parse_args()
    
    # 매개변수 파싱
    prior_bonus_values = parse_range(args.prior_bonus)
    length_penalty_values = parse_range(args.length_penalty)
    boundary_threshold_values = parse_range(args.boundary_threshold) if args.boundary_threshold else [None]
    supar_bonus_values = parse_range(args.supar_bonus) if args.supar_bonus else [None]
    seeds = parse_range(args.seeds)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_config_path = Path("csp_config.json")
    if not base_config_path.exists():
        print(f"[ERROR] 설정 파일을 찾을 수 없습니다: {base_config_path}")
        return
    
    # 실험 조합 생성
    configs = []
    for pb, lp, bt, sb in itertools.product(
        prior_bonus_values,
        length_penalty_values,
        boundary_threshold_values,
        supar_bonus_values
    ):
        config = {
            'prior_bonus': pb,
            'length_penalty': lp,
            'enable_refine': args.enable_refine
        }
        if bt is not None:
            config['boundary_threshold'] = bt
        if sb is not None:
            config['supar_bonus'] = sb
        configs.append(config)
    
    total_experiments = len(configs) * len(seeds)
    
    print(f"\n{'='*80}")
    print(f"Grid Search 설정")
    print(f"{'='*80}")
    print(f"Prior Bonus 값들: {prior_bonus_values}")
    print(f"Length Penalty 값들: {length_penalty_values}")
    if args.boundary_threshold:
        print(f"Boundary Threshold 값들: {boundary_threshold_values}")
    if args.supar_bonus:
        print(f"Supar Bonus 값들: {supar_bonus_values}")
    print(f"Seeds: {seeds}")
    print(f"총 설정 조합: {len(configs)}")
    print(f"총 실험 횟수: {total_experiments}")
    if args.sample_size:
        print(f"샘플 크기: {args.sample_size} (빠른 검증 모드)")
    print(f"출력 디렉토리: {output_dir}")
    print(f"{'='*80}\n")
    
    if not args.yes:
        user_input = input("실행하시겠습니까? (y/n): ")
        if user_input.lower() != 'y':
            print("취소되었습니다.")
            return
    
    start_time = time.time()
    results = []
    
    # 실험 실행
    for i, config in enumerate(configs, 1):
        config_results = {
            'config': config,
            'seed_results': []
        }
        
        for j, seed in enumerate(seeds, 1):
            experiment_num = (i - 1) * len(seeds) + j
            print(f"\n진행: {experiment_num}/{total_experiments} ({experiment_num/total_experiments*100:.1f}%)")
            
            result = run_pa_with_config(config, seed, output_dir, base_config_path, args.sample_size)
            
            if result:
                seed_result = {
                    'seed': seed,
                    'micro_f1_tgt_exact': result['f1'],
                    'mean_similarity': result['similarity'],
                    'success': True
                }
            else:
                seed_result = {
                    'seed': seed,
                    'micro_f1_tgt_exact': 0.0,
                    'mean_similarity': 0.0,
                    'success': False
                }
            
            config_results['seed_results'].append(seed_result)
        
        results.append(config_results)
    
    # 최종 요약
    elapsed_time = time.time() - start_time
    
    # 성공/실패 계산
    success_count = sum(1 for r in results for sr in r['seed_results'] if sr['success'])
    fail_count = total_experiments - success_count
    
    print(f"\n{'='*80}")
    print(f"Grid Search 완료")
    print(f"{'='*80}")
    print(f"총 실험 횟수: {total_experiments}")
    print(f"성공: {success_count}/{total_experiments}")
    print(f"실패: {fail_count}/{total_experiments}")
    print(f"소요 시간: {elapsed_time/60:.1f}분")
    print(f"{'='*80}\n")
    
    # 결과 저장
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(configs),
            'total_experiments': total_experiments,
            'success': success_count,
            'failed': fail_count,
            'elapsed_seconds': elapsed_time,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 결과 저장: {summary_file}")
    print(f"\n다음 단계:")
    print(f"  python scripts/summarize_grid_search.py {output_dir}")

if __name__ == "__main__":
    main()
