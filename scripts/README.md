# Scripts Index

> **경고**: 스크립트를 하위 폴더로 이동하지 마세요. `sys.path` 의존성이 깨집니다.

---

## 🔧 PA/SA 핵심 (Production)

| 스크립트 | 설명 |
|---------|------|
| `tune_pa_dp.py` | Optuna 기반 PA DP 파라미터 최적화 |
| `run_final_validation.py` | PA 최종 검증 실행 |
| `run_pa_models_only.py` | PA 모델만 실행 |
| `run_pa_nightly_test.py` | PA 야간 테스트 |
| `pa_multitest_runner.py` | PA 다중 테스트 실행기 |

---

## 🎯 Hyeonto 분석 (DO NOT MODIFY)

| 스크립트 | 설명 |
|---------|------|
| `hyeonto_build_datasets.py` | Hyeonto 데이터셋 구축 |
| `hyeonto_build_xlsx.py` | Hyeonto XLSX 생성 |
| `validate_hyeonto_patterns_v6.py` | Hyeonto 패턴 검증 |
| `validate_hypothesis_v6.py` | 가설 검증 |
| `validate_tam_bidirectional_v6.py` | TAM 양방향 검증 |

---

## 📊 분석 스크립트 (analyze_*)

| 스크립트 | 설명 |
|---------|------|
| `analyze_alternative_hypotheses.py` | 대안 가설 분석 |
| `analyze_boundary_leak.py` | 경계 누출 분석 |
| `analyze_boundary_patterns.py` | 경계 패턴 분석 |
| `analyze_cooccurrence_network.py` | 공출현 네트워크 분석 |
| `analyze_failure_patterns.py` | 실패 패턴 분석 |
| `analyze_grid_search_trace.py` | 그리드 서치 트레이스 분석 |
| `analyze_hanja_marker_cooccurrence.py` | 한자 마커 공출현 분석 |
| `analyze_marker_distribution.py` | 마커 분포 분석 |
| `analyze_marker_syntactic_function_v6.py` | 마커 통사 기능 분석 |
| `analyze_ngram_sequences.py` | N-gram 시퀀스 분석 |
| `analyze_pa_errors.py` | PA 오류 분석 |
| `analyze_pa_sa_books.py` | PA/SA 책별 분석 |
| `analyze_phonetic_patterns.py` | 음성 패턴 분석 |
| `analyze_residualized_markers.py` | 잔차화된 마커 분석 |
| `analyze_sentence_level_cooccurrence.py` | 문장 수준 공출현 분석 |
| `analyze_src_matched_selected_margins.py` | 원문 매칭 마진 분석 |
| `analyze_stage_drift.py` | 스테이지 드리프트 분석 |
| `analyze_tam_v6.py` | TAM 분석 v6 |
| `analyze_tense_from_translation_v6.py` | 번역문 시제 분석 |
| `analyze_test_failures.py` | 테스트 실패 분석 |
| `analyze_translation_patterns.py` | 번역 패턴 분석 |
| `analyze_trends.py` | 트렌드 분석 |
| `analyze_weight_sensitivity_v6.py` | 가중치 민감도 분석 |

---

## 📈 시각화 스크립트 (visualize_*)

| 스크립트 | 설명 |
|---------|------|
| `visualize_advanced_boundary.py` | 고급 경계 시각화 |
| `visualize_cluster_flow.py` | 클러스터 플로우 시각화 |
| `visualize_clusters_v6.py` | 클러스터 시각화 v6 |
| `visualize_marker_parent_overlay.py` | 마커-부모 오버레이 시각화 |
| `visualize_pa_sa_sankey.py` | PA/SA Sankey 다이어그램 |
| `visualize_parent_marker_joint_embedding.py` | 부모-마커 공동 임베딩 시각화 |
| `visualize_parent_marker_joint_embedding_ext.py` | 부모-마커 공동 임베딩 확장 |
| `visualize_parent_situations.py` | 부모 상황 시각화 |

---

## 🔬 클러스터링

| 스크립트 | 설명 |
|---------|------|
| `cluster_pa_boundary_functions.py` | PA 경계 기능 클러스터링 |
| `cluster_sa_boundary_functions.py` | SA 경계 기능 클러스터링 |
| `find_optimal_k.py` | 최적 K 탐색 |
| `profile_boundary_clusters.py` | 경계 클러스터 프로파일링 |
| `generate_cluster_visualizations.py` | 클러스터 시각화 생성 |

---

## 🎓 학습 스크립트 (train_*)

| 스크립트 | 설명 |
|---------|------|
| `train_alignment_dual_encoder_trainonly.py` | Dual Encoder 정렬 학습 |
| `train_and_eval_pa_trainonly.py` | PA 학습 및 평가 |
| `train_boundary_cluster_labeler.py` | 경계 클러스터 레이블러 학습 |
| `train_sa_alignment_dual_encoder.py` | SA Dual Encoder 학습 |

---

## 🔍 그리드 서치 / 스윕

| 스크립트 | 설명 |
|---------|------|
| `grid_search_boundary_weights.py` | 경계 가중치 그리드 서치 |
| `grid_search_pa_selection_params.py` | PA 선택 파라미터 그리드 서치 |
| `grid_search_pa_weights.py` | PA 가중치 그리드 서치 |
| `grid_search_sa_weights.py` | SA 가중치 그리드 서치 |
| `sweep_pa_alignment_trainonly.py` | PA 정렬 스윕 |
| `sweep_pa_boundary_threshold.py` | PA 경계 임계값 스윕 |

---

## 🛠️ 유틸리티

| 스크립트 | 설명 |
|---------|------|
| `build_alignment_dataset.py` | 정렬 데이터셋 구축 |
| `build_marker_group_map_from_joint_embedding.py` | 마커 그룹 맵 구축 |
| `collect_tgt_mismatch_cases.py` | 타겟 불일치 케이스 수집 |
| `compare_boundary_mismatch_reports.py` | 경계 불일치 리포트 비교 |
| `compare_final_stages.py` | 최종 스테이지 비교 |
| `compare_pa_sa_clusters.py` | PA/SA 클러스터 비교 |
| `compare_trace_scores.py` | 트레이스 점수 비교 |
| `convert_to_boundary_tagging.py` | 경계 태깅으로 변환 |
| `create_correct_split_with_book.py` | 책별 올바른 분할 생성 |
| `diff_pa_outputs.py` | PA 출력 비교 |
| `extract_markers_from_merged.py` | 병합된 데이터에서 마커 추출 |
| `merge_original_pa_sa.py` | 원본 PA/SA 병합 |
| `merge_pa_sa.py` | PA/SA 병합 |
| `normalize_book_names.py` | 책 이름 정규화 |
| `prepare_pa_clusters_for_validation.py` | PA 클러스터 검증 준비 |
| `prepare_sa_clusters_for_validation.py` | SA 클러스터 검증 준비 |
| `prepare_test_inputs.py` | 테스트 입력 준비 |
| `verify_pa_sa_alignment.py` | PA/SA 정렬 검증 |
| `verify_pa_sa_text_match.py` | PA/SA 텍스트 매칭 검증 |

---

## 📋 요약/리포트

| 스크립트 | 설명 |
|---------|------|
| `aggregate_pa_drift_summary.py` | PA 드리프트 요약 집계 |
| `summarize_boundary_delta_patterns.py` | 경계 델타 패턴 요약 |
| `summarize_grid_search.py` | 그리드 서치 요약 |
| `summarize_mismatch_report.py` | 불일치 리포트 요약 |
| `summarize_pa_drift.py` | PA 드리프트 요약 |
| `summarize_pa_trace_selection.py` | PA 트레이스 선택 요약 |
| `pick_best_from_summary.py` | 요약에서 최적 선택 |
| `scan_pa_results.py` | PA 결과 스캔 |

---

## 🧪 디버그/테스트

| 스크립트 | 설명 |
|---------|------|
| `test_dp_debug.py` | DP 디버그 테스트 |
| `test_runner_sanity_check.py` | 러너 새너티 체크 |
| `deep_analysis_for_0.9.py` | 0.9 달성을 위한 심층 분석 |
| `detect_outliers_boundary.py` | 경계 이상치 탐지 |
| `show_trace_segments.py` | 트레이스 세그먼트 표시 |
| `track_56_stages.py` | 56번 스테이지 추적 |

---

## 📁 기타

| 스크립트 | 설명 |
|---------|------|
| `evaluate_hierarchical_segmentation.py` | 계층적 세그멘테이션 평가 |
| `classify_syntactic_function.py` | 통사 기능 분류 |
| `select_representative_seed.py` | 대표 시드 선택 |
| `inspect_pa_csv_meta.py` | PA CSV 메타 검사 |
| `profile_deep_sa.py` | SA 심층 프로파일 |
| `rebuild_all_reports_v6.py` | 모든 리포트 재구축 |
| `validate_weight_justification.py` | 가중치 정당화 검증 |
| `weight_sensitivity_analysis.py` | 가중치 민감도 분석 |

---

## 📂 archive/

아카이브된 스크립트들 (더 이상 사용하지 않음)
