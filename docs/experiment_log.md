# 연구 실행 기록 — TotalSegmentator V3 소아 적응

이 문서는 **이번 프로젝트의 결정·실행·검증·문제·수정 이력의 색인**이다. 정확한 case 목록과 수치는 각 manifest, 실행 시점의 코드와 환경은 source snapshot·`uv.lock`, 실행 상태는 해당 `run.json`·`status.json`·로그가 원본이다. 채팅의 모든 문장을 복제한 기록은 아니다. 아래의 시점별 상태는 이후 실행으로 바뀔 수 있다.

작성 당시 상태 표본: **2026-10-30 14:20 KST**. 이때 `tsv3.fast_cache` 변환 프로세스가 실행 중이었고 276/673건을 완료했다. `outputs/training_abdominal_v1_seed20261029/run.json`은 여전히 `interrupted`였다. 변환 완료와 학습 재개는 이 표본에서 확인하지 않았다.

## 1. 연구 질문과 범위

- 성인 CT 학습 모델 M0에 소아 32건을 추가 학습한 M1과, 같은 예산으로 성인 32건을 추가 학습한 M2를 비교한다. 주 분석은 소아 test의 장기별 M1−M0, M2−M0 및 M1−M2 변화이며 성인 test에서 성능 유지도 확인한다. 장기별 Dice·NSD·HD95와 개선량의 이질성을 기술한다.
- 단일 TotalSegmentator CT V3 배포본을 사용한다. 현재 7개 장기는 liver, spleen, kidney_left/right, pancreas, adrenal_gland_left/right다. 초기의 117-class probe와 128건/군 후보 배정은 **본 실험 설정이 아니다**.
- 학부 연구형 수업 프로젝트다. seed 20261029의 M0/M1/M2 각 1회가 우선이다. 여력이 있어 반복하면 모든 seed를 보고하고 좋은 seed만 선택하지 않는다. 모델 수렴, 임상적 성능, 탑티어 수준의 새 방법을 사전에 주장하지 않는다.

## 2. 사용자가 확정한 연구 판단

| 판단 | 현재 규칙과 한계 | 고정 근거 |
|---|---|
| 분할 단위 | 환자 연결 ID가 없어 **case 단위 탐색 연구**로 진행. 동일 CT content 중복은 제외했지만 같은 환자의 다른 촬영은 완전히 확인할 수 없음 | `configs/study.json`의 `cohort.split_unit_decision`; `manifests/splits_abdominal_v1/splits.json` |
| 라벨 사용 | 공개 reference mask + 자동 기술 QC + 기록된 수동 보류 제외. 전체 volume 전문가 검수나 임상적으로 깨끗한 annotation이라고 주장하지 않음 | `configs/study.json`의 `cohort.label_policy_decision`; `manifests/review_v300/review.csv` |
| 거리 단위 | header 물리 단위가 unknown이므로 HD95는 voxel, NSD 허용 거리는 2 voxel. mm·mm³ 환산 주장 없음 | `configs/study.json`; `manifests/evaluation_abdominal_v1/contract.json` |
| 학습 반복 | 우선 한 seed의 세 모델을 모두 실행. 나중에 추가 반복 여부 판단 | `configs/study.json`의 `paired_training_seeds`, `replication_policy` |
| 긴 실행 | 사용자가 터미널에서 실행. Codex는 짧은 점검·코드 수정·결과 해석을 수행 | 이 세션의 사용자 지시; `README.md` 실행 명령 |
| 보관·배치 | raw 보존, 원인과 제외를 manifest로 기록, 설정은 `configs/`, 공유 코드는 `src/tsv3/`, 입력·QC·split은 `manifests/`, 실행 산출물은 `outputs/`. 재생성 가능한 임시 파일만 검증 후 정리 | `README.md` 코드와 산출물의 역할 |

## 3. 수행 경과와 근거

| 단계 | 확인된 진행 | 원본 기록 |
|---|---|---|
| 저장소·환경 | 프로젝트 root `/home/anna/projects/total_segmentator_v3`, Git origin `Laplace-tech/totalsegmentator-v3-pediatric-adaptation`. Python 3.12, 환경은 `pyproject.toml`·`uv.lock`. 2026-10-30 현재 다수 소스·설정 파일이 Git에서 untracked이므로 원격 Git만으로 재현된다고 주장할 수 없음 | `git status`, `git remote -v`, `uv.lock` |
| 원본 inventory·QC | CT 1,939건과 mask 226,863개를 검사. age 결측·100세 초과·정확한 CT 중복·기술 오류를 기록. test `s1446`과 같은 CT인 train `s1445`를 격리하고 train 중복 `s2107/s2109/s2110`도 제외. 빈 mask·경계 접촉은 검토 flag로 취급 | `manifests/audit_v300/`, `manifests/qc_v300/`, `outputs/stage123_report.md` |
| 초기 후보 split·시험 | M1/M2 각 128건의 **과거 후보** split과 117-class·3mm probe를 실행. 분할과 사용 장기·해상도가 현재 연구와 달라 본 학습 성능 근거로 사용하지 않음. 시험용 checkpoint·예측·NPZ는 정리하고 측정 JSON·로그 보존 | `manifests/splits_v300/`, `outputs/feasibility_v300/`, `outputs/stage123_report.md` |
| 원본 ZIP 정리 | 압축 해제 raw에 대한 QC와 ZIP 항목 수 확인 뒤 2026-10-29 원본 ZIP 삭제. raw CT·mask 자체는 보존 | `outputs/stage123_report.md`, `README.md` |
| 장기 후보·검토 | 7장기 panel과 QC 기반 검토 표를 생성. 소아 133건·성인 1,086건 검토 이미지가 생성됐지만 전체 전문가 검토 완료를 뜻하지 않음. 유일한 수동 판정 ledger는 `review.csv` | `manifests/review_v300/`, `outputs/review_v300/index.html` |
| 현재 split 고정 | seed 20261029: M0 성인 학습 577, M1 소아 32, M2 성인 32, 성인·소아 validation 각 16, 미사용 소아 development 6. 공식 test 성인 88·소아 20은 학습·설정 선택에서 격리. M2와 validation은 M0에서 미리 제외 | `manifests/splits_abdominal_v1/splits.json`, `configs/study.json` |
| 7장기 native-grid feasibility | patch 96×128×128, batch 1, 세 분기 각 7 update, 동일 M0 초기 상태, AMP skip 0, 전체 영상 추론 확인. 실행 가능성 증거이며 정확도 결과 아님 | `outputs/feasibility_abdominal_v1/result.json`, `README.md` |
| 공유 전처리·통합 시험 | development 673건의 손실 없는 원본 NPZ cache 완료. M0 학습 577건만으로 정규화. overlap은 학습 target에서 ignore. cache·source hash, 다음 patch·checkpoint 재개, M1/M2 동일 초기 상태를 통합 시험. 시험용 가중치는 정리 | `data/preprocessed/abdominal_v1/manifest.json`, `normalization.json`, `outputs/overnight_abdominal_v1/` |
| 야간 자동화 | 전처리 완료 대기→통합 시험, 성공 후 Windows 종료 감시를 한 차례 수행. 종료 명령 exit code 0은 기록됐으나 실제 물리적 전원 차단은 별도 측정하지 않음. 이 자동 종료는 현재 본 학습 명령에 연결되어 있지 않음 | `outputs/overnight_abdominal_v1/status.json`, `result.json`, `shutdown.json`, `README.md` |
| 평가 규칙 고정 | 학습 전 test 108 case × 7장기 eligibility를 기록. 장기별 주 분석 소아 표본은 6–9건으로 작음. 공개 mask의 QC 기반 평가이며 전문가 검수 결과가 아님. hard Dice·NSD·HD95 함수 시험 완료 | `manifests/evaluation_abdominal_v1/contract.json`, `eligibility.csv`, `src/tsv3/evaluation.py` |
| 실행 smoke | M0/M1/M2 각 32 update, 모든 development validation 영상 추론, checkpoint 복원 및 동일 초기 상태를 확인. 총 약 16.5분. smoke loss·Dice를 결과나 설정 선택에 사용하지 않음 | `outputs/training_smoke_abdominal_v1/run.json`, branch status·validation JSON, `README.md` |
| 본 학습 첫 실행 | 실제 M0를 시작했으나 GPU 공급 병목 확인 후 사용자가 Ctrl+C로 중단. 중단 상태는 1,352 updates, 마지막 정상 `latest.pt`는 **1,250 updates**. 따라서 재개하면 checkpoint 이후 102 updates를 다시 수행 | `outputs/training_abdominal_v1_seed20261029/run.json`, `M0/status.json`, `M0/latest.pt`, `run.log` |
| GPU 병목 진단 | 30초 표본 GPU utilization 평균 약 25.4%, model/optimizer 값 finite·AMP skip 0. 별도 CPU 프로파일에서 `s2029` 기존 NPZ 로드 0.603초, 압축 해제 0.298초. 직렬 whole-volume DEFLATE 읽기가 유력한 공급 병목. 당시 학습 정확도는 아직 평가되지 않음 | `outputs/training_abdominal_v1_seed20261029/monitoring.json` |
| 읽기 경로 수정 | 원본 cache와 학습 수치 설정을 보존하면서 Blosc2 ZSTD level 1 + shuffle 형식의 별도 lossless cache·manifest·재개 변환을 추가. `s0004` 표본 read 중앙값 0.275→0.065초. 실제 M0 checkpoint의 다음 `s0551` case에서 원본/새 cache의 patch·augmentation 텐서와 RNG가 정확히 일치. 과거 source snapshot 대비 과학적 학습 함수 AST 동일, 단위 시험 30개 통과, 임시 디렉터리 checkpoint 이관 시험 통과. **본 학습 GPU utilization 개선은 아직 관측하지 않음** | `src/tsv3/fast_cache.py`, `training.py`, `tests/test_protocol.py`, `README.md` |
| 현재 변환 | 사용자 터미널에서 `tsv3.fast_cache --resume --host-volume /mnt/c` 실행 중. 위 확인 시점 276/673건, `manifest.json.status=running`. 변환 완료 시 명령의 `&&`로 checkpoint 이관 검사 후 본 학습을 재개할 예정. 원본 run은 아직 `interrupted` | `data/preprocessed/abdominal_v1_fast/manifest.json`, 실행 프로세스, 원본 `run.json` |

## 4. 현재 고정된 학습·평가 계약

- `configs/study.json`이 단일 설정 원본이다. M0는 50,000, M1/M2는 각각 5,000 **성공한 optimizer updates**. M1/M2는 동일한 선정 M0 checkpoint에서 시작하고 optimizer를 각각 새로 만든다.
- PlainConvUNet, 7장기+background, patch 96×128×128, batch 1, SGD Nesterov, M0 LR 0.01, M1/M2 LR 0.001, polynomial power 0.9, soft Dice+CE, AMP FP16, gradient clip 12. Sampling·intensity augmentation·validation·checkpoint 세부 수치는 `README.md`와 `study.json`을 따른다. 공간 augmentation은 없다.
- M0는 2,500 updates마다 adult validation macro Dice 최대 checkpoint를 선택한다. M1/M2는 같은 5,000-update 마지막 checkpoint를 사용한다. 매 250 updates마다 원자적으로 저장한다. 이 규칙은 test를 보고 바꾸지 않는다.
- Test Dice·NSD·HD95는 원본 CT index grid의 voxel 단위다. 장기별 eligibility가 다르고 표본 수가 작아 설명 변수와 개선량 관계는 탐색적으로 해석한다. 환자 ID 부재, 선택 편향, 제한된 라벨 검토, 공간 단위 불확실성을 결과에 기재한다.
- 최종 test 추론·집계 실행은 **아직 미완료**다. 현재 구현된 metric 함수와 고정 eligibility가 최종 성능 결과를 의미하지 않는다.

## 5. 현재 실행의 보관·재개 규칙

- 원본 `data/preprocessed/abdominal_v1/`와 `normalization.json`은 삭제·수정하지 않는다. 고속 복제 `data/preprocessed/abdominal_v1_fast/`가 673/673 complete인지 확인한 뒤에만 사용한다. `--host-volume /mnt/c`와 최소 10 GiB 여유 검사는 WSL 가상디스크의 Windows host 공간 부족을 막는다.
- 첫 재개는 `README.md`의 `fast_cache --resume && training --resume --fast-cache ... --upgrade-fast-cache` 명령이다. 이관은 원본 checkpoint·run·source snapshot의 identity, 입력 동등성, 코드 함수 동일성을 확인하고 원본을 백업한다. 이관 성공 전에는 새 provenance의 본 학습이 시작됐다고 기록하지 않는다.
- 재개 후 다시 중단되면 `training --resume --fast-cache ...`를 사용한다. `run.json.cache_upgrade`, 기존/신규 source snapshot, `M0/pre_fast_cache_upgrade.pt`는 이관의 감사 자료다. 선정 모델과 최종 test 분석 전 임의로 삭제하지 않는다.
- 현재 source와 설정의 상당 부분이 Git untracked다. Git commit 또는 원격 push가 이 기록의 완료 조건으로 자동 수행된 것은 아니다. 실행 시점의 보관된 snapshot과 각 파일 hash를 먼저 확인하고, Git 정리는 별도 검토한다.

## 6. 다음 확인 지점

1. 고속 cache 673/673 완료 및 manifest `status=complete`, 오류 없음. host 여유 공간 확인.
2. 첫 checkpoint 이관 감사 통과 여부: `run.json.cache_upgrade`, 원본 백업, `source_snapshot_fast.tar.gz` 확인. 실패하면 학습을 시작하지 않고 원인 기록.
3. M0 재개 후 실제 seconds/update, GPU utilization, CPU/RAM/swap, AMP skip, checkpoint 진행 및 첫 adult validation 확인. 변환 표본 속도만으로 전체 가속을 주장하지 않음.
4. M0 선정 후 M1/M2가 동일한 M0 weight hash와 고정 예산으로 완료되는지 확인.
5. 완료된 세 모델의 고정 test 추론·지표 집계·장기별 paired 분석·한계 보고. Test 결과를 보고 설정을 재선택하지 않음.

기록을 갱신할 때는 기존 결과를 소급해 바꾸지 말고 날짜·관측 근거·변경 이유를 추가한다. 서로 다른 실험 설정은 새 version/run 경로에 저장한다.

## 2026-10-30 14:29–14:31 KST — 고속 cache 완료 후 저장공간 점검

이 항목은 첫 재개 실행의 `source_snapshot_fast.tar.gz`가 만들어진 **이후** 추가한 운영 기록이다.

- 사용자 요청으로 Windows host C: 공간 감소 원인을 확인했다. `data/preprocessed/abdominal_v1_fast/manifest.json`은 673/673 `complete`였고 크기는 약 21 GiB였다. 기존 원본 NPZ cache는 약 26 GiB, raw는 약 50 GiB다. 새 cache가 생긴 것이 C: 여유가 약 48→28 GiB로 감소한 주된 이유다. 이 수치는 표시 단위 반올림을 포함한다.
- 본 학습 `run.json.status=running`, cache 이관 감사 기록 존재, M0는 14:31 KST 확인 시 1,850 updates로 진행 중이었다. 원본 cache·고속 cache·raw·`.venv`·검토 이미지·checkpoint 백업은 사용 중이거나 연구 근거이므로 삭제하지 않았다.
- 재생성 가능한 `/home/anna/.cache/uv` 약 88 MiB, `/home/anna/.cache/pip` 약 48 MiB, Windows Temp의 오래된 VS Code 설치 실행 파일 약 240 MiB를 정확한 실제 경로와 symlink 여부를 확인한 뒤 삭제했다. Windows Temp의 `swap.vhdx` 약 2.5 GiB는 활성 WSL swap이므로 보존했다. 정리 직후 C: 표시는 약 27 GiB였으며, 진행 중인 WSL 학습 쓰기와 표시 단위 때문에 삭제량만큼의 순증을 관측하지 못했다.
- 현재 학습은 10 GiB host 여유 보전 검사를 checkpoint 저장 전에 수행한다. 큰 폭의 추가 확보는 원본 26 GiB cache의 보존·이동 정책과 WSL 가상디스크 회수를 별도로 정해야 하며, 실행 중인 학습을 방해하지 않는 시점에 처리한다. 이 저장공간 점검은 모델 설정·split·평가 규칙을 변경하지 않았다.

## 2026-10-30 16:18 KST — M0 중간 점검

- 확인 시 M0 16,350/50,000 updates, `run/status=running`, AMP skip 0, 최근 20개 기록 loss 모두 finite였다. `configs/study.json`과 실행의 `study_snapshot`이 같고, 읽은 `latest.pt`(16,250)·`selected.pt`(15,000)의 config와 provenance가 현재 run과 일치했다.
- 성인 validation 16건의 macro Dice는 2,500/5,000/7,500/10,000/12,500/15,000 updates에서 각각 0.1413/0.3792/0.5662/0.6324/0.7897/0.8796이었다. 초기 4장기의 Dice 0은 이후 해소됐다. 15,000-update 장기 평균은 liver 0.9648, spleen 0.9598, kidney_left 0.9388, kidney_right 0.9468, pancreas 0.8486, adrenal_gland_left 0.8313, adrenal_gland_right 0.6671이다. 성인 validation에서의 학습 진전이며 소아 적응 이득 또는 test 성능 결과는 아니다.
- 10개 GPU 표본의 utilization 평균 42.2%, 범위 1–86%, VRAM 3,821–3,852 MiB, 온도 59–64°C였다. 데이터 공급 대기 등으로 사용률이 변동하므로 단일 0% 표본으로 중단을 판단하지 않는다. RAM available 약 11 GiB, swap 사용 약 3.2 GiB, kernel `oom_kill=0`, C: 여유 약 104 GiB를 관측했다. 이전 C: 여유 증가의 원인은 이번 점검에서 규명하지 않았다.
- 재개용 `latest.pt`는 모든 branch에서 250 successful updates마다 원자적으로 교체한다. M0 `selected.pt`는 매 2,500 updates의 adult validation macro Dice가 최고값을 갱신할 때 교체하고, 동점은 이전 것을 유지한다. M1/M2 `selected.pt`는 각 5,000 updates 완료 시 저장한다. 개별 update별 모델 파일을 모두 누적 보관하지 않는다. 이번 점검은 실행 중인 source/config를 수정하지 않았다.

## 2026-10-30 16:21 KST — 50,000-update 예산과 과적합 위험 검토

- M0 약 16,900 updates 시점에 고정 예산의 적절성을 재검토했다. 50,000 updates는 사전 선언한 자원 제한 탐색 예산이며 최적 학습량 또는 수렴을 입증한 값이 아니다. batch 1·case 균등 복원 추출이므로 577건에서 case당 기대 patch 추출 수는 약 86.7회다. 전체 volume을 86.7번 학습하는 epoch와 같지 않다.
- 관측된 성인 validation macro Dice는 10,000→12,500→15,000에서 0.6324→0.7897→0.8796으로 상승했다. 현재까지 지속적 validation 악화나 명확한 plateau는 관측되지 않았다. 12,500의 spleen 점수 일시 하락 후 회복처럼 장기별 변동은 존재한다. 같은 평가 조건의 train 전체 영상 Dice를 측정하지 않아 train–validation gap과 과적합 정도를 직접 정량화할 수는 없다.
- validation 16건으로 여러 checkpoint를 선택하므로 표본 변동과 model-selection bias가 남는다. M0는 고정된 adult validation best checkpoint를 사용하는 규칙을 유지하며, 50,000번째 가중치를 무조건 선택하지 않는다. 이후 validation·장기별 점수의 반복 하락은 점검 사유로 기록한다. Test를 이용해 예산·LR·선정 규칙을 조정하지 않는다.
- M1/M2는 각 32건으로 5,000 updates를 수행해 case당 기대 patch 추출 수가 약 156.3회다. 작은 cohort와 제한된 augmentation은 과적합 가능성을 고려할 이유이며, 실제 발생 여부는 해당 branch validation 추세로 확인해야 한다. 현재 실행 중인 M0 예산·schedule 및 두 fine-tuning군의 동일 예산·마지막 checkpoint 선정 규칙은 변경하지 않았다.
- 참고로 로컬 nnunetv2 2.8.1 기본 trainer는 250 iterations/epoch × 1,000 epochs를 선언한다. batch·architecture·augmentation이 달라 현재 50,000의 정당성을 직접 보증하는 비교는 아니다. 근거: [nnU-Net trainer](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py), [Cawley & Talbot, model selection bias](https://www.jmlr.org/papers/v11/cawley10a.html).
- 운영상 AMP skip은 16,650-update 로그에서 처음 1회로 기록됐고 16,975까지 추가 skip 없이 진행됐다. 이는 과적합 지표가 아니며 성공 update 예산에서 제외된다.

## 2026-10-30 17:48–18:08 KST — M0 압축 해제 오류와 재개 전 점검

- M0는 29,028 successful updates에서 `RuntimeError: Cannot decompress`로 중단됐다. 실패 시도는 `trainer_state.attempts=29033`, AMP skip 누계 5였다. 오류 지점은 `CaseCache.get`의 Blosc2 `unpack_array`이며, `run.json`과 `M0/status.json` 모두 `failed`로 기록했다. 마지막 원자적 `latest.pt`는 29,000 updates, 최고 성인 validation `selected.pt`는 27,500 updates에서 macro Dice 0.927868이다.
- 중단 후 빠른 cache 673건 모두를 읽기 전용으로 검사했다. 각 파일의 크기·SHA256이 manifest와 일치했고 `data`·`seg`·`foreground_locations` 압축 해제 오류는 0건이었다(약 141.6초). checkpoint의 NumPy RNG 소비를 28번의 저장 이후 시도에 대해 재현한 결과 실패 다음 case는 `s1891`로 **추정**됐다. 이 파일도 hash가 일치했고 단독 50회 반복 압축 해제에서 오류가 없었다. 실제 오류 case ID를 실행 로그에 직접 기록하지 않았으므로 추정이라고 표기한다.
- 마지막 `latest.pt`와 `selected.pt`를 읽어 model·optimizer tensor가 모두 finite인지 확인했다. 두 checkpoint의 config·provenance는 현재 run과 일치했다. 학습 코드, study, 원본/고속 cache manifest, lock, 이관 source snapshot의 hash도 모두 실행 기록과 일치했다. 디스크 여유와 OOM 기록에서 부족 징후를 확인하지 못했다. **파일 손상이나 확정된 라이브러리 결함은 입증되지 않았고, 일시적인 압축 해제 실패가 가능한 설명**이다.
- 원본·고속 cache 및 연구 설정을 바꾸지 않은 상태에서 마지막 29,000-update checkpoint로 재개할 수 있다. 28 successful updates는 다시 수행된다. 재개 후 같은 위치에서 다시 실패하면 반복 가능한 결함으로 재분류해 원인을 추가 조사한다. 긴 본 학습 재실행은 사용자 터미널에서 수행한다.

## 2026-10-30 18:35 KST — 재개 후 M0 상태와 성인 validation 추세

- 사용자 재개 후 M0가 오류 지점을 넘어 32,500/50,000 successful updates까지 진행 중이었다. `run.json`과 `M0/status.json` 모두 `running`, 오류 없음, AMP skip 누계 7, 최근 30개 로그의 loss는 모두 finite. M1/M2는 아직 시작하지 않았다. C: 여유 약 102 GiB, kernel `oom_kill=0`이었다.
- 성인 validation 16건의 macro Dice는 22,500/25,000/27,500/30,000 updates에서 0.907734/0.908875/**0.927868**/0.900931이었다. 현재 최고 선정 모델은 27,500-update `M0/selected.pt`다. 30,000에서 1회 하락했으며, 반복적인 하락 또는 과적합으로 확정할 근거는 아직 부족하다. 각 장기 결과는 `M0/validation_*.json`이 원본이다.
- 32,500-update validation 진입 시 GPU 10초 표본 utilization 평균 84.2%(범위 82–87%), VRAM 약 2.5 GiB, 온도 67–68°C를 관측했다. 단일 GPU 표본 0%는 validation의 case 전환 시점에 발생할 수 있어 프로세스·status·여러 표본을 함께 확인했다. 이 점검은 성인 development validation이며 소아 적응이나 test 성능 결과가 아니다.

## 2026-10-30 18:47 KST — M0 성인 validation 곡선

- `M0/validation_002500.json`부터 `validation_032500.json`까지 13회 × 성인 16건의 결과로 macro 및 7장기 Dice 곡선(`outputs/training_abdominal_v1_seed20261029/plots/m0_adult_validation_dice.png`)을 생성했다. 같은 그림의 SVG도 `plots/`에 보존했다. 각 JSON의 case 수 16, macro Dice와 7장기 평균의 일치를 검사한 뒤 그렸다. 실행 중인 학습은 변경하지 않았다.
- 최고는 27,500 updates의 macro Dice 0.927868이다. 30,000에서 0.900931, 32,500에서 0.772584로 하락했다. 특히 왼쪽 adrenal 평균 Dice는 0.898741→0.131201로 내려갔고 **성인 validation 16건 모두**에서 해당 장기 Dice가 하락했다. 한 case의 우연한 변동으로 설명할 수 없는 cohort 전반의 퇴행이다. 원인은 아직 확인되지 않았으며, 다음 validation에서 회복 여부를 확인해야 한다. 최고 선정 checkpoint는 27,500 updates에 보존돼 있다.

## 2026-10-30 19:50 KST — M0 후기 validation 점검

- M0가 42,525/50,000 successful updates까지 진행 중이었고 `run.json`/branch status 모두 `running`, 오류 없음, AMP skip 누계 11, 최근 loss는 finite였다. GPU 단일 표본 utilization 70%, VRAM 약 3.8 GiB, 온도 61°C, C: 여유 약 102 GiB, OOM 0을 확인했다. 소아 M1과 성인 대조 M2는 아직 시작하지 않았다.
- 32,500-update 성인 validation의 macro Dice 0.772584와 왼쪽 adrenal 0.131201은 35,000에서 각각 0.907942와 0.8317로 회복했다. 이후 macro Dice는 37,500/40,000/42,500에서 0.922540/0.921007/**0.930741**이었다. 42,500-update가 현재 선정 M0 checkpoint이며, 왼쪽 adrenal 평균은 0.8969, 오른쪽 adrenal은 0.8502다. 급락은 일시적이었지만 원인을 규명한 것은 아니고, 성인 validation 16건에서 반복 선택한 점수이므로 일반화 성능은 공식 test 전까지 미확인이다.
- 기존 `plots/m0_adult_validation_dice.png`와 `.svg`를 42,500-update까지의 17개 평가 시점으로 갱신했다. 입력 JSON의 성인 16건·macro와 7장기 평균의 일치를 다시 확인했다. 실험 설정·학습 프로세스는 변경하지 않았다.

## 2026-10-30 20:56 KST — M0 완료, M1 진행

- M0는 50,000/50,000 successful updates를 완료했다. AMP skip 13회, `M0/status.json.status=complete`, 마지막 성인 validation 16건의 macro Dice **0.9373953960**이 20회 평가 중 최고로 50,000-update `selected.pt`를 선택했다. 장기별 평균 Dice 순서(liver, spleen, kidney_left/right, pancreas, adrenal_gland_left/right)는 **0.980381/0.974023/0.968228/0.971722/0.889082/0.913897/0.864435**다. 이 점수는 성인 development validation이며 test 성능이 아니다.
- `selected.pt` SHA256은 `fa21372654af1992ca580a033fb6a10b46a436bfd949364fa7bd709dc6dd447b`로 `status.json`과 일치했다. M1의 1,250-update checkpoint가 기록한 시작 M0 checkpoint SHA와 초기 가중치 hash도 이 선정 M0와 일치했다.
- 같은 실행 프로세스가 자동으로 M1에 진입해 점검 시 1,300/5,000 updates, 오류 없음, AMP skip 0회였다. 1,000-update M1 validation macro Dice는 성인 0.911968, 소아 0.796694(각 16건)다. M0의 소아 validation 추론 결과는 아직 없어 이 수치만으로 소아 적응 이득을 계산할 수 없다. M2는 아직 시작하지 않았다.
- `plots/m0_adult_validation_dice.png`와 `.svg`를 M0 완료 시점의 20개 성인 validation 기록으로 갱신했다. 이번 점검에서는 학습 설정과 실행 중인 프로세스를 수정하지 않았다.

## 2026-10-30 22:27 KST — 세 branch 학습 완료와 development validation 집계

- `run.json` 및 `M0`·`M1`·`M2`의 `status.json`이 모두 `complete`다. successful updates는 M0 50,000, M1/M2 각 5,000이며 AMP skip은 각각 13/0/0회다. 세 `selected.pt` 파일의 SHA256이 각 status 기록과 일치하고, M1·M2의 초기 가중치 hash가 일치한다.
- 최종 selected checkpoint의 development validation macro Dice(각 cohort 동일 16건): 성인 M0 **0.937395**, M1 **0.905337**, M2 **0.938348**; 소아 M1 **0.817575**, M2 **0.674918**. M0 소아 validation은 학습 중 측정하지 않았다. M1−M2 소아 macro 차이는 **+0.142657**, M1−M0 성인 macro 차이는 **−0.032058**이다. 이 차이는 각 16건의 development validation 관찰값이며 test 효과 추정치가 아니다.
- 각 JSON의 case별 7장기 Dice에서 장기별 평균과 macro를 재계산해 기록값과 일치함을 확인했고, branch 간 cohort별 case ID가 동일함을 확인했다. 집계 CSV(`outputs/training_abdominal_v1_seed20261029/validation_summary.csv`), 장기별 비교(`outputs/training_abdominal_v1_seed20261029/plots/validation_organ_comparison.png`), fine-tuning 경과(`outputs/training_abdominal_v1_seed20261029/plots/validation_trajectory.png`)를 같은 run 폴더에 저장했다. 그래프 SVG도 함께 보관했다.
- 공식 test 추론 및 Dice·NSD·HD95 평가는 아직 수행하지 않았다. M0 소아 baseline과 최종 adaptation gain은 공식 test의 동일 case·장기 비교 이후 보고한다. 오늘은 추가 학습이나 test 추론을 시작하지 않았다.

## 2026-10-30 22:33 KST — 장기별 validation 해석

- 최종 M1·M2 소아 validation 동일 16건에서 M1−M2 평균 Dice 차이는 간 +0.031219, 비장 +0.038672, 왼쪽 신장 +0.152806, 오른쪽 신장 +0.068814, 췌장 +0.118093, 왼쪽 부신 +0.403915, 오른쪽 부신 +0.185081이다. 양쪽 부신이 7장기 균등평균 차이의 약 59%를 차지한다. 왼쪽 부신은 16/16건에서 M1 우세지만 M1의 절대 Dice는 0.626913으로 낮다.
- 왼쪽 신장 차이는 평균 +0.152806, case별 중앙값 +0.029057이다. `s1601`과 `s1700`의 차이가 각각 +0.849585와 +0.751864로 평균에 큰 영향을 준다. 이 차이를 모든 case에 고르게 나타난 신장 개선으로 기술하지 않는다.
- 같은 성인 validation 16건에서 M1−M0 평균 Dice 차이는 간 −0.010598, 비장 −0.007295, 왼쪽 신장 −0.009965, 오른쪽 신장 −0.016722, 췌장 −0.026752, 왼쪽 부신 −0.064130, 오른쪽 부신 −0.088945이다. 성인 macro는 16/16건에서 M1이 낮았고, 양쪽 부신이 균등평균 하락의 약 68%를 차지한다.
- 이는 사후 기술 분석이며 원자료는 `M0/validation_050000.json`, `M1/validation_005000.json`, `M2/validation_005000.json`이다. M0 소아 validation과 공식 test·NSD·HD95는 아직 없다. 장기 크기와 개선량의 연관성도 아직 계산하지 않았다.

## 2026-10-30 22:48 KST — 선정 M0의 소아 development validation 추가

- 학습을 마친 M0 `selected.pt`를 변경하지 않고 기존 `validate_model`과 동일 소아 validation 16건의 fast cache로 추론했다. 산출물은 `M0/pediatric_validation_selected.json`이며 checkpoint SHA256은 M0 status 기록과 일치한다. 학습 중의 M0 checkpoint 선택은 성인 validation만 사용했고, 이번 소아 점수는 사후에 측정했다. 공식 test 데이터는 사용하지 않았다.
- 소아 validation macro Dice: M0 **0.705362**, M1 **0.817575**, M2 **0.674918**. 같은 16건의 paired macro 차이는 M1−M0 **+0.112213**(16/16건 양수), M1−M2 **+0.142657**(16/16건 양수), M2−M0 **−0.030444**(3/16건 양수)다. 성인 validation macro Dice는 M0 0.937395, M1 0.905337, M2 0.938348이다.
- 소아 장기별 M1−M0 평균 Dice 차이는 liver +0.027914, spleen +0.030556, kidney_left +0.034084, kidney_right +0.030909, pancreas +0.102514, adrenal_gland_left +0.360526, adrenal_gland_right +0.198984다. 양쪽 부신이 7장기 균등평균 차이의 약 71%를 차지한다. 특히 kidney_left의 M1−M2 차이 +0.152806은 M2−M0가 −0.118722인 영향이 커서 M1의 M0 대비 개선 +0.034084와 구분해야 한다.
- Case 단위 20,000회 복원추출의 탐색적 95% percentile 구간은 소아 M1−M0 macro **[+0.0896,+0.1352]**, 소아 M1−M2 **[+0.1091,+0.1785]**, 성인 M1−M0 **[−0.0391,−0.0253]**이다. 환자 연결 정보 부재, 단일 seed, 작은 validation 표본 때문에 확인적 신뢰구간이나 일반화 보장으로 해석하지 않는다.
- 입력 JSON의 모든 case × 7장기 값으로 장기 평균·macro의 일치를 확인하고 세 branch의 cohort별 case ID가 동일함을 검사했다. `validation_summary.csv`에 M0 소아 행을 추가하고, `validation_case_scores.csv`에 32 case × (7장기 + macro)의 세 모델 점수와 paired 차이를 기록했다. 기존 장기 비교·경과 그림을 갱신하고 `plots/validation_organ_deltas.png/.svg`를 생성했다. 공식 test Dice·NSD·HD95 평가는 여전히 미실행이다.

## 2026-11-03 — 전체 1,939건의 미사용 이유 재감사

- `manifests/splits_abdominal_v1/assignments.csv` 1,939행을 `manifests/qc_v300/case_qc.csv` 및 `manifests/review_v300/cases.csv`와 case ID로 결합했다. 서로 겹치지 않는 최종 행선지는 QC 제외 28, 공식 test 108, 개발 후보의 7장기 중 하나 이상 빈 mask 866, 7장기 모두 nonempty이나 하나 이상 border 접촉 258, 개발 학습·validation 673, 적격 소아 미배정 6건으로 합계 1,939다. 따라서 `excluded` 1,152건은 28+866+258이며 전부 QC 오류가 아니다.
- QC 제외 28건은 연령 결측 15(공식 test 1 포함), 100세 초과 출처 확인 필요 3, 정확한 CT 중복 격리 4, 기술 오류 6이다. 기술 오류 6건 중 5건은 현재 7장기 panel 밖의 mask에서 발견됐다. 현재 고정 규칙은 원본 QC의 case 전체 적격성을 따르므로, panel 외 오류만 있는 영상을 후속 분석에서 재포함하려면 별도 사전 정의와 새 split이 필요하다.
- 자동 QC 통과 공식 train 1,803건은 성인 1,538·소아 265건이다. 7장기 중 하나 이상 빈 mask 866건은 7장기 전부 빈 584건과 일부만 nonempty인 282건으로 나뉜다. 7장기 모두 nonempty 937건 중 258건이 하나 이상 경계에 닿아 제외되고, 679건(성인 625, 소아 54)이 남았다. 경계 접촉만으로 실제 장기 절단이나 annotation 오류를 확정할 수 없으며, 빈 mask의 원인도 case별로 확정되지 않았다.
- 소아 적격 54건 중 32건은 M1, 16건은 validation, 6건(`s1558`, `s1586`, `s1670`, `s1840`, `s1864`, `s2043`)은 무작위 배정 후 미사용으로 남았다. 모두 자동 7장기 nonempty/no-border 기준 통과이며 수동 전문가 완전 검수 완료라는 의미는 아니다. `s1152`는 제외가 아닌 공식 성인 test case다.
- 성인 validation 16건은 성인 적격 625건 중 약 2.6%이며 M0 checkpoint를 20차례 성인 validation으로 비교했다. 소아 validation 16건은 소아 적격 54건 중 약 29.6%다. 두 군 모두 절대 평가 표본 수가 작고 성인 모델 선택 편향 및 test 장기별 소표본의 위험을 명시한다. 완료된 모델 결과를 본 뒤 현 split을 소급 변경하지 않는다.

## 2026-11-03 — V3 공식 train/test 중복 CT 재검증 및 upstream 보고

- 원본 추출 디렉터리의 `meta.csv`에서 `s1445=train`, `s1446=test`를 확인했다. 두 `ct.nii.gz`는 각각 6,074,797 byte로 **파일 SHA256이 동일**(`9a03e6fb5ea903fc9ffd4939d839419c58ab01c9827561950f571bdd3336c30c`)하고, NiBabel로 읽은 `(155,155,209)` voxel 배열과 affine도 동일했다. 공개 `age` 열은 서로 다르다.
- 각 case의 117 mask는 74개가 파일 및 voxel 단위로 같고 43개가 실제 voxel에서 달랐다. 같은 grid에서 `liver`는 7,159 voxel 차이/Dice 0.981561, `spleen`은 7,388 voxel 차이/Dice 0.896381이었다. 이는 동일 CT의 상이한 라벨 버전 또는 메타데이터 사유를 확정하는 증거는 아니다.
- 기존 연구 split은 train 측 `s1445`를 격리하고 test 측 `s1446`를 유지해 이 정확한 CT 중복이 본 학습에 들어가지 않았다. 공개된 모델의 실제 학습 포함 여부나 발표 성능 영향은 확인하지 않았고 주장하지 않는다. 로컬 ZIP을 삭제한 뒤이므로 공식 archive checksum을 이번 점검에서 재검증하지는 못했다.
- 공식 `wasserth/TotalSegmentator` 저장소의 ID 및 관련 키워드 Issue 검색에서 동일 제보를 찾지 못했다. GitHub 앱의 Issue 작성은 upstream 권한 403으로 실패했으나, 사용자가 승인한 `gh` CLI의 활성 `Laplace-tech` 계정으로 [Issue #638](https://github.com/wasserth/TotalSegmentator/issues/638)을 게시했다. 재현 코드·해시·라벨 차이 통계만 올렸고 CT·mask 파일은 첨부하지 않았다.

## 2026-11-06 — 멀티시드 계획과 저장공간 감사

- 사용자는 여러 training seed에서 결과의 일관성을 확인하기로 했다. 제안은 기존 20261029에 사전에 후보로 기록한 20261030·20261031을 추가해 총 3개의 M0/M1/M2 triplet을 비교하는 것이다. 새 2개 seed 실행은 아직 시작하지 않았다. 첫 validation 결과 확인 후 반복을 추가하기로 했다는 순서를 공개한다.
- 모든 seed에서 cohort/split·7장기·정규화·전처리·모델·optimizer·LR·증강·sampling 규칙·학습 예산·checkpoint 선택 기준을 유지한다. Training seed만 바꾸고 각 seed의 M0를 무작위 초기화부터 다시 학습한다. 그 seed에서 선정된 동일 M0로 M1/M2를 각각 시작한다. 같은 split에서 학습 무작위성에 대한 안정성을 보는 설계이며 split 변경이나 외부 데이터 일반화를 검증하는 설계는 아니다.
- 각 seed의 장기별 M1−M0·M1−M2·성인 성능 변화를 모두 보고하고 평균·sample SD·범위·개선 방향의 일치 횟수를 기술한다. 같은 case의 여러 seed 결과를 독립 표본으로 세지 않는다. Case bootstrap을 시행하면 같은 case의 모든 seed/model 결과를 함께 재표집하며, 고정된 seed들에 조건부인 case 불확실성으로 설명한다. Test 공개 전에 반복 수·집계 규칙을 확정한다.
- 현재 `branch_config()`는 `paired_training_seeds[0]`만 사용하고 기본 출력 경로도 20261029로 고정돼 있다. Seed 선택·고유 출력 경로·resume identity·smoke 검증을 연결해야 한다. 완료된 첫 실행의 source snapshot과 결과는 보존한다. 추가 seed마다 본 학습 60,000 successful updates가 필요하며, 첫 실행의 fast-cache 구간과 validation 측정으로 약 7.5–8시간/triplet, 여유를 포함해 8–10시간을 계획한다. 추가 2개는 약 16–20시간이며 test 추론·사전 검사는 별도다.
- 심볼릭 링크를 따라가지 않고 작업 디렉토리 전체를 조사했다. 일반 파일 260,478개, 할당된 파일 블록 합계 약 104.356 GiB(디렉토리 메타데이터 제외)다. Raw 49.179 GiB, 원본 전처리 cache 25.553 GiB, fast cache 20.530 GiB, 환경 6.364 GiB, 연구 run 약 1.714 GiB, 검토 갤러리 약 0.935 GiB다. 확인 당시 Windows C: 여유는 약 102 GiB였다.
- Raw의 CT 1,939개는 33.505 GiB, 선택한 7장기 mask 13,573개는 1.041 GiB, 나머지 mask 213,290개는 14.633 GiB다. 미사용 case나 panel 밖 mask도 배포본·QC·upstream 제보의 원본 근거이므로 원본을 훼손하는 부분 삭제는 제안하지 않는다. 원본 ZIP과 `.tmp/.part/.partial/.bak` 후보 파일은 발견하지 않았다.
- M0/M1/M2 `selected.pt` SHA256이 status와 일치했다. 각 branch의 `latest.pt`와 `selected.pt`를 CPU에서 로드해 모델·optimizer·RNG·config·provenance를 포함한 전체 payload의 일치를 확인했다(파일 SHA256은 serialization 이름 차이 등으로 서로 다름). 완료된 run의 `latest.pt` 3개는 약 751.5 MiB의 중복 저장이며 선정본을 유지하면 삭제 후보가 된다. `pre_fast_cache_upgrade.pt`는 약 250.5 MiB로 I/O 이관 전 상태의 감사 근거라 보존한다.
- Fast cache manifest는 complete/error=null, 673건이다. Fast 학습 경로에서 원본 cache는 `manifest.json`과 `normalization.json`을 참조하며 원본 NPZ 배열을 읽지 않는다. 전체 fast 파일 hash·압축 해제·array schema 검증 이후 원본 NPZ 673개(약 25.55 GiB)는 조건부 삭제 후보다. 원본 cache 폴더 전체를 삭제하면 안 된다. 이번 점검은 20.53 GiB fast cache 전체 재검사를 실행하지 않았으며, 과거 압축 해제 오류 때문에 해당 검사를 생략하지 않는다. 원본 NPZ를 삭제하면 원래 cache를 사용하는 smoke/night/변환 재실행에는 재생성이 필요하다.
- 검토 PNG 1,219개는 약 954.8 MiB로 raw와 manifest에서 재생성 가능하지만, 삭제하면 갤러리 링크를 사용할 수 없다. 수동 판정 ledger와 진행 기록은 보존한다. `.venv`는 다음 학습·평가에 필요하고 smoke 결과 JSON은 본 학습의 시작 검증에 실제로 참조되므로 유지한다. 이번 작업에서는 파일 삭제나 학습 실행 없이 감사와 계획만 기록했다.

## 2026-11-06 — 멀티시드 실행기와 검증 후 cache 정리 구현

- 사용자 승인 후 `training.replication_plan`에 총 3개 seed(20261029·20261030·20261031), 첫 validation 이후 반복 결정, 고정 split, seed별 M0 재학습, 모든 seed 보고·case 단위 불확실성 규칙을 기록했다. 첫 seed의 snapshot·결과·배정·정규화는 유지한다.
- `tsv3.training --seed`가 명시한 seed를 모든 branch 설정·M0 초기화 검사·checkpoint·run provenance에 전달한다. 기본 출력은 seed별로 구분하며 다른 seed의 출력 경로 및 checkpoint 재개를 거부한다. 기존 smoke는 동일 numeric plan의 실행 근거로 유지하고 추가 seed 경로는 별도 CPU 통합 테스트로 검증했다. M0·M1·M2의 학습 budget·loss·sampling·증강·추론 함수 AST는 첫 seed의 `source_snapshot_fast.tar.gz`와 비교해 동일함을 확인했다.
- `tsv3.fast_cache --verify --prune-source`를 추가했다. 전체 fast hash·압축 해제·dtype/shape/finite·7장기·foreground 좌표, 남아 있는 source NPZ hash를 검사한 뒤에만 manifest에 명시된 source NPZ를 삭제한다. Source manifest·정규화·로그·raw는 보존한다. `verification.json`에 진행·삭제·오류를 기록한다. 학습의 shared cache lock과 maintenance의 exclusive lock으로 동시 삭제를 차단한다.
- 34개 테스트가 통과했다. 추가 두 seed의 CPU M0/M1/M2 실행·재개, seed 간 M0 초기화 차이와 seed 내 M1/M2 동일 초기화, 다른 seed checkpoint 거부를 확인했다. 마지막 fast 파일 hash 손상·압축 해제 실패·활성 reader가 있으면 source 배열을 삭제하지 않는 검사, 정상 정리 후 fast loader 입력 동등성·메타데이터 보존·반복 실행 검사도 통과했다.
- 실제 데이터의 두 `--dry-run`이 모두 통과했다. 각 673건, split SHA256 `95946b908a5bc85ab6e6f43888e96a2954f1f4e082e72e15e9bd4a25e8a2df0e`, normalization SHA256 `706ab89111d711a0c59e341086a059a2094df6ef6704364f7d44743efc6145e9`로 동일하다. 실제 fast cache `s0001`·큰 case `s2105`의 hash와 배열 schema 검사도 통과했다. 전체 cache 검사·추가 GPU 본 학습·test 추론은 실행하지 않았고 사용자 터미널 명령을 전달한다.
- 중복 checkpoint 정리 직전 다시 확인했을 때 M0/M1/M2의 `latest.pt` 3개는 이미 없었다. 이 작업에서 삭제한 파일은 0개다. 세 `selected.pt`의 SHA256은 모두 기존 완료 status와 일치했다. 이전 감사 때와 비교하면 중복 latest 약 751.5 MiB가 사라졌지만 삭제 주체는 확인하지 않았다. `pre_fast_cache_upgrade.pt`와 선정 모델·학습 기록은 보존한다.

## 2026-11-06 — 사용자 cache 검사 실행과 seed별 수동 시작

- 사용자가 `tsv3.fast_cache --verify --prune-source`를 실행했다. 점검 당시 PID 8498, 350/673건 검사 중, error=null, 삭제한 source bytes=0이었다. 전체 검사 통과나 최종 삭제 완료를 이 시점에서 주장하지 않는다.
- 사용자는 추가 seed를 하나씩 실행하고 자동 연속 실행하지 않기로 했다. README의 연결 명령을 분리했고, 중단된 seed만 동일 명령으로 재개하도록 안내했다.
- 추가 삭제 후보로 raw에서 재생성할 수 있는 검토 PNG 1,219개(954.77 MiB)의 삭제 명령을 제공한다. PNG 삭제 시 기존 갤러리 이미지는 열리지 않으며 수동 판정 ledger·HTML·진행 기록은 남긴다. 이 점검에서 PNG 삭제를 실행하지 않았다. C: 여유는 약 99 GiB였으며 원본 데이터·fast cache·환경·manifest·선정 checkpoint는 계속 보존한다.

## 2026-11-06 13:49–13:51 KST — seed 20261030 초기 학습 점검

- PID 11211의 M0가 점검 중 1,450→1,625/50,000 updates로 진행했다. Run/branch 상태 running, error 없음, AMP skip 0이다. M1/M2는 아직 시작하지 않았고 첫 성인 validation은 2,500 updates에 예정되어 실제 validation Dice는 아직 없다.
- 기록된 61개 loss는 모두 finite였다. 첫 10개(25–250 updates)의 평균은 0.34283, 마지막 10개(1,300–1,525)는 0.17405이며 최근 20개 평균은 0.15785다. 서로 다른 무작위 patch의 loss이므로 이 변화만으로 일반화 성능이나 수렴을 판정하지 않는다. 최근 500 updates의 학습 시간은 약 0.4212초/update다.
- RTX 3060 Ti의 12회 약 1초 간격 표본에서 GPU utilization 평균 62.8%, 범위 11–85%, VRAM 3,281 MiB/8,192 MiB, 온도 61–65.5°C였다. 짧은 표본이며 지속적으로 GPU를 포화시키는 상태라는 주장은 하지 않는다. 관측한 vmstat 구간의 swap-out은 0, CPU I/O wait은 0%였다.
- WSL available RAM 약 11 GiB, 학습 process RSS 약 2.9 GiB, 전체 swap 사용 약 2.6 GiB(학습 process 약 1.1 GiB), 커널 oom_kill=0이다. Memory pressure avg10 약 0.23%, I/O pressure avg10 약 0.44%를 관측했다. C: 여유 약 101 GiB로 이번 시점의 즉각적인 용량 부족은 보이지 않았다.
- 1,500-update `M0/latest.pt`를 CPU에서 열어 seed·branch·update, optimizer/RNG 존재, 모든 model tensor finite를 확인했다. 현재 source 파일과 study 설정의 hash가 run 시작 provenance와 일치한다. 본 점검은 read-only이며 학습 코드·설정·프로세스를 변경하지 않았다. 이 이력 문서만 갱신했다.

## 2026-11-06 13:52–13:53 KST — GPU 활용도 추가 검토

- M0 약 1,950–2,025 updates 구간에서 30회 약 1초 간격 표본을 수집했다. GPU utilization 평균 48.87%, 범위 4–87%, VRAM 3,257–3,282 MiB/8,192 MiB(약 40%), 온도 59.7–65.6°C, 평균 전력 108.21 W였다. SM clock은 2,010 MHz, memory clock은 7,001 MHz였다. 별도 performance 질의 순간에는 thermal/power slowdown이 활성화되지 않았고 idle reason이 활성화돼 있었다. 이는 짧은 장치 전체 표본이며 FLOPS/Tensor Core 사용률 또는 학습 전체 peak VRAM 측정은 아니다.
- 최근 1,000 updates의 학습 시간은 0.4102초/update다. 코드상 `CaseCache.get`의 전체 volume 읽기·압축 해제·정규화, patch 추출·GPU 전송, update, `torch.cuda.synchronize`가 순차 수행되며 다음 case prefetch는 없다. M0 577건의 CT+target 비압축 합계는 약 87.94 GiB이고 RAM LRU 한도는 2 GiB다. 입력 준비와 GPU 연산의 겹침이 없는 구조가 대기의 유력 원인이지만 구간별 profiler가 없어 각 비용의 비율을 확정하지 않았다. Checkpoint 저장 250 updates 간격도 짧은 utilization 표본에 영향을 줄 수 있다.
- 충분히 최적화된 GPU 활용이라고 판단하지 않는다. 개선 후보는 메모리 한도를 둔 CPU prefetch와 pinned/asynchronous patch 전송이며, 실제 sampling·augmentation·RNG 순서와 재개 일치 및 처리속도를 확인해야 한다. Batch/patch 변경은 고정된 연구 조건을 바꾸므로 현재 멀티시드 반복의 처리속도 개선과 별도로 다뤄야 한다. 이 점검에서는 학습·설정을 변경하거나 성능 개선 폭을 단정하지 않았다.

## 2026-11-06 — 중단 후 입력 prefetch 및 batch/patch 성능 비교 준비

- 사용자가 GPU 최적화와 batch/patch 후보 시험을 허용했고 Ctrl+C 중단 완료를 알렸다. 학습 process가 없고 seed 20261030 run이 interrupted인 상태를 확인했다. M0 status는 2,500-update validation 중 중단됐지만 실제 `M0/latest.pt`는 2,250 updates이며 M0 selected와 M1/M2는 아직 없다. 학습 checkpoint·split·study 설정은 수정하지 않았다.
- 기존 `training.py`에 다음 CT 한 건만 준비하는 CPU worker와 공유 `training_input`을 추가했다. NumPy RNG 상태 복사본으로 다음 case를 예측하고 authoritative RNG·patch sampling·GPU augmentation 순서는 유지한다. Validation 전에 worker를 drain하고 종료 시 join한다. `--prefetch`는 명시적 옵션이며 batch·patch·학습 예산은 기존 설정을 사용한다.
- 기존 `probe.py --throughput`에 중단 checkpoint 기반 비교를 구현했다. 실제 M0 입력 16개의 case·증강 tensor·RNG hash를 비교하고, serial/prefetch/prefetch/serial 네 pass를 같은 checkpoint에서 시작해 각각 warmup 10회+측정 100회 실행한다. 첫 update 가중치 허용오차 검사, 입력 대기/연산 시간, AMP skip, 최대 VRAM을 기록한다. Batch 1/2/4와 batch 2의 128³ patch도 GPU 상주 입력으로 별도 측정한다. 후자는 end-to-end 처리속도가 아니며 연구 조건 변경 후보임을 명시한다. Test 데이터는 사용하지 않는다.
- 기존 source provenance를 우회하지 않고 `--upgrade-prefetch`의 명시적 이관을 구현했다. 현재 source·원본 checkpoint에 대응하는 동등성 검사와 최소 1.05배 속도 근거가 있어야 한다. 원래 run·checkpoint·source snapshot과 benchmark 근거를 보존하며 데이터·환경·수치 설정 변경은 거부한다. 이관 전 일반 resume은 source 불일치로 거부된다. 실제 GPU 비교나 이관은 아직 실행하지 않았다. 따라서 성능 개선을 확인했다는 주장은 하지 않는다.
- 37개 CPU 테스트가 통과했다. Prefetch 입력·증강·RNG/중단 재개 일치, 잘못된 case 및 worker 압축 해제 오류 전파, 근거 없는 이관·변경된 연구 설정 거부, 이관 시 원본 보존을 확인했다. 기존 두 seed M0/M1/M2 CPU 통합 테스트 중 하나는 prefetch 경로로 실행했다. 실제 데이터 기반 throughput `--dry-run`도 통과했고 checkpoint 2,250/seed 20261030을 확인했다. Surface-distance 라이브러리의 기존 SciPy deprecation warning은 있으나 테스트 실패는 없다.
- 사용자 실행 명령은 `tsv3.probe --throughput --run outputs/training_abdominal_v1_seed20261030`이다. 산출물은 단일 `outputs/throughput_abdominal_v1_seed20261030/` 아래 결과 JSON·로그·source snapshot만 보관하며 시험 가중치는 저장하지 않는다. 측정 결과를 보고 prefetch 적용과 batch/patch 변경 여부를 판단한다. Batch/patch를 변경한 run은 기존 seed와 동일 조건 반복으로 합산하지 않는다.

## 2026-11-06 — 성능 비교 완료 및 seed 20261030 새 학습 준비

- 사용자 benchmark가 정상 완료됐다. Serial 평균 0.407547초/update, prefetch 0.292128초/update로 처리량 1.3951배·update 시간 약 28.3% 감소다. 16개 입력·RNG hash 일치, 네 pass case 순서 일치, 첫 update 가중치 허용오차 검사를 통과했고 AMP skip은 0이었다. Validation·checkpoint 저장이 제외된 짧은 구간 측정이므로 전체 학습 시간 개선율을 보장하지 않는다.
- GPU 상주 입력 시험은 batch 1: 0.18646초/5.363 patches/s, batch 2: 0.34125초/5.861 patches/s, batch 4: 9.69204초/0.413 patches/s였다. Batch 4 peak allocated 6,882.6 MiB, reserved 10,716 MiB로 기록됐으며 장치 전체 VRAM 표본은 7,948/8,192 MiB였다. 메모리 압박에 의한 저하 가능성이 있으나 원인을 profiler로 확정하지 않았다. Batch 2·128³은 0.45045초/4.440 patches/s였다. Batch/patch 변경은 채택하지 않고 기존 연구 조건에 prefetch만 적용한다.
- 사용자가 기존 중단 산출물을 지우고 처음부터 재학습하도록 요청했다. Benchmark 종료·exclusive run lock·seed·interrupted 상태·checkpoint SHA 일치를 확인한 후 `outputs/training_abdominal_v1_seed20261030/`만 삭제했다(250.680 MiB). 이 폴더의 checkpoint·run/status/log·source snapshot을 삭제했고 첫 seed 결과·원본/cache·benchmark 근거는 보존했다. 해당 중단 checkpoint의 재개/benchmark 재실행은 이제 불가능하다. 기존 연구 일지에는 이번 폐기·재시작 사실을 남긴다.
- 새 시작 명령은 `tsv3.training --seed 20261030 --prefetch --fast-cache data/preprocessed/abdominal_v1_fast --host-volume /mnt/c`이며 resume/upgrade 옵션 없이 M0 update 0부터 시작한다. 실제 manifest 기반 dry-run 통과(673건, 예산 50,000/5,000/5,000)를 확인했고 본 학습은 사용자가 실행한다. 실행 도중 중단했다면 같은 명령에 `--resume`을 추가한다. 다음 seed는 자동 실행하지 않는다.

## 2026-11-06 15:08–15:12 KST — 재시작 seed 20261030 실제 학습 정밀 점검

- 앞선 짧은 모니터링은 사용자 요청으로 중단했고 학습 PID 22515는 유지했다. 이번 재요청으로 120초 동안 2초 간격 61회 장치·process·WSL 상태를 측정했다. 이 구간 M0 8,725→9,150 updates, 실제 확인 마지막 status 9,375/50,000(18.75%)이며 M1/M2는 아직 시작하지 않았다. 학습 process·코드·설정은 변경하지 않았다.
- 측정 구간 training counter 기준 0.28336초/update, wall clock과 주기적 status 기준 약 0.28235초/update다. Status가 25 updates마다 갱신되므로 양 끝에 정렬 오차가 있다. 이전 실제 0.4102초/update 대비 update 시간 약 30.9% 감소, 처리량 약 1.448배다. 더 긴 구간은 500–2,500: 0.27459, 3,000–5,000: 0.27308, 5,500–7,500: 0.27099초/update였다. Validation·checkpoint 저장 시간은 training counter에 포함되지 않는다. 별도 ABBA benchmark와 실제 실행 양쪽에서 속도 이득이 관측됐다.
- GPU utilization 평균 61.05%(4–93%), VRAM 3,366/8,192 MiB, 온도 평균 65.66°C/최대 68°C, SM clock 1,995–2,010 MHz, 평균 전력 129.37 W였다. 이전 표본은 평균 48.87%와 62.8%로 표본별 변동이 있어 utilization의 확정적 개선율은 제시하지 않는다. GPU 포화 또는 최대 최적화 상태는 아니다. 별도 질의 순간 thermal/power slowdown은 비활성이다.
- Process RSS 2,161.8–2,940.4 MiB, WSL available RAM 11.48–12.30 GiB, process swap 약 1.82–1.86 GiB였다. 120초 동안 시스템 swap-in 162.06 MiB, swap-out 249.30 MiB가 발생해 swap I/O가 없다는 주장은 하지 않는다. 새 OOM kill은 0, 종료 시 memory pressure avg10 0.02%, I/O pressure avg10 0.16%로 즉각적인 메모리 고갈은 보이지 않았다. Host C: 여유 103.43 GiB, Linux 가상 filesystem 여유 869.34 GiB이며 실제 저장 한도는 host도 함께 보아야 한다.
- 기록된 loss 375개 모두 finite, 최초 20개 평균 0.30122→최근 20개 −0.23176이다. 설치된 nnU-Net loss 구현은 CE와 음의 soft Dice를 더하므로 음수 자체는 오류가 아니다. AMP skip 0, run/log error 없음이다. CPU로 읽은 latest 9,250·selected 7,500 checkpoint의 model/optimizer tensor가 모두 finite이고 optimizer state 82개·RNG를 보존한다. LR가 고정 polynomial schedule과 일치하고 234개 state tensor가 selected 이후 달라져 실제 가중치 갱신을 확인했다. Selected update는 best_update와 일치한다.
- 현재 source·study·uv.lock·source snapshot·source/fast manifest·normalization hash가 run provenance와 모두 일치했다. Validation 세 파일은 같은 성인 16건/7장기이며 범위·macro·장기 평균 재계산 검사를 통과했다. Train/test case ID 겹침은 없었다. 전체 20 GiB fast cache 재검사나 추가 GPU 추론은 실행하지 않았다.
- 성인 validation macro Dice는 2,500: 0.17219, 5,000: 0.48145, 7,500: 0.57736이다. 첫 seed의 같은 7,500 시점은 0.56616이었다. 현재 7,500 장기 평균은 liver 0.81463, spleen 0.85800, kidney_left 0.88106, kidney_right 0.84551, pancreas 0.64234, 양 부신 0이다. Liver는 5,000의 0.92994에서 하락했고 16건 중 15건에서 감소했다. 전체 평균 상승만으로 모든 장기 개선을 주장하지 않는다.
- 첫 seed도 부신이 10,000까지 0이고 12,500부터 양수가 됐지만 이번 seed의 향후 회복을 보장하지 않는다. 현 시점은 실행 장애 증거 없이 전체 validation이 개선되는 학습 진행 상태이며, 완전 수렴·과적합 부재·최종 성공을 판정할 근거는 부족하다. 10,000–15,000 validation에서 간 회복과 부신 학습 여부를 계속 확인한다. 근거는 해당 run의 단일 `monitoring.json`에 raw samples·checkpoint 검사·비교 수치를 모았다.

## 2026-11-06 16:32–16:36 KST — seed 20261030 중간 M0 점검

- 학습 PID 22515는 running이고 마지막 live status 약 23,000/50,000 successful updates(46%)다. M1/M2는 아직 시작하지 않았다. 22,500 validation 종료 직후 학습 구간 60초 2초 간격 31회에서 22500→22725 updates로 진행했으며 GPU utilization 평균 62.29%(8–93%), VRAM 평균 3,435 MiB/8,192, GPU 온도 최대 67.4°C, 평균 전력 128.19 W, SM clock 1,995–2,010 MHz였다. 직전 120초 training 구간 평균 GPU 61.05%·VRAM 3,366 MiB와 큰 차이가 없어 GPU 포화나 추가 큰 효율 개선을 주장하지 않는다. Validation 말미 별도 10초 표본 GPU 평균 14.7%, VRAM 약 1,478 MiB였으므로 phase를 섞어 GPU 평균을 해석하지 않는다.
- 현재 학습 구간 status counter 기준 약 0.28349초/update, 상태 갱신 간격으로 인한 양 끝 오차가 있다. 직전 2,000 updates의 누적 training_seconds 차이는 약 0.27422초/update로 이전 synchronous 0.4102 대비 update 시간이 약 33% 짧다. Validation 시간을 포함한 최근 1,000 updates wall clock으로 속도를 산출하면 22,500 validation이 포함되어 부정확하므로 사용하지 않았다.
- M0 22,500 성인 validation macro Dice는 0.916393, 첫 seed 동일 시점 0.907734다. 현재 20,000의 0.9029에서 회복했고 최고 adult score/best_update가 0.916393/22,500으로 갱신됐다. Liver 0.97138, pancreas 0.89097, adrenal left/right 0.86971/0.83841이다. 16건·7장기의 값은 모두 0–1 사이 finite이고 macro 재계산과 일치한다. 초기 liver 하락과 부신 0은 해소됐으나 장기별 차이는 남는다.
- Run·source·study·uv.lock·fast manifest·source snapshot hash가 맞고, CPU로 읽은 latest 22,750/selected 22,500 checkpoint의 model/optimizer tensor가 모두 finite다. Optimizer state 82개·RNG 존재, config/provenance·LR 일치, selected update는 best_update와 일치한다. 약 917개 기록 loss 모두 finite, error 없음, 누적 AMP skip 3회/attempts=updates+3으로 성공 update 예산은 줄지 않았다.
- Process RSS 학습 구간 평균 약 3,089 MiB, process swap 평균 1,818 MiB, available RAM 평균 11.10 GiB다. 60초간 swap-in/out 약 4.57/31.91 MiB, 새 OOM 0, memory pressure avg10 0.07%, I/O pressure avg10 0.18%, host C: 여유 약 102.54 GiB였다. Swap은 발생하지만 이번 구간의 뚜렷한 진행 장애 근거는 없다.
- 16:36 KST 23,000 updates를 기준으로 최근 2,000 successful updates training counter 0.27422초/update와 완료된 M0 validation 평균 217.77초/회, 이전 seed M1/M2 실측 합계 약 1.57시간을 적용하면 M0 종료 약 19:20, 현재 triplet 종료 약 20:54 KST다. 이는 추정이며 validation·저장·장치 상태에 따라 변한다. 학습 및 연구 설정은 바꾸지 않았고 원시 GPU 표본은 해당 run의 `monitoring.json` 한 파일에 보관했다.

## 2026-11-06 18:45 KST — seed 20261030 M0 후반 이상 점검

- M0 status running 45,000/50,000 successful updates, M1/M2 미시작, run error 없음. 점검 시 45,000 update validation 중이라 마지막 완료 validation은 42,500이다. 기록 loss 1,800개 모두 finite, AMP skip 11회/attempts 45,011회로 성공 update 예산 유지. Latest/selected checkpoint 각 약 250 MiB 존재; source·study·fast manifest SHA와 seed/prefetch provenance 일치.
- 성인 validation macro Dice 32,500 최고 0.93245(선정 checkpoint), 35,000 0.89616, 37,500 0.92239, 40,000 0.92943, 42,500 0.92868. 직전 최고를 다시 넘지는 않았지만 최근 두 값은 0.929 안팎으로 유지. 42,500 같은 시점 첫 seed 0.93074. 현재 간 0.9709, 췌장 0.8774, 양 부신 0.9038/0.8375. 16개 case의 점수 finite. Validation 소폭 등락만으로 과적합을 확정할 수 없다.
- 점검 순간 GPU utilization 83%, VRAM 2,177/8,192 MiB, 온도 68.2°C, 전력 166 W로 validation phase 한 시점이다. Process RSS 약 4.63 GiB, host C: 여유 103 GiB, memory pressure avg10 0%, I/O pressure avg10 0.05%. GPU 학습 구간 평균을 이 순간 표본으로 대체하지 않는다. 현 시점에 즉각 중단을 요구하는 오류나 저장공간 부족은 관측되지 않았다. 학습 process·설정·checkpoint에 변경 없음.

## 2026-11-06 19:55 KST — seed 20261030 M0 학습 완료 결과

- M0 50,000/50,000 successful updates, attempts 50,013, AMP skip 13, error 없음. 학습은 성인 577건만 사용했고 M1/M2는 같은 seed의 M0 selected에서 분기한다. 점검 시 run은 M1 실행 중이므로 전체 triplet은 아직 완료되지 않았다.
- Adult validation 16건/7장기 macro Dice 최고 **0.941428 at update 50,000**. Protocol에 따라 선택한 `M0/selected.pt`는 마지막 update이고 `M0/latest.pt`와 모델·optimizer state가 같았다. Selected SHA256 `7fc131f11381b4bf3d41298e851b14c25aeaef9ae05269ff5fa9c1ffaf2d339d`는 status와 일치했다. 모델·optimizer tensor 모두 finite, RNG/설정/provenance/학습률 일치. Source·study·fast manifest·source snapshot SHA도 run 기록과 일치했다.
- 아래 곡선은 동일 성인 validation 16건을 2,500 updates마다 측정한 값이다. 모든 20파일의 16×7 Dice는 finite·0–1 범위이며 macro 재계산과 일치했다. 첫 seed도 같은 case·장기로 평가했다.

| Update | 이번 seed 20261030 | 첫 seed 20261029 |
|---:|---:|---:|
| 2,500 | 0.172191 | 0.141299 |
| 5,000 | 0.481449 | 0.379180 |
| 7,500 | 0.577363 | 0.566156 |
| 10,000 | 0.806956 | 0.632436 |
| 12,500 | 0.882532 | 0.789684 |
| 15,000 | 0.873658 | 0.879608 |
| 17,500 | 0.908402 | 0.914006 |
| 20,000 | 0.902856 | 0.918097 |
| 22,500 | 0.916393 | 0.907734 |
| 25,000 | 0.908173 | 0.908875 |
| 27,500 | 0.927789 | 0.927868 |
| 30,000 | 0.932349 | 0.900931 |
| 32,500 | 0.932448 | 0.772584 |
| 35,000 | 0.896161 | 0.907942 |
| 37,500 | 0.922386 | 0.922540 |
| 40,000 | 0.929427 | 0.921007 |
| 42,500 | 0.928679 | 0.930741 |
| 45,000 | 0.937715 | 0.928128 |
| 47,500 | 0.939354 | 0.934153 |
| 50,000 | 0.941428 | 0.937395 |

- 최종 장기별 성인 validation Dice와 두 seed 간 기술적 차이:

| 장기 | 첫 seed | 이번 seed | 이번−첫 |
|---|---:|---:|---:|
| liver | 0.980381 | 0.979504 | -0.000877 |
| spleen | 0.974023 | 0.975377 | +0.001354 |
| kidney_left | 0.968228 | 0.972295 | +0.004067 |
| kidney_right | 0.971722 | 0.973533 | +0.001811 |
| pancreas | 0.889082 | 0.901840 | +0.012758 |
| adrenal_gland_left | 0.913897 | 0.917806 | +0.003909 |
| adrenal_gland_right | 0.864435 | 0.869641 | +0.005206 |

- Macro 차이는 +0.004033; 동일 case별 macro는 11/16건에서 이번 seed가 높고 5/16건에서 낮았다. 이번 seed case macro 범위 0.888248–0.973598이다. 같은 validation을 checkpoint 선택에 20회 사용했으므로 최고점 차이만으로 seed 일반화나 통계적 우월성을 주장하지 않는다.
- 시간: 이번 seed training 3.749h, adult validation 1.195h, 합계 4.944h. 첫 seed 합계 6.151h; 이번 합계가 약 19.6% 짧다. 첫 seed에는 중단·cache 이관 전 구간이 있어 완전 동일 환경의 시간 비교는 아니며 별도 ABBA 성능 시험으로 prefetch의 단기 이득을 확인했다.
- M0 학습 로그의 loss 2,000개는 모두 finite, 첫 20개 평균 0.301219→마지막 20개 −0.526806이다. Loss는 CE+음의 soft Dice로 구성되어 음수 자체는 오류가 아니다. 마지막 validation에서 최고점이 나왔지만 검증 표본 16건, 모델 선택에 같은 검증 집합 반복 사용, 환자 연결 정보 부재로 일반화·과적합 부재를 확정하지 않는다.
- 이 M0의 **소아 validation은 아직 별도 평가되지 않았고 공식 test도 미사용**이다. 소아 성능과 적응 이득은 M1/M2 완료 후 같은 seed의 세 모델을 동일 소아·성인 validation case에서 비교해야 한다.

## 2026-11-06 19:57 KST — 첫 두 seed의 연구 설정·실행 환경 비교

- 저장된 M0 `selected.pt`의 config 전체 키와 값을 직접 비교한 결과 차이는 `seed` 20261029 대 20261030 하나다. Patch 96×128×128, batch 1, 50,000 successful updates, SGD Nesterov/LR 0.01/momentum 0.99/weight decay 3e-5/polynomial schedule, gradient clip 12, intensity augmentation, foreground sampling, model 구조, validation 2,500 update 간격, checkpoint 250 update 간격이 같다. 저장된 M1 config도 seed만 다르며 M2는 첫 seed 실제 checkpoint와 둘째 seed 고정 계획의 config 차이가 seed뿐이다(점검 시 둘째 M2 미시작).
- GPU 이름 RTX 3060 Ti, Torch 2.8.0+cu126, CUDA 12.6, uv.lock SHA 및 run environment snapshot은 같다. Split·정규화·source manifest·fast cache manifest·evaluation identity SHA도 같다. `study.json` snapshot의 유일한 내용 차이는 첫 실행 이후 추가한 `training.replication_plan` 선언이다. Study SHA가 다른 이유는 이 추가 계획이며 M0/M1/M2 numeric config에는 차이가 없다.
- 실행 source SHA는 동일하지 않다. `training.py`에는 seed 선택/출력 분리와 CPU case prefetch가 추가되고, `probe.py`는 비교 시험 기능, `fast_cache.py`는 검증·source prune 관리 기능이 추가됐다. 첫 seed의 fast-cache source archive와 현재 source의 함수 AST를 비교하면 sample_patch, architecture, build_model, optimization_components, update_step, infer_volume, validation_dice, augment_patch, learning_rate_at은 동일하다. Fast cache의 read_case 등 학습 데이터 읽기 함수 AST도 동일하고 달라진 함수는 main, validate_arrays, verify_and_prune이다.
- 첫 seed M0는 원래 압축 cache에서 1,250-update checkpoint까지 실행 후 lossless fast cache로 검증·이관했고, 둘째 seed는 처음부터 같은 fast cache와 한 건 CPU prefetch를 사용했다. Prefetch 시험은 실제 16입력 case/patch/augmentation/RNG hash와 첫 update 가중치 일치를 확인했지만 전체 장기 학습 경로를 seed 외 완전히 bitwise 동일하게 만들었다는 의미는 아니다.
- 기술 문구: "동일한 데이터 분할, 전처리, 모델, 손실함수, 하이퍼파라미터 및 장비·라이브러리 버전에서 학습 seed를 바꿔 반복했다. 두 실행 사이에 검증된 무손실 cache와 입력 prefetch 등 데이터 공급 구현이 변경되었으며 이를 실행 이력에 기록했다." 따라서 "코드와 실행 방식까지 완전히 동일하고 seed만 다르다"는 표현은 사용하지 않는다. 점수 비교는 고정 수치 protocol의 반복으로 기술하되 시간 차이를 seed만의 효과나 순수 prefetch 효과로 귀속하지 않는다.

## 2026-11-06 20:00 KST — nnU-Net 버전과 실제 U-Net 구성 확인

- 로컬 `.venv`와 `pyproject.toml`/`uv.lock`은 `nnunetv2==2.8.1`, `dynamic-network-architectures==0.4.4`, PyTorch 2.8.0+cu126이다. 2026-11-06 확인한 PyPI의 최신 공개 패키지도 nnunetv2 2.8.1(2026-07-01), dynamic-network-architectures 0.4.4(2026-06-07)이다. 근거: https://pypi.org/project/nnunetv2/ , https://pypi.org/project/dynamic-network-architectures/ .
- 실제 M0/M1/M2는 `nnunetv2`의 모델 생성·loss·추론 구성 요소와 `dynamic_network_architectures.architectures.unet.PlainConvUNet`을 사용한다. 명시적으로 고정한 5 stage(32/64/128/256/320 channels)·patch 96×128×128·batch 1의 3D convolutional U-Net이며 자동 planning된 표준 nnU-Net 전체 workflow가 아니다. 공식 nnU-Net 문서는 ResEnc L을 새 기본 권장 구성으로 소개한다: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md . 따라서 패키지 버전은 최신 공개판이지만 현재 아키텍처가 가장 최신 권장 preset 또는 최고 성능 구성이라는 뜻은 아니다.
- 연구용 M0는 V3 데이터셋의 성인 코호트에서 무작위 초기화부터 학습한 자체 7장기 모델이다. 공식 TotalSegmentator V3 pretrained weight를 미세조정한 실험이나 공식 V3 architecture/전체 학습 pipeline의 재현이라고 기술하지 않는다. 실행 중 아키텍처를 교체하면 seed 반복 비교가 무너져 변경하지 않는다.

## 2026-11-06 20:52 KST — 두 seed의 development validation 적응 패턴 검증

- Seed 20261030의 M0 selected를 고정 소아 validation 16건에 사후 추론해 `M0/pediatric_validation_selected.json`에 보존했다. Checkpoint SHA는 M0 status와 일치하고 M0 선택에는 소아 점수를 사용하지 않았다. 공식 test 입력/예측/점수는 사용하지 않았다. 두 seed 모두 M0/M1/M2 완주(50,000/5,000/5,000 successful updates)와 selected SHA를 확인했고, 각 seed의 M1/M2는 동일 selected M0 checkpoint와 동일 모델 state hash에서 시작하며 branch 외 numeric config가 같다.
- 모든 모델과 seed의 성인 16건/소아 16건 case ID가 같고, 7장기×case Dice가 finite/0–1이며 저장 macro·장기 평균과 재계산 결과가 일치했다. Seed별 점수:

| Seed | 소아 M0 | 소아 M1 | 소아 M2 | 성인 M0 | 성인 M1 | 성인 M2 |
|---|---:|---:|---:|---:|---:|---:|
| 20261029 | 0.705362 | 0.817575 | 0.674918 | 0.937395 | 0.905337 | 0.938348 |
| 20261030 | 0.709255 | 0.819491 | 0.700943 | 0.941428 | 0.910406 | 0.942540 |

- 고정 16건의 paired macro 차이:

| 비교 | 첫 seed | 둘째 seed | 두 seed 평균 | sample SD (n=2) |
|---|---:|---:|---:|---:|
| pediatric_M1_minus_M2 | +0.142657 | +0.118548 | +0.130603 | 0.017048 |
| pediatric_M1_minus_M0 | +0.112213 | +0.110235 | +0.111224 | 0.001398 |
| adult_M1_minus_M0 | -0.032058 | -0.031022 | -0.031540 | 0.000733 |
| adult_M2_minus_M0 | +0.000953 | +0.001111 | +0.001032 | 0.000112 |

- 소아 장기별 비교:

| 장기 | M1−M2 첫/둘째 | M1−M0 첫/둘째 |
|---|---:|---:|
| liver | +0.031219 / +0.031061 | +0.027914 / +0.028609 |
| spleen | +0.038672 / +0.031655 | +0.030556 / +0.036285 |
| kidney_left | +0.152806 / +0.027679 | +0.034084 / +0.018381 |
| kidney_right | +0.068814 / +0.057150 | +0.030909 / +0.038574 |
| pancreas | +0.118093 / +0.101330 | +0.102514 / +0.076967 |
| adrenal_gland_left | +0.403915 / +0.348826 | +0.360526 / +0.349697 |
| adrenal_gland_right | +0.185081 / +0.232135 | +0.198984 / +0.223135 |

- 두 seed 모두 소아 M1−M2와 M1−M0는 7장기 모두 양수이며, 각각 소아 case 16/16건의 7장기 macro가 양수다. M1 성인 M1−M0는 두 seed 모두 7장기 음수이고 성인 case 16/16건에서 감소했다. 성인 M2−M0는 약 +0.001로 작다. 소아 M1−M0에서 양쪽 부신이 7장기 균등평균 개선의 약 71.2%/74.2%를 차지한다.
- **크기 재현은 균일하지 않다.** 왼쪽 신장의 M1−M2는 +0.152806→+0.027679로 축소됐다. 첫 seed의 M2−M0 왼쪽 신장 −0.118722가 큰 영향을 주었으며 둘째 seed에서는 −0.009298이다. 따라서 첫 seed의 큰 M1−M2 신장 이득을 소아 적응만의 안정적 효과로 보고하지 않는다. M1−M0 왼쪽 신장 +0.034084→+0.018381도 두 seed에서 양수지만 크기가 다르다.
- 같은 16 case를 단위로 두 seed·모델 값을 함께 재표집한 10,000회 탐색적 paired bootstrap 95% percentile 범위는 소아 M1−M2 [+0.1030,+0.1602], 소아 M1−M0 [+0.0899,+0.1330], 성인 M1−M0 [−0.0384,−0.0252]다. Seed를 32명의 독립 환자로 세지 않았고, 두 seed에 조건부인 소규모 development validation 범위다. 반복 사용된 validation과 환자 ID 부재로 확인적 CI 또는 외부 일반화 보장으로 해석하지 않는다.
- 결과와 per-case paired 값, checkpoint SHA 및 무결성 검사는 `outputs/training_abdominal_v1_seed20261030/replication_validation.json`에 한 파일로 보존한다. 사전 고정된 세 번째 seed 20261031 완료 후 세 seed 전체의 평균·SD·방향 일치 횟수를 다시 계산하고, 그 전에는 공식 test를 열거나 seed를 선택하지 않는다.

## 2026-11-06 21:06 KST — 연구 저장소 정리와 공개 범위 확인

- 현재 코드·설정·lockfile을 환경, cohort/QC, 전처리/cache, 평가 계약, 학습/검증 단계별 Git 커밋으로 정리했다. 원본 CT·mask, 생성된 case별 manifest, checkpoint, 실행 출력은 `.gitignore`로 제외했다. 앞선 실행의 원본 시점은 run별 `source_snapshot.tar.gz`에 보존되므로 새 Git 커밋을 과거 학습 시점의 bitwise 동일 source로 오인하지 않는다.
- README에 연구 질문, M0/M1/M2 비교, 두 seed의 development validation macro·장기별 차이, 성인 성능 변화, 실행 경로 차이와 남은 test 평가를 명시했다. 원격에서 열리지 않는 로컬 출력 링크는 경로 표기로 바꿨다. 세 번째 seed 20261031과 공식 test는 아직 평가 결과가 없다.
- 현재 source/config를 대상으로 `python -m unittest discover -s tests -q`: 37개 통과. `git diff --check`와 단계별 staged diff 검사를 수행했다. 이 검사는 완료된 학습의 성능을 재검증하는 절차가 아니라 공개할 코드·문서의 기본 점검이다.
