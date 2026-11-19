# 배선도만으로 초파리의 눈을 학습시키다 — 논문 재현과 머신러닝 디코딩 실험

머신러닝 수업 논문 발표 자료.
Lappalainen et al., "Connectome-constrained networks predict neural activity across the fly visual system", *Nature* 634 (2024) · 공식 코드 [flyvis](https://github.com/TuragaLab/flyvis)

- **발표 사이트**: `site/` (정적 웹사이트, 슬라이드 25장, ← → 키로 이동, T 키로 밝게/어둡게)
- **결과 요약**: [results/summary.md](results/summary.md)
- **발표 대본**: [docs/발표준비.md](docs/발표준비.md), 실험 설명: [docs/내실험.md](docs/내실험.md)

## 핵심 결과

| | |
|---|---|
| 재현 | 검증 오차 1위 사전학습 모델이 T4a–d, T5a–d 8개 하위 유형의 선호 방향을 모두 맞힘. 과제 오차 순위 1–10위는 평균 5.1개, 41–50위는 2.8개 |
| 디코딩 (잡음 1배) | 세포 유형 65채널 99.9%, 가상 EEG 4채널 84.1%, 1채널 40.3% (8방향, 우연 12.5%) |
| 특징 선택 | 세포 유형 3개로 98.1%, 세 fold 모두 T4a·T5d 선택 |
| 데이터 누수 | 정답을 섞은 데이터에서 올바른 방법 ≤ 8.7%, 테스트 정답을 보고 고르면 22.4% |

## 실행 (WSL2 Ubuntu, CPU만)

```bash
# uv로 만든 Python 3.12 가상환경 안에서
uv pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
flyvis download-pretrained --skip_large_files      # .env의 FLYVIS_ROOT_DIR(data/flyvis)로 받는다

bash scripts/run.sh scripts/00_smoke_test.py
bash scripts/run.sh scripts/01_reproduce_t4t5.py record   # 50개 모델, 약 50분, data/에 캐시
bash scripts/run.sh scripts/01_reproduce_t4t5.py plot
bash scripts/run.sh scripts/02_make_stimuli_and_record.py # 약 16분, data/에 캐시
bash scripts/run.sh scripts/03_build_signals.py
bash scripts/run.sh scripts/04_decode.py
bash scripts/run.sh scripts/05_ml_experiments.py 1.0      # 잡음 2배는 인자 2.0
bash scripts/run.sh scripts/06_build_site.py              # results/ -> site/
```

## 배포 (Render Static Site)

Publish directory를 `site`로, Build command는 비워 둔다. `site/index.html`은 인터넷 없이 파일을 더블클릭해도 열린다(데이터를 `data/data.js`로 넣었기 때문).
