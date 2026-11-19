"""여러 스크립트가 함께 쓰는 경로, 상수, 그림 설정."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
WEB = RESULTS / "web"
MEDIA = RESULTS / "media"
for _d in (DATA, RESULTS, WEB, MEDIA):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 0  # 모든 난수 시드 고정 (재현성)

# ---- 3~5단계 실험 조건 ----
DIRECTIONS = list(range(0, 360, 45))  # 8방향, 45도 간격 (분류 정답)
INTENSITIES = [1, 0]                  # 1 = 밝은 막대, 0 = 어두운 막대 (배경은 회색 0.5)
SPEEDS = [9.7, 13, 19]                # 반 기둥/초. leave-one-speed-out 교차검증의 fold 기준
NOISE_LEVELS = [0.1, 0.5, 1.0, 2.0]   # 잡음 표준편차 = 채널 신호 표준편차 x 이 배수
N_REPEATS = 20                        # 조건마다 "반복 측정" 횟수
FS = 100                              # 저장·분석 샘플링 주파수 (Hz). 시뮬레이션은 200 Hz

SIGNAL_NAMES = {
    "A": "신호 A: 세포 유형 65채널",
    "B4": "신호 B: 가상 EEG 4채널",
    "B1": "신호 B: 가상 EEG 1채널",
}
SIGNAL_COLORS = {"A": "#1f77b4", "B4": "#ff7f0e", "B1": "#7f7f7f"}


def setup_matplotlib():
    """한글 폰트 설정. WSL에는 한글 폰트가 없어서 Windows의 맑은 고딕을 직접 등록한다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in ["/mnt/c/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgunbd.ttf"]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    # flyvis를 불러오면 matplotlib 글자 크기 설정이 작게 바뀌므로 크기를 직접 지정한다
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "figure.titlesize": 14,
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def best_model_name():
    """검증 오차(과제 오차)가 가장 낮은 사전학습 모델 이름."""
    from flyvis.network import Ensemble

    ens = Ensemble("flow/0000")
    losses = ens.min_validation_losses(subdir="validation", file="loss")
    return ens.names[int(np.argmin(losses))], float(np.min(losses))
