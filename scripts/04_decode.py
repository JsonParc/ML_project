"""
4단계: 움직임 방향 8개 분류 — 신호 A vs 가상 EEG(B4, B1), 잡음 단계별

머신러닝 절차 (수업의 4단계를 그대로)
1) 데이터: 03에서 만든 특징 (잡음 단계마다 960샘플, 8방향 균형)
2) 특징 정규화: z-score. 평균·표준편차는 매 fold의 학습 데이터로만 fit (전체로 하면 데이터 누수)
3) 분류기: 로지스틱 회귀 + L2 정규화 (8방향이므로 소프트맥스 = 단층 퍼셉트론과 같다)
   왜? 단순한 선형 분류기로도 맞히면 정보가 분류기가 아니라 신호 자체에 있다는 뜻이다. L2는 과적합 방지.
4) 검증: 속도 기준 leave-one-speed-out (두 속도로 학습 -> 남은 한 속도로 테스트, 3회)
   왜? 같은 자극의 반복 측정 20개가 학습과 테스트에 나뉘어 들어가면 "외운 것"을 맞히는 셈이라
   정확도가 부풀려진다(누수). 처음 보는 속도로 테스트해야 진짜 일반화다.

산출물: results/fig_noise_vs_accuracy.png, results/fig_confusion.png,
        results/accuracy.csv, results/web/decoding.json
실행: python scripts/04_decode.py   (수십 초)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA, DIRECTIONS, NOISE_LEVELS, RESULTS, SEED, SIGNAL_COLORS, SIGNAL_NAMES,
                    SPEEDS, WEB, setup_matplotlib)

CHANCE = 1 / len(DIRECTIONS)


def make_model(C=1.0):
    # Pipeline으로 묶으면 fit할 때 정규화도 학습 데이터로만 맞춰진다 -> 누수 방지가 코드 구조로 보장됨
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, random_state=SEED))


def loso_eval(X, y, groups, model_fn=make_model):
    """leave-one-speed-out 교차검증. fold별 정확도와 합친 혼동 행렬을 돌려준다."""
    accs, cm = [], np.zeros((len(DIRECTIONS), len(DIRECTIONS)), int)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        model = model_fn().fit(X[tr], y[tr])
        pred = model.predict(X[te])
        accs.append(float((pred == y[te]).mean()))
        cm += confusion_matrix(y[te], pred, labels=range(len(DIRECTIONS)))
    return np.array(accs), cm


def main():
    t0 = time.time()
    F = dict(np.load(DATA / "features.npz", allow_pickle=False))
    y, speed = F["y"], F["speed"]
    rows, cms = [], {}
    for k in SIGNAL_NAMES:
        for lvl in NOISE_LEVELS:
            X = F[f"time_{k}_{lvl}"]
            accs, cm = loso_eval(X, y, speed)
            cms[(k, lvl)] = cm
            rows.append(dict(signal=k, signal_name=SIGNAL_NAMES[k], noise=lvl,
                             n_features=X.shape[1], acc_mean=accs.mean(), acc_std=accs.std(),
                             **{f"acc_test_speed_{s}": a for s, a in zip(np.unique(speed), accs)}))
            print(f"{SIGNAL_NAMES[k]:22s} 잡음 {lvl:>3}배: 정확도 {accs.mean():.3f} ± {accs.std():.3f}"
                  f"  (fold별 {np.round(accs, 3).tolist()})")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "accuracy.csv", index=False, float_format="%.4f")

    plt = setup_matplotlib()
    # ---- 메인 그림: 잡음 크기 vs 정확도 ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k in SIGNAL_NAMES:
        d = df[df.signal == k]
        ax.errorbar(d.noise, d.acc_mean * 100, yerr=d.acc_std * 100, marker="o", ms=7, lw=2.2,
                    capsize=4, color=SIGNAL_COLORS[k], label=SIGNAL_NAMES[k])
    ax.axhline(CHANCE * 100, color="k", ls="--", lw=1.2, label="우연 수준 (12.5%)")
    ax.set_xscale("log")
    ax.set_xticks(NOISE_LEVELS)
    ax.set_xticklabels([f"{v}배" for v in NOISE_LEVELS])
    ax.set_ylim(0, 105)
    ax.set_xlabel("측정 잡음 크기 (채널 신호 표준편차의 배수)")
    ax.set_ylabel("8방향 분류 정확도 (%)")
    ax.set_title("신호를 섞으면 움직임 방향을 얼마나 덜 읽어낼까?\n"
                 "(로지스틱 회귀, 속도 기준 교차검증 3 fold, 오차 막대 = fold 표준편차)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(RESULTS / "fig_noise_vs_accuracy.png")
    plt.close(fig)

    # ---- 백업: 잡음 1배 혼동 행렬 (A vs B4) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    labels = [f"{d}°" for d in DIRECTIONS]
    for ax, k in zip(axes, ["A", "B4"]):
        cm = cms[(k, 1.0)]
        cmn = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(cmn * 100, cmap="Blues", vmin=0, vmax=100)
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(8), labels)
        ax.set_yticks(range(8), labels)
        ax.set_xlabel("예측한 방향")
        ax.set_ylabel("실제 방향")
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{SIGNAL_NAMES[k]} (정확도 {acc * 100:.1f}%)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="행 기준 비율 (%)")
    fig.suptitle("잡음 1배에서의 혼동 행렬 (3 fold 합계)", fontsize=13)
    fig.savefig(RESULTS / "fig_confusion.png")
    plt.close(fig)

    # ---- 웹용 JSON ----
    web = {"directions": DIRECTIONS, "noise_levels": NOISE_LEVELS, "chance": CHANCE,
           "signals": {k: SIGNAL_NAMES[k] for k in SIGNAL_NAMES}, "results": {}}
    for k in SIGNAL_NAMES:
        web["results"][k] = []
        for lvl in NOISE_LEVELS:
            r = df[(df.signal == k) & (df.noise == lvl)].iloc[0]
            web["results"][k].append(dict(noise=lvl, acc_mean=round(r.acc_mean, 4),
                                          acc_std=round(r.acc_std, 4),
                                          n_features=int(r.n_features),
                                          confusion=cms[(k, lvl)].tolist()))
    (WEB / "decoding.json").write_text(json.dumps(web, ensure_ascii=False))
    print(f"완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
