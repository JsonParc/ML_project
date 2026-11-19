"""
5단계: 머신러닝 기법 비교 (발표의 머신러닝 핵심)

4단계와 같은 데이터·같은 속도 기준 교차검증(바깥 fold 3개)을 쓴다. 기본 잡음 단계는 아래 NOISE.

실험
 1. 특징 선택 (순차 전진 선택, 래퍼 방식): 세포 유형 채널을 하나씩 추가하며 정확도 곡선.
    - 올바른 방법: 바깥 fold의 "학습 데이터 안에서만" 안쪽 교차검증(남은 두 속도로 leave-one-speed-out)으로 고른다.
    - 누수 있는 방법(대조군): 테스트 fold 정확도를 보고 고른다 = 시험 문제를 보고 공부한 셈.
    - fold별로 뽑힌 세포 유형을 기록해 어떤 세포가 방향 정보를 담는지 본다.
 2. 채널 수를 맞춘 비교: 신호 A에서 세포 유형 4개를 무작위로 200번 뽑아 B 4채널과 비교.
    "채널 수" 때문인지 "채널 종류(세포별로 골라 재기)" 때문인지 구분한다.
 3. 특징 추출 비교: 시간 영역 vs PSD(주파수) vs 둘 다.
 4. 분류기 비교: 로지스틱 회귀 vs K-NN vs 선형 SVM. 하이퍼파라미터(C, k)는 안쪽 교차검증으로만 고른다(nested CV).
 5. 검증 방법에 따른 누수: 무작위 5-fold(같은 자극의 반복 측정이 학습·테스트에 섞임) vs
    속도 기준 fold, 그리고 정규화를 전체 데이터로 먼저 한 경우.

산출물: results/fig_ml_*.png, results/web/ml.json
실행: python scripts/05_ml_experiments.py   (몇 분)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA, DIRECTIONS, NOISE_LEVELS, RESULTS, SEED, SIGNAL_COLORS, SIGNAL_NAMES,
                    WEB, setup_matplotlib)

NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
K_MAX = 8          # 전진 선택으로 최대 몇 개 세포 유형까지 고를지
N_RANDOM = 200     # 무작위 4개 유형 뽑기 반복 횟수
CHANCE = 1 / len(DIRECTIONS)
rng = np.random.default_rng(SEED)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))


def cols_for_types(type_ids, n_types):
    """시간 특징 배열은 [최고(유형0..64), 최저(..), 최고점시간(..), 평균(..)] 순서 -> 유형별 4열 선택"""
    return np.concatenate([np.asarray(type_ids) + j * n_types for j in range(4)])


def acc_fit(model, Xtr, ytr, Xte, yte):
    return float((clone(model).fit(Xtr, ytr).predict(Xte) == yte).mean())


def loso(X, y, g, model=None):
    model = model or lr()
    return np.array([acc_fit(model, X[g != s], y[g != s], X[g == s], y[g == s]) for s in np.unique(g)])


# ------------------------------------------------------------ 1. 특징 선택
def forward_selection(score_fn, n_types, k_max):
    """탐욕적 전진 선택: 매 단계 점수를 가장 많이 올리는 세포 유형 하나를 추가."""
    from joblib import Parallel, delayed

    chosen, path = [], []
    for _ in range(k_max):
        cand = [t for t in range(n_types) if t not in chosen]
        # 후보 세포 유형 평가는 서로 독립이라 CPU 코어에 나눠 동시에 계산 (결과는 순차 계산과 같다)
        scores = Parallel(n_jobs=-1)(delayed(score_fn)(chosen + [t]) for t in cand)
        best = cand[int(np.argmax(scores))]
        chosen.append(best)
        path.append(float(np.max(scores)))
    return chosen, path


def exp_feature_selection(X, y, g, types):
    n = len(types)
    speeds = np.unique(g)
    honest = np.zeros((len(speeds), K_MAX))
    leaky = np.zeros((len(speeds), K_MAX))
    chosen_per_fold = []
    for fi, s in enumerate(speeds):
        tr, te = g != s, g == s
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]

        # 올바른 방법: 학습 데이터 안의 안쪽 교차검증 점수로만 고른다
        def inner(ids):
            return loso(Xtr[:, cols_for_types(ids, n)], ytr, gtr).mean()

        chosen, _ = forward_selection(inner, n, K_MAX)
        chosen_per_fold.append([str(types[t]) for t in chosen])
        for k in range(K_MAX):
            c = cols_for_types(chosen[: k + 1], n)
            honest[fi, k] = acc_fit(lr(), Xtr[:, c], ytr, X[te][:, c], y[te])

        # 누수 있는 방법: 테스트 fold의 정확도를 직접 보고 고른다 (하면 안 되는 방법)
        def peek(ids):
            c = cols_for_types(ids, n)
            return acc_fit(lr(), Xtr[:, c], ytr, X[te][:, c], y[te])

        _, path = forward_selection(peek, n, K_MAX)
        leaky[fi] = path
        print(f"  fold(테스트 속도 {s}): 선택 순서 {chosen_per_fold[-1]}")
    from collections import Counter
    counts = Counter(t for f in chosen_per_fold for t in f[:4])
    return dict(k=list(range(1, K_MAX + 1)),
                honest_mean=honest.mean(0).tolist(), honest_std=honest.std(0).tolist(),
                leaky_mean=leaky.mean(0).tolist(), leaky_std=leaky.std(0).tolist(),
                chosen_per_fold=chosen_per_fold,
                top4_counts=dict(counts.most_common()))


# ------------------------------------------------------------ 2. 무작위 4개 유형
def exp_random4(X, y, g, types, b4_acc):
    from joblib import Parallel, delayed

    n = len(types)
    draws = [rng.choice(n, 4, replace=False) for _ in range(N_RANDOM)]  # 뽑기는 고정 시드로 먼저
    accs = np.array(Parallel(n_jobs=-1)(
        delayed(lambda ids: loso(X[:, cols_for_types(ids, n)], y, g).mean())(ids) for ids in draws))
    return dict(accs=np.round(accs, 4).tolist(), b4_acc=b4_acc,
                frac_better_than_b4=float((accs > b4_acc).mean()),
                median=float(np.median(accs)))


# ------------------------------------------------------------ 3. 특징 종류
def exp_features(F, y, g):
    out = {}
    for k in SIGNAL_NAMES:
        Xt, Xp = F[f"time_{k}_{NOISE}"], F[f"psd_{k}_{NOISE}"]
        out[k] = {name: dict(mean=float(a.mean()), std=float(a.std()))
                  for name, a in [("시간 영역", loso(Xt, y, g)), ("PSD", loso(Xp, y, g)),
                                  ("둘 다", loso(np.hstack([Xt, Xp]), y, g))]}
    return out


# ------------------------------------------------------------ 4. 분류기 (nested CV)
def exp_classifiers(F, y, g):
    grids = {
        "로지스틱 회귀": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
                    {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "선형 SVM": (make_pipeline(StandardScaler(), LinearSVC(max_iter=20000, random_state=SEED)),
                   {"linearsvc__C": [0.01, 0.1, 1, 10]}),
        "K-NN": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
                 {"kneighborsclassifier__n_neighbors": [1, 5, 15, 45]}),
    }
    out = {}
    for k in SIGNAL_NAMES:
        X = F[f"time_{k}_{NOISE}"]
        out[k] = {}
        for name, (pipe, grid) in grids.items():
            accs, picked = [], []
            for s in np.unique(g):
                tr, te = g != s, g == s
                # 안쪽 교차검증도 속도 기준 (학습 데이터의 두 속도로 leave-one-speed-out)
                gs = GridSearchCV(pipe, grid, cv=LeaveOneGroupOut(), n_jobs=-1)
                gs.fit(X[tr], y[tr], groups=g[tr])
                accs.append(float((gs.predict(X[te]) == y[te]).mean()))
                picked.append(list(gs.best_params_.values())[0])
            out[k][name] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), picked=picked)
    return out


# ------------------------------------------------------------ 5. 검증 방법과 누수
def exp_leakage(F, y, g):
    out = {}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for k in SIGNAL_NAMES:
        X = F[f"time_{k}_{NOISE}"]
        honest = loso(X, y, g)
        # (a) 무작위 5-fold: 같은 조건의 반복 측정이 학습과 테스트에 섞인다
        rnd = np.array([acc_fit(lr(), X[tr], y[tr], X[te], y[te]) for tr, te in skf.split(X, y)])
        # (b) 정규화를 전체 데이터로 먼저 fit (테스트 데이터의 평균·분산이 학습에 새어 들어감)
        Xz = StandardScaler().fit_transform(X)
        pre = loso(Xz, y, g, model=LogisticRegression(max_iter=5000, random_state=SEED))
        out[k] = {"속도 기준 fold (올바름)": dict(mean=float(honest.mean()), std=float(honest.std())),
                  "전체 데이터로 정규화": dict(mean=float(pre.mean()), std=float(pre.std())),
                  "무작위 5-fold (반복 측정 섞임)": dict(mean=float(rnd.mean()), std=float(rnd.std()))}
    return out


# ------------------------------------------------------------ 6. 가짜 정답에서의 누수
def exp_shuffled_labels(X, y, g, types, k_max=6):
    """정답(방향)을 무작위로 섞은 데이터: 신호와 정답 사이에 관계가 전혀 없으므로 정직한 정확도는 우연(12.5%)이어야 한다.
    그런데 테스트 정답을 보고 특징을 고르면 우연보다 높은 '가짜 성능'이 나온다 = 누수가 만드는 착시.
    같은 조건의 반복 측정 20개는 같은 가짜 정답을 갖도록 조건 단위로 섞는다."""
    y_cond = y[::20]  # 특징 배열은 조건마다 20개씩 연속으로 저장되어 있다 (03 참고)
    y_fake = np.repeat(y_cond[np.random.default_rng(SEED + 1).permutation(len(y_cond))], 20)
    return exp_feature_selection_k(X, y_fake, g, types, k_max)


def exp_feature_selection_k(X, y, g, types, k_max):
    global K_MAX
    old, K_MAX = K_MAX, k_max
    try:
        return exp_feature_selection(X, y, g, types)
    finally:
        K_MAX = old


def main():
    t0 = time.time()
    F = dict(np.load(DATA / "features.npz", allow_pickle=False))
    y, g, types = F["y"], F["speed"], F["types"]
    XA = F[f"time_A_{NOISE}"]
    print(f"== 잡음 {NOISE}배 ==")
    b4 = float(loso(F[f"time_B4_{NOISE}"], y, g).mean())
    full_a = float(loso(XA, y, g).mean())

    print("1. 특징 선택"); r1 = exp_feature_selection(XA, y, g, types)
    print(f"   올바른 방법: {np.round(r1['honest_mean'], 3)}\n   누수 방법:   {np.round(r1['leaky_mean'], 3)}")
    print("2. 무작위 4개 유형 vs B4"); r2 = exp_random4(XA, y, g, types, b4)
    print(f"   B4 {b4:.3f}, 무작위4 중앙값 {r2['median']:.3f}, B4보다 나은 비율 {r2['frac_better_than_b4']:.2f}")
    print("3. 특징 종류"); r3 = exp_features(F, y, g)
    for k, v in r3.items():
        print("  ", k, {n: round(d["mean"], 3) for n, d in v.items()})
    print("4. 분류기 (nested CV)"); r4 = exp_classifiers(F, y, g)
    for k, v in r4.items():
        print("  ", k, {n: (round(d["mean"], 3), d["picked"]) for n, d in v.items()})
    print("5. 검증 방법과 누수"); r5 = exp_leakage(F, y, g)
    for k, v in r5.items():
        print("  ", k, {n: round(d["mean"], 3) for n, d in v.items()})

    print("6. 가짜 정답(섞은 방향)에서 특징 선택"); r6 = exp_shuffled_labels(XA, y, g, types)
    print(f"   올바른 방법: {np.round(r6['honest_mean'], 3)}\n   누수 방법:   {np.round(r6['leaky_mean'], 3)}")

    res = dict(noise=NOISE, chance=CHANCE, full_A=full_a, B4=b4, feature_selection=r1,
               random4=r2, features=r3, classifiers=r4, leakage=r5, shuffled=r6)
    tag = "" if NOISE == 1.0 else f"_noise{NOISE}"
    (WEB / f"ml{tag}.json").write_text(json.dumps(res, ensure_ascii=False))
    plot(res, tag)
    print(f"완료 {time.time() - t0:.0f}초")


def plot(res, tag):
    plt = setup_matplotlib()
    ch = CHANCE * 100
    sub = f"(잡음 {res['noise']}배, 속도 기준 교차검증)"

    # 1. 특징 선택 곡선
    r = res["feature_selection"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    k = np.array(r["k"])
    ax.errorbar(k, np.array(r["honest_mean"]) * 100, yerr=np.array(r["honest_std"]) * 100, marker="o",
                lw=2.2, capsize=3, color="#1f77b4", label="올바른 선택 (학습 데이터 안에서만)")
    ax.errorbar(k, np.array(r["leaky_mean"]) * 100, yerr=np.array(r["leaky_std"]) * 100, marker="s",
                lw=2, ls="--", capsize=3, color="#d62728", label="누수 있는 선택 (테스트 정답을 보고)")
    ax.axhline(res["full_A"] * 100, color="#1f77b4", lw=1, ls=":", label=f"65개 유형 전부 ({res['full_A'] * 100:.1f}%)")
    ax.axhline(res["B4"] * 100, color=SIGNAL_COLORS["B4"], lw=1.5, ls=":", label=f"가상 EEG 4채널 ({res['B4'] * 100:.1f}%)")
    ax.axhline(ch, color="k", lw=1, ls="--", label="우연 수준")
    ax.set_xlabel("고른 세포 유형 수 (유형마다 특징 4개)")
    ax.set_ylabel("테스트 정확도 (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"순차 전진 선택: 몇 종류의 뉴런이면 충분한가?\n{sub}")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(RESULTS / f"fig_ml_feature_selection{tag}.png")
    plt.close(fig)

    # 1-b. 선택 빈도
    cnt = r["top4_counts"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(cnt)[:12]
    ax.bar(names, [cnt[n] for n in names], color="#4c72b0")
    ax.set_ylabel("뽑힌 횟수 (3 fold x 상위 4개)")
    ax.set_title("fold마다 먼저 뽑힌 세포 유형 (안정적으로 뽑히면 중요한 유형)")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(RESULTS / f"fig_ml_selected_types{tag}.png")
    plt.close(fig)

    # 2. 무작위 4개 유형
    r = res["random4"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(np.array(r["accs"]) * 100, bins=25, color=SIGNAL_COLORS["A"], alpha=0.8,
            label=f"신호 A에서 무작위 4개 유형 ({len(r['accs'])}번)")
    ax.axvline(r["b4_acc"] * 100, color=SIGNAL_COLORS["B4"], lw=3, label=f"가상 EEG 4채널 ({r['b4_acc'] * 100:.1f}%)")
    ax.axvline(ch, color="k", ls="--", label="우연 수준")
    ax.set_xlabel("테스트 정확도 (%)")
    ax.set_ylabel("횟수")
    ax.set_title(f"채널 수를 4개로 맞추면? — 무작위 4개 유형이 B4보다 나은 비율 "
                 f"{r['frac_better_than_b4'] * 100:.0f}%\n{sub}")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(RESULTS / f"fig_ml_random4{tag}.png")
    plt.close(fig)

    # 3, 4, 5. 묶음 막대그래프
    def grouped(key, title, fname, colors):
        d = res[key]
        methods = list(next(iter(d.values())).keys())
        fig, ax = plt.subplots(figsize=(8, 4.6))
        w = 0.8 / len(methods)
        for mi, m in enumerate(methods):
            xs = np.arange(len(d)) + (mi - (len(methods) - 1) / 2) * w
            vals = [d[s][m]["mean"] * 100 for s in d]
            errs = [d[s][m]["std"] * 100 for s in d]
            bars = ax.bar(xs, vals, w, yerr=errs, capsize=3, label=m, color=colors[mi])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=8)
        ax.axhline(ch, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(d)), [SIGNAL_NAMES[s] for s in d])
        ax.set_ylabel("테스트 정확도 (%)")
        ax.set_ylim(0, 115)
        ax.set_title(f"{title}\n{sub}")
        ax.legend(frameon=False, fontsize=9, ncol=len(methods), loc="upper right")
        fig.savefig(RESULTS / fname)
        plt.close(fig)

    grouped("features", "특징 추출 비교: 시간 영역 vs 주파수(PSD)", f"fig_ml_features{tag}.png",
            ["#4c72b0", "#dd8452", "#55a868"])
    grouped("classifiers", "분류기 비교 (하이퍼파라미터는 안쪽 교차검증으로 선택)", f"fig_ml_classifiers{tag}.png",
            ["#4c72b0", "#8172b3", "#55a868"])
    grouped("leakage", "검증 방법에 따라 정확도가 얼마나 부풀려지나", f"fig_ml_leakage{tag}.png",
            ["#4c72b0", "#ccb974", "#c44e52"])

    # 6. 가짜 정답에서의 누수
    r = res["shuffled"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    k = np.array(r["k"])
    ax.errorbar(k, np.array(r["honest_mean"]) * 100, yerr=np.array(r["honest_std"]) * 100, marker="o",
                lw=2.2, capsize=3, color="#1f77b4", label="올바른 선택 (학습 데이터 안에서만)")
    ax.errorbar(k, np.array(r["leaky_mean"]) * 100, yerr=np.array(r["leaky_std"]) * 100, marker="s",
                lw=2, ls="--", capsize=3, color="#d62728", label="누수 있는 선택 (테스트 정답을 보고)")
    ax.axhline(ch, color="k", lw=1.2, ls="--", label="우연 수준 12.5% (정답이 가짜이므로 이게 정답)")
    ax.set_xlabel("고른 세포 유형 수")
    ax.set_ylabel("테스트 정확도 (%)")
    ax.set_ylim(0, 50)
    ax.set_title(f"정답을 무작위로 섞은 가짜 데이터에서도 '성능'이 나올까?\n{sub}")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(RESULTS / f"fig_ml_shuffled{tag}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
