"""
2단계: 논문 핵심 결과 재현 — T4/T5 방향 선택성 (공식 튜토리얼 04 "Moving Edge Responses" 방식)

두 모드로 나뉜다.
  python scripts/01_reproduce_t4t5.py record   # 무거운 시뮬레이션. 모델마다 data/reproduce/에 캐시.
                                               # 중간에 멈춰도 다시 실행하면 이어서 한다.
  python scripts/01_reproduce_t4t5.py plot     # 캐시만 읽어 그림·JSON·GIF·MP4 생성 (수십 초)

왜 이렇게 나눴나
- 사전학습 모델 하나에 24개 자극(12방향 x 밝은/어두운 가장자리)을 넣는 데 노트북 CPU로 약 1분 반.
  앙상블 50개 전부는 약 1시간이라, 한 번만 돌려 캐시하고 그림은 캐시에서 빠르게 다시 그린다.
- 모델은 "검증 오차(과제 오차)가 낮은 순서"로 기록한다. 중간에 멈춰도 성능 상위 모델 결과는 확보된다.

머신러닝 관점
- 네트워크는 재학습하지 않는다(평가만). 이 모델은 커넥톰으로 구조가 고정된 RNN이고,
  학습 때 쓰지 않은 질문(방향 선택성)에 실제 뇌와 같은 답을 내는지 = 일반화 평가.
- 앙상블 50개는 같은 조건으로 학습했어도 서로 다른 해에 도달한다.
  "과제를 잘 푸는 모델(검증 오차 낮음)일수록 실제 뇌처럼 반응하는가"를 산점도로 본다(모델 선택의 근거).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "reproduce"
RESULTS = ROOT / "results"
WEB = RESULTS / "web"
MEDIA = RESULTS / "media"
for d in (CACHE, RESULTS, WEB, MEDIA):
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(0)
torch.manual_seed(0)

T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]
ANGLES = list(range(0, 360, 30))  # 12방향, 논문·튜토리얼과 같은 30도 간격
SPEED = 19  # 반 기둥(half ommatidia)/초. 50개 모델을 다 돌릴 수 있게 속도는 하나만 쓴다.
DT = 1 / 200  # 튜토리얼과 같은 적분 간격 5 ms (1/100로 키우면 전압이 최대 0.49 달라져서 유지)


def make_dataset():
    from flyvis.datasets.moving_bar import MovingEdge

    # 튜토리얼 04의 moving_edge_responses 기본 설정과 같게 하고 속도만 하나로 줄였다.
    return MovingEdge(
        offsets=(-10, 11), intensities=[0, 1], speeds=[SPEED], angles=ANGLES,
        height=80, post_pad_mode="continue", t_pre=1.0, t_post=1.0, dt=DT, device="cpu",
    )


def load_ensemble_ranked():
    """앙상블 50개를 불러와 검증 오차(과제 오차) 오름차순으로 정렬한 (이름, 오차) 목록."""
    from flyvis.network import Ensemble

    ens = Ensemble("flow/0000")
    # 공개된 사전학습 모델에는 검증 손실이 'loss' 파일로 저장되어 있다 (flyvis 기본값 'epe' 없음)
    losses = ens.min_validation_losses(subdir="validation", file="loss")
    order = np.argsort(losses)
    return ens, [(ens.names[i], float(losses[i])) for i in order]


# ------------------------------------------------------------------ record
def record():
    from flyvis.analysis.moving_bar_responses import peak_responses

    ens, ranked = load_ensemble_ranked()
    dataset = make_dataset()
    network = None  # 같은 Network 객체를 재사용해 매번 새로 만드는 시간을 아낀다
    for rank, (name, loss) in enumerate(ranked):
        out = CACHE / f"{name.replace('/', '_')}.npz"
        if out.exists():
            continue
        t0 = time.time()
        view = ens[name]
        if network is not None:
            view.network(network=network, lazy=True)
        # flyvis 공식 함수: 중심 기둥의 뉴런(세포 유형마다 1개, 65개)의 전압 반응
        ds = view.moving_edge_responses(dataset=dataset, batch_size=8)
        network = view._network.network
        peak = peak_responses(ds)  # 가장자리가 지나가는 동안의 최대 반응(음수는 0으로 정류)
        cell_types = ds["cell_type"].values.astype(str)
        keep = np.isin(cell_types, T4 + T5)
        np.savez_compressed(
            out,
            name=name, rank=rank, val_loss=loss,
            angle=ds["angle"].values, intensity=ds["intensity"].values,
            time=ds["time"].values, cell_types=cell_types,
            peak=peak.values[0].astype(np.float32),  # (24 자극, 65 세포 유형)
            # T4/T5 시계열은 웹에서 반응 곡선을 그릴 때 쓰려고 저장 (float16으로 가볍게)
            traces=ds["responses"].values[0][:, :, keep].astype(np.float16),
            trace_types=cell_types[keep],
        )
        view._clear_memory()  # flyvis 자체 캐시는 지워 디스크를 아낀다 (우리 npz가 캐시)
        print(f"[{rank + 1:2d}/50] {name} 검증오차 {loss:.3f} — {time.time() - t0:.0f}초", flush=True)


# ------------------------------------------------------------------ plot
def load_cache():
    files = sorted(CACHE.glob("*.npz"), key=lambda f: int(np.load(f)["rank"]))
    return [dict(np.load(f, allow_pickle=False)) for f in files]


def tuning(m, cell_type):
    """한 모델의 방향 튜닝 곡선: 각 방향에서의 최대 반응 (T4는 밝은, T5는 어두운 가장자리)."""
    inten = 1 if cell_type.startswith("T4") else 0
    j = list(m["cell_types"]).index(cell_type)
    sel = m["intensity"] == inten
    ang = m["angle"][sel]
    r = m["peak"][sel, j][np.argsort(ang)]
    return r


def dsi_and_pd(r):
    """방향 선택성 지수(DSI)와 선호 방향(도). flyvis와 같은 벡터 합 정의.
    DSI = |sum r(θ)e^{iθ}| / sum r(θ). 0이면 모든 방향에 같게, 1이면 한 방향에만 반응."""
    th = np.deg2rad(ANGLES)
    v = np.sum(r * np.exp(1j * th))
    return float(np.abs(v) / (r.sum() + 1e-12)), float(np.rad2deg(np.angle(v)) % 360)


def angdist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def plot():
    sys.path.insert(0, str(Path(__file__).parent))
    from common import setup_matplotlib

    from flyvis.utils.groundtruth_utils import preferred_directions, tuning_curves

    plt = setup_matplotlib()
    models = load_cache()
    n = len(models)
    print(f"캐시된 모델 {n}개")
    best = models[0]
    TOPK = min(10, n)
    top = models[:TOPK]

    # 실제 측정(Maisak et al. 2013)의 튜닝 곡선 (flyvis에 포함된 정답 데이터)
    gt = {ct: np.array(tuning_curves[ct]) for ct in T4 + T5}

    th = np.deg2rad(ANGLES + [ANGLES[0]])
    colors = {"T4a": "#d62728", "T4b": "#1f77b4", "T4c": "#2ca02c", "T4d": "#ff7f0e"}
    colors.update({c.replace("T4", "T5"): v for c, v in colors.items()})

    def norm(x):
        return x / (x.max() + 1e-12)

    # ---- 모델별 지표: 선호 방향을 맞힌 하위 유형 수, 실제 측정과의 튜닝 상관 ----
    rows = []
    for m in models:
        dsis, errs, corrs = [], [], []
        for ct in T4 + T5:
            r = tuning(m, ct)
            d, pd = dsi_and_pd(r)
            dsis.append(d)
            errs.append(angdist(pd, preferred_directions[ct]))
            # 반응이 모든 방향에서 0이면(정류 후) 상관을 정의할 수 없으므로 0으로 둔다
            corrs.append(float(np.corrcoef(r, gt[ct])[0, 1]) if r.std() > 1e-9 else 0.0)
        rows.append(dict(name=str(m["name"]), rank=int(m["rank"]), val_loss=float(m["val_loss"]),
                         dsi=float(np.mean(dsis)), pd_err=float(np.mean(errs)),
                         corr=float(np.nan_to_num(np.mean(corrs))),
                         n_correct=int(np.sum(np.array(errs) < 45)),
                         correct={ct: bool(e < 45) for ct, e in zip(T4 + T5, errs)}))
    loss = np.array([r["val_loss"] for r in rows])
    corr = np.array([r["corr"] for r in rows])
    ncor = np.array([r["n_correct"] for r in rows])
    from scipy.stats import spearmanr

    rho_c = float(spearmanr(loss, corr).statistic)
    rho_n = float(spearmanr(loss, ncor).statistic)
    groups = [(i, min(i + 10, n)) for i in range(0, n, 10)]  # 검증 오차 순위 10개씩
    grp_mean = [float(ncor[a:b].mean()) for a, b in groups]
    grp_std = [float(ncor[a:b].std()) for a, b in groups]

    # ---- (1) 하위 유형별 극좌표 그래프 8개 모음 ----
    # 최고 모델(굵은 선) + 상위 10개 모델 각각(옅은 선). 평균을 내면 서로 다른 해가 섞여 뭉개지므로 각각 그린다.
    fig, axes = plt.subplots(2, 4, subplot_kw={"projection": "polar"}, figsize=(13, 9.4))
    for row, group in enumerate([T4, T5]):
        for col, ct in enumerate(group):
            ax = axes[row, col]
            for k, m in enumerate(top[1:]):
                r = norm(tuning(m, ct))
                ax.plot(th, np.r_[r, r[0]], color=colors[ct], lw=0.8, alpha=0.3,
                        label=f"상위 2~{TOPK}위 모델 (각각)" if k == 0 else None)
            ax.plot(th, np.r_[norm(gt[ct]), norm(gt[ct])[0]], color="0.25", ls="--", lw=1.6,
                    label="실제 측정 (Maisak 2013)")
            r_best = norm(tuning(best, ct))
            ax.plot(th, np.r_[r_best, r_best[0]], color=colors[ct], lw=2.8, label="최고 모델 (검증 오차 1위)")
            ax.fill(th, np.r_[r_best, r_best[0]], color=colors[ct], alpha=0.12)
            ok = sum(r_["correct"][ct] for r_ in rows[:TOPK])
            ax.set_title(f"{ct} ({'밝은' if row == 0 else '어두운'} 가장자리)\n상위 {TOPK}개 중 {ok}개 정답",
                         fontsize=11.5, pad=14)
            ax.set_yticklabels([])
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
            ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle("T4/T5 하위 유형의 방향 튜닝: 모델 예측 vs 실제 측정\n"
                 "(반지름 = 정규화한 최대 전압 반응, 각도 = 가장자리가 움직이는 방향, 정답 = 선호 방향 오차 45° 미만)",
                 fontsize=13.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=6)
    fig.savefig(RESULTS / "fig_reproduce_all.png", dpi=150)
    plt.close(fig)

    # ---- (2) 하위 유형별 개별 그림 (최고 모델) ----
    for ct in T4 + T5:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4.2, 4.6))
        r_best = norm(tuning(best, ct))
        ax.plot(th, np.r_[norm(gt[ct]), norm(gt[ct])[0]], color="0.25", ls="--", label="실제 측정")
        ax.plot(th, np.r_[r_best, r_best[0]], color=colors[ct], lw=2.5, label="모델 (검증 오차 1위)")
        ax.fill(th, np.r_[r_best, r_best[0]], color=colors[ct], alpha=0.15)
        ax.set_yticklabels([])
        ax.set_title(f"{ct} 방향 튜닝", fontsize=13)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_reproduce_{ct}.png", dpi=150)
        plt.close(fig)

    # ---- (3) 앙상블: 과제 오차 vs 생물학적 정확도 (모델 선택의 근거) ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, n)
    ax.scatter(loss, ncor + jit, c=loss, cmap="viridis_r", s=45, edgecolor="k", lw=0.4)
    ax.scatter(loss[:1], ncor[:1], marker="*", s=320, color="crimson", edgecolor="k", lw=0.6,
               label="녹화 장치로 쓴 최고 모델", zorder=3)
    ax.set_xlabel("과제 오차 (광류 예측 검증 손실, 낮을수록 좋음)")
    ax.set_ylabel("선호 방향을 맞힌 하위 유형 수 (8개 중)")
    ax.set_yticks(range(9))
    ax.set_title(f"모델 {n}개: 과제 오차 vs 정답 수 (순위 상관 ρ = {rho_n:.2f})")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    labels = [f"{a + 1}~{b}위" for a, b in groups]
    ax.bar(labels, grp_mean, yerr=grp_std, capsize=4, color="#4c72b0", edgecolor="k")
    for i, v in enumerate(grp_mean):
        ax.text(i, v + grp_std[i] + 0.15, f"{v:.1f}", ha="center", fontsize=11)
    ax.set_ylim(0, 8.5)
    ax.set_xlabel("검증 오차 순위 구간 (10개씩)")
    ax.set_ylabel("평균 정답 하위 유형 수 (8개 중)")
    ax.set_title("과제를 잘 푸는 모델일수록 실제 뇌와 더 비슷하다")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_ensemble_task_vs_tuning.png", dpi=150)
    plt.close(fig)

    # ---- (4) 웹용 JSON ----
    def as_list(x):
        return [round(float(v), 4) for v in x]

    rep = {
        "description": "T4/T5 하위 유형별 방향 튜닝 (각 방향 최대 반응). T4=밝은 가장자리, T5=어두운 가장자리",
        "angles_deg": ANGLES, "speed": SPEED, "n_models_cached": n, "top_k": TOPK,
        "known_preferred_deg": {ct: preferred_directions[ct] for ct in T4 + T5},
        "cell_types": {},
    }
    for ct in T4 + T5:
        d1, pd1 = dsi_and_pd(tuning(best, ct))
        rep["cell_types"][ct] = {
            "edge": "bright" if ct.startswith("T4") else "dark",
            "best_model_raw": as_list(tuning(best, ct)),
            "best_model_norm": as_list(norm(tuning(best, ct))),
            "top_models_norm": [as_list(norm(tuning(m, ct))) for m in top],
            "measured_norm": as_list(norm(gt[ct])),
            "best_dsi": round(d1, 3), "best_pref_deg": round(pd1, 1),
            "top_correct": int(sum(r_["correct"][ct] for r_ in rows[:TOPK])),
        }
    tt = list(best["trace_types"])
    step = 2
    rep["traces"] = {"time_s": as_list(best["time"][::step]), "data": {}}
    for k, (a, i) in enumerate(zip(best["angle"], best["intensity"])):
        key = f"{'bright' if i == 1 else 'dark'}_{int(a)}"
        rep["traces"]["data"][key] = {
            ct: as_list(np.nan_to_num(best["traces"][k, ::step, tt.index(ct)].astype(np.float32)))
            for ct in (T4 if i == 1 else T5)
        }
    (WEB / "reproduce.json").write_text(json.dumps(rep, ensure_ascii=False))
    ens = {"description": "앙상블 모델별 과제 오차와 T4/T5 튜닝 정확도",
           "spearman_loss_vs_ncorrect": rho_n, "spearman_loss_vs_corr": rho_c,
           "group_labels": labels, "group_mean_correct": grp_mean, "group_std_correct": grp_std,
           "models": rows}
    (WEB / "ensemble.json").write_text(json.dumps(ens, ensure_ascii=False, indent=1))

    make_edge_gifs(plt)

    print("== 요약 ==")
    print(f"최고 모델 {best['name']} (검증 오차 {float(best['val_loss']):.3f}), 정답 {ncor[0]}/8")
    for ct in T4 + T5:
        d1, pd1 = dsi_and_pd(tuning(best, ct))
        print(f"  {ct}: 실제 {preferred_directions[ct]:3d}° | 최고 모델 {pd1:5.1f}° (DSI {d1:.2f})"
              f" | 상위 {TOPK}개 중 정답 {rep['cell_types'][ct]['top_correct']}개")
    print(f"순위 구간별 평균 정답 수: {dict(zip(labels, np.round(grp_mean, 2)))}")
    print(f"과제 오차 vs 정답 수 스피어만 ρ = {rho_n:.3f}, vs 튜닝 상관 ρ = {rho_c:.3f} (모델 {n}개)")


def make_edge_gifs(plt):
    """재현에 쓴 자극(밝은/어두운 가장자리, 12방향)을 겹눈 육각 격자 위에 그린 GIF. 웹용으로 작게."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    from flyvis import NetworkView
    from flyvis.datasets.moving_bar import MovingEdge
    from flyvis.utils.hex_utils import hex_to_pixel

    if all((MEDIA / f"stimulus_edge_{t}_{a:03d}.gif").exists() for t in ("bright", "dark") for a in ANGLES):
        return
    nodes = NetworkView("flow/0000/000").connectome.nodes
    r1 = nodes.type[:].astype(str) == "R1"  # 광수용체 R1의 위치 = 자극 픽셀(721개)의 위치
    sx, sy = hex_to_pixel(nodes.u[:][r1], nodes.v[:][r1])
    edges = MovingEdge(offsets=(-10, 11), intensities=[0, 1], speeds=[SPEED], angles=ANGLES,
                       height=80, post_pad_mode="continue", t_pre=0.2, t_post=0.3, dt=1 / 100, device="cpu")
    for inten, tag in [(1, "bright"), (0, "dark")]:
        for a in ANGLES:
            s = edges.get(angle=a, width=80, intensity=inten, speed=SPEED).numpy()
            s = s[~np.isnan(s).any(1)][::4]  # 25 fps
            fig, ax = plt.subplots(figsize=(2.4, 2.4))
            fig.patch.set_facecolor("black")
            sc = ax.scatter(sx, sy, c=s[0], cmap="gray", vmin=0, vmax=1, s=9, marker="h")
            ax.set_aspect("equal")
            ax.axis("off")
            fig.subplots_adjust(0, 0, 1, 1)
            anim = FuncAnimation(fig, lambda f, sc=sc, s=s: (sc.set_array(s[f]), sc)[1:], frames=len(s), blit=True)
            anim.save(MEDIA / f"stimulus_edge_{tag}_{a:03d}.gif", writer=PillowWriter(fps=25), dpi=60)
            plt.close(fig)
    print(f"가장자리 자극 GIF {2 * len(ANGLES)}개 저장")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plot"
    {"record": record, "plot": plot}[mode]()
