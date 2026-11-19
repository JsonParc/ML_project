"""
3단계 (2/2): 녹화한 반응으로 신호 A/B 만들기 -> 측정 잡음 더하기 -> 특징 추출

신호
- 신호 A (세포 단위, 65채널): 세포 유형마다 뉴런 전압 평균. 실험실에서 특정 세포 유형만 골라 기록하는 상황.
- 신호 B4 (가상 EEG, 4채널): 세포 유형 구분 없이 눈의 4개 구역마다 모든 뉴런 전압의 합.
- 신호 B1 (가상 EEG, 1채널): 눈 전체 모든 뉴런 전압의 합.
  왜 두 버전? 실제 EEG도 전극이 여러 개다. 공간 정보를 일부 남긴 경우(4)와 다 없앤 경우(1)를 비교한다.

전처리
- 기준선 보정: 막대가 나오기 전 1초(회색 화면) 평균을 채널마다 뺀다. EEG에서 흔히 하는 baseline correction.
  이것은 샘플 하나 안에서만 계산하므로 다른 샘플 정보가 섞이지 않는다(누수 없음).

측정 잡음
- 왜? 시뮬레이션은 결정론적이라 잡음이 없으면 정확도가 무의미하게 100%가 될 수 있다. 실제 기록 잡음을 흉내 낸다.
- 채널마다 독립 가우시안 잡음(시점마다 독립), 표준편차 = 그 채널 신호 표준편차 x (0.1, 0.5, 1, 2).
  한 값으로 정하면 근거를 대기 어려워서 4단계로 바꿔가며 결론이 어떻게 변하는지 본다.
- 신호 표준편차는 "측정 장비의 잡음 크기"를 정하는 값일 뿐 분류기가 배우는 값이 아니라서 전체 데이터로 정한다.
- 조건마다 20번 반복 측정 -> 잡음 단계마다 48 x 20 = 960샘플.

특징 (막대가 나타난 순간부터 끝까지의 구간에서)
- 시간 영역: 채널마다 최고 전압, 최저 전압, 최고점까지 걸린 시간, 평균 전압 = 4개
  왜? 뉴런 전압은 진동하지 않고 짧게 반응하므로 "언제, 얼마나 세게"가 신호 성질에 맞는다.
- PSD(5단계 비교용): Welch 방법으로 구한 6개 주파수 대역의 로그 파워

실행: python scripts/03_build_signals.py   (약 1분)
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA, DIRECTIONS, MEDIA, N_REPEATS, NOISE_LEVELS, RESULTS, SEED,
                    SIGNAL_COLORS, SIGNAL_NAMES, SPEEDS, WEB, setup_matplotlib)

PSD_BANDS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 50)]  # Hz


def build_clean_signals(rec):
    """(유형 x 구역) 합 -> 신호 A, B4, B1. 반환: {이름: (48, T, 채널)}"""
    sums, counts = rec["sums"], rec["counts"]
    A = sums.sum(3) / counts.sum(1)[None, None, :]  # 유형별 평균 전압
    B4 = sums.sum(2)                                # 구역별 합
    B1 = sums.sum((2, 3))[..., None]                # 전체 합
    return {"A": A, "B4": B4, "B1": B1}


def baseline_correct(sig, n_pre):
    return sig - np.nanmean(sig[:, :n_pre], axis=1, keepdims=True)


def time_features(seg, fs):
    """seg: (프레임, 채널) -> (채널*4,) [최고, 최저, 최고점 시간(초), 평균]"""
    return np.concatenate([seg.max(0), seg.min(0), seg.argmax(0) / fs, seg.mean(0)])


def psd_features(seg, fs):
    f, p = welch(seg, fs=fs, nperseg=min(64, len(seg)), axis=0)
    bands = [np.log10(p[(f >= lo) & (f < hi)].mean(0) + 1e-12) for lo, hi in PSD_BANDS]
    return np.concatenate(bands)


def main():
    rec = dict(np.load(DATA / "recordings.npz", allow_pickle=False))
    fs = int(rec["fs"])
    n_pre = int(rec["t_pre"] * fs)  # 막대가 나오기 전 회색 화면 구간
    lengths = rec["lengths"]
    clean = {k: baseline_correct(v, n_pre) for k, v in build_clean_signals(rec).items()}
    n_cond = len(lengths)
    y_dir = np.array([DIRECTIONS.index(int(a)) for a in rec["angle"]])

    # 채널별 신호 표준편차 (막대 구간 전체, 48조건)
    sig_std = {}
    for k, s in clean.items():
        seg = np.concatenate([s[i, n_pre:lengths[i]] for i in range(n_cond)])
        sig_std[k] = seg.std(0)

    out = {"y": np.repeat(y_dir, N_REPEATS),
           "speed": np.repeat(rec["speed"], N_REPEATS),
           "intensity": np.repeat(rec["intensity"], N_REPEATS),
           "cond": np.repeat(np.arange(n_cond), N_REPEATS),
           "repeat": np.tile(np.arange(N_REPEATS), n_cond)}
    for si, (k, s) in enumerate(clean.items()):
        # 잡음 없는 특징도 참고로 저장 (조건당 1개)
        out[f"time_{k}_clean"] = np.stack([time_features(s[i, n_pre:lengths[i]], fs)
                                           for i in range(n_cond)])
        for li, lvl in enumerate(NOISE_LEVELS):
            rng = np.random.default_rng([SEED, si, li])  # 신호·잡음 단계마다 고정 시드
            Xt, Xp = [], []
            for i in range(n_cond):
                seg = s[i, n_pre:lengths[i]]
                for _ in range(N_REPEATS):
                    noisy = seg + rng.normal(0, lvl * sig_std[k], size=seg.shape)
                    Xt.append(time_features(noisy, fs))
                    Xp.append(psd_features(noisy, fs))
            out[f"time_{k}_{lvl}"] = np.array(Xt, np.float32)
            out[f"psd_{k}_{lvl}"] = np.array(Xp, np.float32)
        print(f"{SIGNAL_NAMES[k]}: 채널 {s.shape[2]}개 -> 시간 특징 {out[f'time_{k}_1.0'].shape[1]}개,"
              f" PSD 특징 {out[f'psd_{k}_1.0'].shape[1]}개, 샘플 {out[f'time_{k}_1.0'].shape[0]}개")
    out["types"] = rec["types"]
    np.savez_compressed(DATA / "features.npz", **out)

    # 8방향 샘플 수 확인 (표집 편향 방지)
    print("방향별 샘플 수:", np.bincount(out["y"]).tolist())

    make_figures(rec, clean, sig_std, n_pre, fs)


def make_figures(rec, clean, sig_std, n_pre, fs):
    plt = setup_matplotlib()
    types = list(rec["types"])
    lengths = rec["lengths"]
    ang, inten, spd = rec["angle"], rec["intensity"], rec["speed"]

    def cond(a, i=1, s=SPEEDS[1]):
        return int(np.where((ang == a) & (inten == i) & (spd == s))[0][0])

    # ---- (1) 4개 구역 지도 ----
    zone_names = ["왼쪽 위", "오른쪽 위", "왼쪽 아래", "오른쪽 아래"]
    zcol = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    x, y, z = rec["neuron_x"], rec["neuron_y"], rec["neuron_zone"]
    sel = rec["neuron_type_idx"] == types.index("T4a")  # 721개 기둥 위치 (한 유형만 그리면 충분)
    fig, ax = plt.subplots(figsize=(5, 5))
    for k in range(4):
        m = sel & (z == k)
        ax.scatter(x[m], y[m], s=22, marker="h", color=zcol[k], label=f"채널 {k + 1}: {zone_names[k]}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("가상 EEG 4채널 = 눈의 4개 구역\n(점 하나 = 기둥 하나, 구역 안 모든 유형의 뉴런 전압을 더함)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9)
    fig.savefig(RESULTS / "fig_zones.png")
    plt.close(fig)

    # ---- (2) 예시 신호: 반대 방향 두 개(0도, 180도), 잡음 없음 vs 잡음 1배 ----
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for col, a in enumerate([0, 180]):
        i = cond(a)
        t = np.arange(lengths[i] - n_pre) / fs
        panels = [("A", [types.index(c) for c in ["T4a", "T4b", "Mi1", "Tm3"]], ["T4a", "T4b", "Mi1", "Tm3"]),
                  ("B4", range(4), [f"채널{k + 1}" for k in range(4)]),
                  ("B1", [0], ["전체 합"])]
        for row, (k, chs, labels) in enumerate(panels):
            ax = axes[row, col]
            for c, lab in zip(chs, labels):
                s = clean[k][i, n_pre:lengths[i], c]
                line, = ax.plot(t, s, lw=1.8, label=lab)
                ax.plot(t, s + rng.normal(0, sig_std[k][c], size=s.shape), lw=0.5, alpha=0.35,
                        color=line.get_color())
            ax.set_title(f"{SIGNAL_NAMES[k]} — 막대 {a}° 방향", fontsize=11)
            ax.set_ylabel("전압 변화 (임의 단위)" if k == "A" else "전압 합 (임의 단위)")
            ax.legend(fontsize=8, ncol=4, frameon=False, loc="upper right")
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("막대가 나타난 뒤 시간 (초)")
    fig.suptitle("같은 뇌, 세 가지 방식으로 본 신호 (굵은 선 = 잡음 없음, 옅은 선 = 잡음 1배)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_signals_example.png")
    plt.close(fig)

    # ---- (3) 웹용: 방향별 가상 EEG 파형 (밝은 막대, 중간 속도) ----
    web = {"fs": fs, "speed": SPEEDS[1], "directions": DIRECTIONS,
           "zone_names": zone_names, "signals": {}}
    for a in DIRECTIONS:
        i = cond(a)
        seg = slice(n_pre, lengths[i])
        web["signals"][str(a)] = {
            "B4": np.round(clean["B4"][i, seg].T / sig_std["B4"][:, None], 3).tolist(),
            "B1": np.round(clean["B1"][i, seg, 0] / sig_std["B1"][0], 3).tolist(),
            "T4": {c: np.round(clean["A"][i, seg, types.index(c)], 4).tolist()
                   for c in ["T4a", "T4b", "T4c", "T4d"]},
        }
    web["note"] = "B4/B1은 채널 표준편차로 나눈 값. 잡음 k배 = 표준편차 k인 가우시안을 더한 것"
    (WEB / "signals.json").write_text(json.dumps(web, ensure_ascii=False))

    # ---- (4) 자극 GIF (방향별, 웹용) ----
    make_stimulus_gifs(rec, plt)
    # ---- (5) 겹눈 격자 위 뉴런 활동 애니메이션 (MP4) ----
    make_activity_video(rec, clean, n_pre, fs, plt)


def make_activity_video(rec, clean, n_pre, fs, plt):
    """예시 조건(밝은 막대, 0도, 가장 빠른 속도)에서 막대 자극과 T4a~d 활동을 육각 격자 위에 그린다.
    오른쪽 아래에는 같은 순간의 가상 EEG 4채널 파형을 함께 그려, 세포 활동이 어떻게 섞이는지 보여준다."""
    import matplotlib
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    import imageio_ffmpeg

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    act = rec["ex_act"].astype(np.float32)
    stim = rec["ex_stim"].astype(np.float32)
    et, ex, ey = rec["ex_types"], rec["ex_x"], rec["ex_y"]
    i = int(np.where((rec["angle"] == 0) & (rec["intensity"] == 1) & (rec["speed"] == SPEEDS[-1]))[0][0])
    T = min(len(act), len(stim), int(rec["lengths"][i]))
    frames = list(range(int(0.8 * fs), T, 2))  # 회색 화면 0.2초 전부터, 50 fps 데이터 -> 25 fps 재생(0.5배속)
    panels = [("자극 (밝은 막대 → 0°)", None)] + [(c, c) for c in ["T4a", "T4b", "T4c", "T4d"]]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7))
    axes = axes.ravel()
    scs = []
    for ax, (title, ct) in zip(axes, panels):
        if ct is None:
            sc = ax.scatter(rec["stim_x"], rec["stim_y"], c=stim[frames[0]], cmap="gray", vmin=0, vmax=1,
                            s=14, marker="h")
        else:
            m = et == ct
            base = act[: n_pre, m].mean(0)
            vals = act[:, m] - base  # 기준선 대비 변화
            # 색 범위: 상위 0.5% 값의 60%까지 (최댓값 하나에 맞추면 대부분이 옅게 보여서)
            lim = 0.6 * np.percentile(np.abs(vals[n_pre:T]), 99.5)
            sc = ax.scatter(ex[m], ey[m], c=vals[frames[0]], cmap="RdBu_r", vmin=-lim, vmax=lim,
                            s=14, marker="h")
            sc._vals = vals
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")
        scs.append(sc)
    ax = axes[5]
    t = (np.arange(T) - n_pre) / fs
    B4 = clean["B4"][i, :T]
    for k in range(4):
        ax.plot(t, B4[:, k] / np.abs(B4[n_pre:]).max(), lw=1.5, label=f"채널{k + 1}")
    cursor = ax.axvline(t[frames[0]], color="k", lw=1)
    ax.set_title("가상 EEG 4채널 (구역별 합)", fontsize=12)
    ax.set_xlabel("시간 (초)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_xlim(t[frames[0]], t[T - 1])
    txt = fig.suptitle("", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(f):
        scs[0].set_array(stim[f])
        for sc in scs[1:]:
            sc.set_array(sc._vals[f])
        cursor.set_xdata([t[f], t[f]])
        txt.set_text(f"겹눈 721개 기둥 위의 뉴런 활동 (빨강=탈분극, 파랑=과분극)   t = {t[f]:+.2f} 초")
        return scs + [cursor, txt]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    # mp4: 웹 재생이 바로 시작되도록 faststart. H.264는 크롬·엣지·사파리에서 재생됨
    anim.save(MEDIA / "activity.mp4", writer=FFMpegWriter(fps=25, bitrate=1800,
              extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]), dpi=100)
    plt.close(fig)
    # WebM(VP9): H.264를 못 트는 브라우저(VS Code 내장 브라우저 등)용. 페이지는 WebM을 먼저 시도한다
    import subprocess

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(MEDIA / "activity.mp4"),
                    "-c:v", "libvpx-vp9", "-b:v", "1500k", "-pix_fmt", "yuv420p", str(MEDIA / "activity.webm")],
                   check=True)
    print(f"활동 애니메이션 저장: {MEDIA / 'activity.mp4'} ({len(frames)} 프레임)")


def make_stimulus_gifs(rec, plt):
    """02에서 렌더링한 막대 자극(71개 위치)을 육각 격자 위에 그려 방향별 GIF로 저장 (가볍게: 작은 크기, 25 fps)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    offs = np.load(DATA / "stim_offsets.npz")
    sx, sy = rec["stim_x"], rec["stim_y"]
    gray = np.full((6, len(sx)), 0.5, np.float32)  # 앞뒤에 회색 화면 잠깐
    for inten, tag in [(1, "bright"), (0, "dark")]:
        for a in DIRECTIONS:
            bright = offs[str(a)]
            s = np.concatenate([gray, bright if inten == 1 else 1.0 - bright, gray])
            fig, ax = plt.subplots(figsize=(2.4, 2.4))
            fig.patch.set_facecolor("black")
            sc = ax.scatter(sx, sy, c=s[0], cmap="gray", vmin=0, vmax=1, s=9, marker="h")
            ax.set_aspect("equal")
            ax.axis("off")
            fig.subplots_adjust(0, 0, 1, 1)

            def update(f, sc=sc, s=s):
                sc.set_array(s[f])
                return (sc,)

            anim = FuncAnimation(fig, update, frames=len(s), blit=True)
            anim.save(MEDIA / f"stimulus_{tag}_{a:03d}.gif", writer=PillowWriter(fps=25), dpi=60)
            plt.close(fig)
    print(f"자극 GIF {2 * len(DIRECTIONS)}개 저장: {MEDIA}")


if __name__ == "__main__":
    main()
