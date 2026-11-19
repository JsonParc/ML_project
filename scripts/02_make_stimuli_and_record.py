"""
3단계 (1/2): 자극 만들고 가상 뇌의 반응 "녹화"하기

- 자극: 8방향(45도 간격) x 밝기 2(밝은/어두운 막대) x 속도 3 = 48조건. 8방향 샘플 수는 모두 같다.
  공식 튜토리얼 07 "Custom Stimuli"처럼 flyvis의 렌더링 도구(HexEye, MovingBar와 같은 함수)로 직접 만든다.
  렌더링이 느려서(막대 1개 약 80초) 대칭을 이용해 4개만 그리고 나머지 12개는 계산으로 만든다.
- 녹화 장치: 성능 1위(검증 오차 최저) 사전학습 모델. 재학습하지 않는다.
  왜? 실제 실험에서 연구자는 뇌를 바꿀 수 없고 기록만 한다. 그 상황을 흉내 낸다.
- 시뮬레이션은 결정론적이라 같은 자극엔 항상 같은 반응 -> 조건당 1번만 돌린다.

저장(data/recordings.npz)
- 원래 계획은 "모든 뉴런(45,669개)의 전압 시계열"이었지만 48조건이면 약 4GB라서,
  (세포 유형 65 x 공간 구역 4)마다 뉴런 전압의 "합"과 뉴런 수를 저장한다.
  신호 A(유형별 평균), 신호 B 4채널(구역별 합), B 1채널(전체 합)이 모두 이 값에서 정확히 계산되므로
  분석에 필요한 정보는 하나도 잃지 않는다.
- 발표용 애니메이션을 위해 예시 조건 1개만 일부 세포 유형의 전체 뉴런 전압을 따로 저장한다.

실행: python scripts/02_make_stimuli_and_record.py   (약 5분)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, DIRECTIONS, FS, INTENSITIES, SEED, SPEEDS, best_model_name

from flyvis import NetworkView
from flyvis.utils.hex_utils import hex_to_pixel

torch.manual_seed(SEED)
np.random.seed(SEED)
DT = 1 / 200  # 시뮬레이션 적분 간격 5 ms (튜토리얼 값)
OUT = DATA / "recordings.npz"
EXAMPLE_TYPES = ["R1", "L1", "Mi1", "Tm3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def zone_of(x, y):
    """눈을 4개 공간 구역(사분면)으로 나눈다. 0=왼쪽 위, 1=오른쪽 위, 2=왼쪽 아래, 3=오른쪽 아래.
    실제 EEG 전극 몇 개가 머리의 넓은 영역을 각각 뭉뚱그려 재는 상황을 흉내 낸다."""
    return (x >= 0).astype(int) + 2 * (y < 0).astype(int)


OFFSETS = np.arange(-35, 36)        # 막대 위치: 반 기둥(2.25도) 단위, 71곳
LED = np.radians(2.25)              # 위치 한 칸 = 2.25도 (flyvis MovingBar와 같은 값)
OMM = np.radians(5.8)               # 기둥 하나 = 5.8도
STIM_CACHE = DATA / "stim_offsets.npz"


def _render_one(angle):
    """밝은 막대 하나를 71개 위치에 그린다 (flyvis HexEye, MovingBar와 같은 함수). 1개에 약 80초."""
    torch.set_num_threads(2)
    from flyvis.datasets.rendering import HexEye

    eye = HexEye(721, 25)
    return eye.render_offset_bars(
        bar_width_rad=4 * LED, bar_height_rad=80 * LED, n_bars=1, offsets=OFFSETS * LED,
        bar_intensity=1, bg_intensity=0.5, moving_angle=angle, bar_loc_horizontal=np.radians(90),
    ).cpu().numpy()


def render_offsets():
    """8방향 밝은 막대 자극. 렌더링이 느려서(16개면 약 22분) 대칭을 써서 4개만 그린다.
    - 방향 a+180도 = 방향 a의 위치 순서를 거꾸로 (막대는 앞뒤 대칭이고, 육각 격자는 점대칭)
    - 어두운 막대 = 1 - 밝은 막대
    결과는 data/stim_offsets.npz에 캐시."""
    if STIM_CACHE.exists():
        z = np.load(STIM_CACHE)
        return {a: torch.from_numpy(z[str(a)]) for a in DIRECTIONS}
    import os
    import subprocess

    base = [a for a in DIRECTIONS if a < 180]
    part = {a: DATA / f"stim_offsets_{a:03d}.npy" for a in base}
    t0 = time.time()
    # 방향마다 파이썬을 따로 띄워 동시에 렌더링 (각자 파일로 저장 -> 하나가 실패해도 어느 것인지 바로 보인다)
    # 렌더링 1개가 메모리를 많이 써서(4개 동시 실행 시 메모리 부족으로 강제 종료됨) 2개씩만 동시에 돌린다.
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    todo = [a for a in base if not part[a].exists()]
    for i in range(0, len(todo), 2):
        procs = {a: subprocess.Popen([sys.executable, __file__, "render", str(a)], env=env)
                 for a in todo[i:i + 2]}
        for a, p in procs.items():
            if p.wait() != 0:
                raise RuntimeError(f"{a}도 렌더링 실패 (종료 코드 {p.returncode}, -9면 메모리 부족)")
    out = {a: np.load(part[a]) for a in base}
    print(f"막대 렌더링 {len(base)}개 (동시 실행): {time.time() - t0:.0f}초")
    for a in base:
        out[a + 180] = out[a][::-1].copy()
    # 대칭 확인: 180도를 실제로 3개 위치만 그려서, 0도를 뒤집은 것과 비교
    from flyvis.datasets.rendering import HexEye

    eye = HexEye(721, 25)
    chk = eye.render_offset_bars(bar_width_rad=4 * LED, bar_height_rad=80 * LED, n_bars=1,
                                 offsets=np.array([-20, 0, 13]) * LED, bar_intensity=1, bg_intensity=0.5,
                                 moving_angle=180, bar_loc_horizontal=np.radians(90)).numpy()
    idx = [list(OFFSETS).index(o) for o in (-20, 0, 13)]
    err = np.abs(chk - out[180][idx]).max()
    print(f"대칭 확인: 실제 180도 렌더링 vs 0도 뒤집기, 최대 차이 {err:.4f}")
    assert err < 0.05, "대칭 가정이 성립하지 않음"
    np.savez_compressed(STIM_CACHE, **{str(a): v.astype(np.float32) for a, v in out.items()})
    return {a: torch.from_numpy(out[a]) for a in DIRECTIONS}


def make_movie(frames, speed):
    """위치 71개 -> 시간 프레임. flyvis MovingBar와 같은 방식(resample, pad):
    회색 1초 + 막대가 speed(반 기둥/초)로 지나가는 시간 + 회색 1초"""
    from flyvis.datasets.rendering.utils import pad, resample

    t_stim = len(OFFSETS) * LED / (speed * OMM)
    s = resample(frames, t_stim, DT, dim=0, device="cpu")
    s = pad(s, t_stim + 1.0, DT, mode="start", fill=0.5)
    s = pad(s, t_stim + 2.0, DT, mode="end", pad_mode="value", fill=0.5)
    return s.float()


def main():
    t_start = time.time()
    render_offsets()  # 모델을 불러오기 전에(프로세스 복제가 안전할 때) 자극부터 렌더링·캐시
    name, loss = best_model_name()
    print(f"녹화 장치: {name} (검증 오차 {loss:.3f})")
    view = NetworkView(name)
    net = view.init_network()

    # ---- 뉴런 정보: 세포 유형, 육각 좌표 -> 평면 좌표 -> 구역 ----
    nodes = view.connectome.nodes
    ntype = nodes.type[:].astype(str)
    types = list(view.connectome.unique_cell_types[:].astype(str))
    x, y = hex_to_pixel(nodes.u[:], nodes.v[:])
    zone = zone_of(x, y)
    type_idx = np.array([types.index(t) for t in ntype])
    # (뉴런 수, 65*4) 지시 행렬: 전압 x 이 행렬 = (유형, 구역)별 합. 행렬곱 한 번으로 빠르게 묶는다.
    group = type_idx * 4 + zone
    M = torch.zeros(len(ntype), len(types) * 4)
    M[torch.arange(len(ntype)), torch.from_numpy(group)] = 1.0
    counts = M.sum(0).numpy().reshape(len(types), 4)

    # ---- 자극: 화면 전체를 가로지르는 긴 막대 ----
    # flyvis 기본 막대(높이 9, 이동 ±10)는 눈 중앙만 지나가서, 4개 구역을 모두 지나가도록
    # 높이 80(사실상 화면 끝까지), 이동 범위 ±35 반 기둥(약 ±79도)으로 늘렸다.
    offsets = render_offsets()  # {각도: (71 위치, 721 픽셀)} 밝은 막대
    stimuli, rows = [], []
    for a in DIRECTIONS:
        for inten in INTENSITIES:
            # 어두운 막대 = 1 - 밝은 막대 (배경이 0.5라서 정확히 성립)
            frames = offsets[a] if inten == 1 else 1.0 - offsets[a]
            for sp in SPEEDS:
                stimuli.append(make_movie(frames, sp))
                rows.append((a, inten, sp))
    import pandas as pd
    arg_df = pd.DataFrame(rows, columns=["angle", "intensity", "speed"])
    print(f"자극 조건 {len(arg_df)}개 (방향 {len(DIRECTIONS)} x 밝기 {len(INTENSITIES)} x 속도 {len(SPEEDS)})")

    # 방향 규약 확인: 각도 0/90도일 때 막대가 평면 좌표에서 어느 쪽으로 움직이는가
    xin, yin = hex_to_pixel(nodes.u[:][ntype == "R1"], nodes.v[:][ntype == "R1"])
    direction_check = {}
    for a in [0, 90]:
        s = offsets[a].numpy()
        w = np.clip(s - 0.5, 0, None)
        t = np.where(w.sum(1) > 0)[0]
        c0 = (w[t[5]] @ np.c_[xin, yin]) / w[t[5]].sum()
        c1 = (w[t[-5]] @ np.c_[xin, yin]) / w[t[-5]].sum()
        d = c1 - c0
        direction_check[a] = (d / np.linalg.norm(d)).round(2).tolist()
    print("막대 이동 방향(평면 좌표, 단위벡터):", direction_check)

    # ---- 시뮬레이션: 속도별로 묶어서(길이가 같으므로) 배치 8개씩 ----
    with torch.no_grad():
        s0 = net.steady_state(1.0, DT, batch_size=1)  # 1초 회색 화면 뒤의 안정 상태에서 시작
    recs, example = [None] * len(arg_df), None
    for speed in SPEEDS:
        idx = arg_df.index[arg_df.speed == speed].values
        for b in range(0, len(idx), 8):
            ib = idx[b:b + 8]
            stim = torch.stack([stimuli[i] for i in ib])  # (B, 프레임, 721), 같은 속도라 길이가 같다
            t0 = time.time()
            with torch.no_grad():
                net.stimulus.zero(len(ib), stim.shape[1])
                net.stimulus.add_input(stim[:, :, None])
                act = net(net.stimulus(), DT, state=s0)  # (B, 프레임, 45669) 전압
            # 200 Hz -> 100 Hz: 이웃한 두 프레임 평균 (저장 용량 절반, 반응은 수십 ms 단위라 충분)
            F = act.shape[1] // 2 * 2
            act100 = act[:, :F].reshape(len(ib), F // 2, 2, -1).mean(2)
            sums = (act100 @ M).reshape(len(ib), F // 2, len(types), 4)
            for k, i in enumerate(ib):
                recs[i] = sums[k].numpy().astype(np.float32)
                r = arg_df.loc[i]
                if r.angle == 0 and r.intensity == 1 and r.speed == SPEEDS[-1]:
                    keep = np.isin(ntype, EXAMPLE_TYPES)
                    example = dict(act=act100[k][:, keep].numpy().astype(np.float16),
                                   stim=stim[k, : F : 2].numpy().astype(np.float16),
                                   types=ntype[keep], x=x[keep], y=y[keep])
            print(f"  속도 {speed}: 조건 {b + 1}~{b + len(ib)} / {len(idx)}, "
                  f"{stim.shape[1]} 프레임, {time.time() - t0:.0f}초", flush=True)

    # 속도마다 길이가 달라서 가장 긴 길이에 맞춰 NaN으로 채워 한 배열로 저장
    T = max(r.shape[0] for r in recs)
    data = np.full((len(recs), T, len(types), 4), np.nan, np.float32)
    lengths = np.array([r.shape[0] for r in recs])
    for i, r in enumerate(recs):
        data[i, : r.shape[0]] = r
    np.savez_compressed(
        OUT, sums=data, lengths=lengths, counts=counts, types=np.array(types),
        angle=arg_df.angle.values, intensity=arg_df.intensity.values, speed=arg_df.speed.values,
        fs=FS, t_pre=1.0, model=name,
        ex_act=example["act"], ex_stim=example["stim"], ex_types=example["types"],
        ex_x=example["x"], ex_y=example["y"], stim_x=xin, stim_y=yin,
        neuron_x=x.astype(np.float32), neuron_y=y.astype(np.float32), neuron_zone=zone,
        neuron_type_idx=type_idx,
    )
    meta = dict(model=name, val_loss=loss, n_conditions=len(recs), fs=FS, lengths=lengths.tolist(),
                zone_neuron_counts=counts.sum(0).tolist(), direction_check=direction_check,
                seconds=round(time.time() - t_start))
    (DATA / "recordings_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"저장: {OUT} ({OUT.stat().st_size / 1e6:.0f} MB), 구역별 뉴런 수 {counts.sum(0)}, "
          f"총 {time.time() - t_start:.0f}초")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "render":  # 하위 프로세스: 한 방향만 렌더링
        a = int(sys.argv[2])
        t0 = time.time()
        np.save(DATA / f"stim_offsets_{a:03d}.npy", _render_one(a).astype(np.float32))
        print(f"  {a}도 렌더링 완료 {time.time() - t0:.0f}초", flush=True)
    else:
        main()
