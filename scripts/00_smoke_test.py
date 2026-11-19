"""
1단계: 환경 확인 (스모크 테스트)

무엇을 확인하나
- 사전학습 모델 1개를 불러올 수 있는가
- 짧은 자극(움직이는 가장자리 1개)을 넣었을 때 시뮬레이션이 얼마나 걸리고 메모리를 얼마나 쓰는가
- 세포 유형 목록과 개수, 뉴런(기둥)의 공간 좌표(u, v)를 어떻게 얻는가
- 전압 반응의 대략적인 크기

실행: python scripts/00_smoke_test.py
"""

import resource
import time

import numpy as np
import torch

import flyvis
from flyvis import NetworkView
from flyvis.datasets.moving_bar import MovingEdge

torch.manual_seed(0)
np.random.seed(0)


def mem_mb():
    # 리눅스에서 ru_maxrss 단위는 KB -> MB로 변환 (프로세스가 지금까지 쓴 최대 메모리)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


t0 = time.time()
# 사전학습 앙상블 "flow/0000"의 첫 번째 모델(000). 앙상블 안 50개 모델 중 하나다.
view = NetworkView("flow/0000/000")
network = view.init_network()  # 가장 검증 오차가 낮은 체크포인트를 불러온다
print(f"[모델 불러오기] {time.time() - t0:.1f}초, 최대 메모리 {mem_mb():.0f} MB")

# ---- 세포 유형과 뉴런 수 ----
nodes = view.connectome.nodes
cell_types_per_neuron = nodes.type[:].astype(str)  # 뉴런마다 세포 유형 이름
u = nodes.u[:]  # 육각 격자 좌표 (u, v): 뉴런이 놓인 기둥의 위치
v = nodes.v[:]
unique_types = view.connectome.unique_cell_types[:].astype(str)
print(f"[뉴런 수] {len(cell_types_per_neuron)}개")
print(f"[세포 유형 수] {len(unique_types)}개")
print("[세포 유형 목록]", ", ".join(unique_types))
counts = {t: int((cell_types_per_neuron == t).sum()) for t in unique_types}
print("[유형별 뉴런 수 예시]", {k: counts[k] for k in list(counts)[:6]}, "...")
print(f"[유형별 뉴런 수 종류] {sorted(set(counts.values()))}")
print(f"[공간 좌표 u 범위] {u.min()}~{u.max()}, v 범위 {v.min()}~{v.max()}")
print("[학습 파라미터 개수]",
      {n: p.numel() for n, p in network.named_parameters()})

# ---- 짧은 자극 1개: 밝은 가장자리가 0도 방향으로 움직임 ----
dt = 1 / 200  # 공식 튜토리얼과 같은 적분 시간 간격 (5 ms)
edges = MovingEdge(offsets=(-10, 11), intensities=[1], speeds=[19], angles=[0],
                   height=80, post_pad_mode="continue", t_pre=1.0, t_post=1.0, dt=dt,
                   device="cpu")
stim = edges[0]  # (프레임 수, 721 육각 픽셀)
stim = stim[~torch.isnan(stim).any(dim=1)]  # 속도별 길이를 맞추려 붙인 NaN 제거
print(f"[자극 모양] {tuple(stim.shape)} = (프레임, 육각 픽셀), {stim.shape[0] * dt:.2f}초")

t0 = time.time()
with torch.no_grad():
    # (batch, frame, 1, hexals) 모양으로 넣는다. 초기 상태는 1초 회색 화면 뒤 안정 상태.
    act = network.simulate(stim[None, :, None], dt)  # (1, 프레임, 뉴런 수)
sim_time = time.time() - t0
act = act[0].numpy()
print(f"[시뮬레이션 1회] {sim_time:.1f}초, 최대 메모리 {mem_mb():.0f} MB, 출력 {act.shape}")

# ---- 전압 크기 ----
print(f"[전체 전압] 최소 {act.min():.3f}, 최대 {act.max():.3f}, 표준편차 {act.std():.3f}")
for t in ["R1", "L1", "Mi1", "Tm3", "T4a", "T4b", "T5a", "TmY3"]:
    if t in counts:
        a = act[:, cell_types_per_neuron == t]
        print(f"   {t:5s} 평균 {a.mean():+.3f}, 시간 변화폭(최대-최소) {np.ptp(a.mean(1)):.3f}")
