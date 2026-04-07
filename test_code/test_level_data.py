import torch
import geoopt
import numpy as np
from collections import deque

# ========== 1. 构建一棵真实的树，计算树上精确距离 ==========

def build_tree_distances(depth=6, branching=3):
    """构建平衡树，返回叶子间的树距离矩阵"""
    # BFS 构建树
    edges = []
    node_id = 0
    queue = deque([(node_id, 0)])  # (node, current_depth)
    all_nodes = [0]
    leaves = []

    while queue:
        parent, d = queue.popleft()
        if d >= depth:
            leaves.append(parent)
            continue
        for _ in range(branching):
            node_id += 1
            all_nodes.append(node_id)
            edges.append((parent, node_id))
            if d + 1 == depth:
                leaves.append(node_id)
            else:
                queue.append((node_id, d + 1))

    # 用 BFS 算叶子间最短路
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    n_leaves = len(leaves)
    dist_matrix = np.zeros((n_leaves, n_leaves))

    for i, src in enumerate(leaves):
        visited = {src: 0}
        q = deque([src])
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if nb not in visited:
                    visited[nb] = visited[node] + 1
                    q.append(nb)
        for j, tgt in enumerate(leaves):
            dist_matrix[i, j] = visited.get(tgt, 0)

    return dist_matrix, n_leaves

# 深度6, 分支3 → 729个叶子
tree_dists, n = build_tree_distances(depth=6, branching=3)
print(f"叶子数: {n}")
print(f"树距离范围: [{tree_dists[tree_dists>0].min():.0f}, {tree_dists.max():.0f}]")

target = torch.tensor(tree_dists, dtype=torch.float32)

# ========== 2. 学习嵌入：Euclidean vs Poincaré ==========

def distortion(pred_dists, true_dists, mask):
    """计算 distortion: 预测距离和真实距离的平均相对误差"""
    ratios = pred_dists[mask] / (true_dists[mask] + 1e-8)
    return (ratios.max() / ratios.min()).item()  # worst-case distortion

def learn_euclidean(target, dim, steps=2000, lr=0.01):
    embeddings = torch.nn.Parameter(torch.randn(n, dim) * 0.01)
    optimizer = torch.optim.Adam([embeddings], lr=lr)
    mask = target > 0

    for step in range(steps):
        optimizer.zero_grad()
        # Euclidean 两两距离
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
        pred = diff.norm(dim=-1)
        loss = ((pred[mask] - target[mask]) ** 2).mean()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            d = distortion(pred.detach(), target, mask)
            print(f"    [Euclid dim={dim}] step {step}: loss={loss.item():.4f}, distortion={d:.2f}")

    pred = (embeddings.unsqueeze(0) - embeddings.unsqueeze(1)).norm(dim=-1)
    return distortion(pred.detach(), target, mask), loss.item()

def learn_poincare(target, dim, steps=2000, lr=0.01):
    ball = geoopt.PoincareBall(c=1.0)
    embeddings = geoopt.ManifoldParameter(
        ball.expmap0(torch.randn(n, dim) * 0.01),
        manifold=ball
    )
    optimizer = geoopt.optim.RiemannianAdam([embeddings], lr=lr)
    mask = target > 0

    for step in range(steps):
        optimizer.zero_grad()
        # Poincaré 两两距离
        pred = ball.dist(embeddings.unsqueeze(0), embeddings.unsqueeze(1))
        loss = ((pred[mask] - target[mask]) ** 2).mean()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            d = distortion(pred.detach(), target, mask)
            print(f"    [Poincaré dim={dim}] step {step}: loss={loss.item():.4f}, distortion={d:.2f}")

    pred = ball.dist(embeddings.unsqueeze(0), embeddings.unsqueeze(1))
    return distortion(pred.detach(), target, mask), loss.item()


# ========== 3. 关键对比：在不同维度下对比 ==========
print("\n" + "=" * 60)
print("维度 | Euclidean Distortion | Poincaré Distortion | 赢家")
print("=" * 60)

for dim in [2, 4, 8, 16]:
    print(f"\n--- dim = {dim} ---")
    e_dist, e_loss = learn_euclidean(target, dim, steps=2000, lr=0.01)
    p_dist, p_loss = learn_poincare(target, dim, steps=2000, lr=0.005)
    winner = "Poincaré ✅" if p_dist < e_dist else "Euclidean"
    print(f"\n  dim={dim:2d} | E distortion={e_dist:.2f} | "
          f"P distortion={p_dist:.2f} | {winner}")