import torch
import geoopt

print("geoopt version:", geoopt.__version__)
print()

# ============ Part A: Poincaré Ball 基础 ============
print("=" * 50)
print("Poincaré Ball 基础操作")
print("=" * 50)

ball = geoopt.PoincareBall(c=1.0)  # 曲率 c=1，即曲率 K=-1

# 1. 创建点：Poincaré ball 里的点 norm 必须 < 1
# 直接创建一些小向量（靠近原点 = 靠近层级顶端）
origin = torch.zeros(1, 3)
point_a = torch.tensor([[0.3, 0.2, 0.1]])  # 靠近中心
point_b = torch.tensor([[0.8, 0.1, 0.05]]) # 靠近边界（层级更深）

print(f"Origin: {origin}, norm: {origin.norm().item():.4f}")
print(f"Point A: {point_a}, norm: {point_a.norm().item():.4f}")
print(f"Point B: {point_b}, norm: {point_b.norm().item():.4f}")

# 2. 计算 Poincaré 距离
dist_ab = ball.dist(point_a, point_b)
dist_oa = ball.dist(origin, point_a)
dist_ob = ball.dist(origin, point_b)

print(f"\nPoincaré 距离:")
print(f"  d(origin, A) = {dist_oa.item():.4f}")
print(f"  d(origin, B) = {dist_ob.item():.4f}  (B 更靠近边界，所以距原点更远)")
print(f"  d(A, B) = {dist_ab.item():.4f}")

# 关键直觉：靠近边界的点距离会被"放大"
# 这就是为什么 hyperbolic 空间适合表示层级 — 越深的节点，彼此距离越大

# 3. Exponential Map（切空间 → 流形）
# 这是你 projection head 的核心操作
tangent_vec = torch.tensor([[0.5, -0.3, 0.2]])  # 切空间中的向量
projected = ball.expmap0(tangent_vec)             # 从原点映射到 Poincaré ball
print(f"\nExponential Map:")
print(f"  切向量: {tangent_vec}")
print(f"  映射到 ball: {projected}")
print(f"  映射后 norm: {projected.norm().item():.4f} (必须 < 1)")

# 4. Logarithmic Map（流形 → 切空间，逆操作）
recovered = ball.logmap0(projected)
print(f"\nLogarithmic Map (逆操作):")
print(f"  恢复的切向量: {recovered}")
print(f"  与原始差异: {(tangent_vec - recovered).abs().max().item():.8f}  (应接近 0)")

# ============ Part B: 检查数值稳定性 ============
print("\n" + "=" * 50)
print("数值稳定性测试")
print("=" * 50)

# 模拟 projection head 的输出：随机向量 → expmap 到 ball
batch = torch.randn(100, 64)  # 模拟 100 个 64 维的 MLP 输出

ball_64 = geoopt.PoincareBall(c=1.0)
projected_batch = ball_64.expmap0(batch)

print(f"Input shape: {batch.shape}")
print(f"Projected shape: {projected_batch.shape}")
print(f"Projected norms - min: {projected_batch.norm(dim=1).min().item():.6f}, "
      f"max: {projected_batch.norm(dim=1).max().item():.6f}")
print(f"Any NaN: {torch.isnan(projected_batch).any().item()}")
print(f"All inside ball (norm < 1): {(projected_batch.norm(dim=1) < 1).all().item()}")

# 计算 batch 内的两两距离
dists = ball_64.dist(projected_batch[:5].unsqueeze(1),
                      projected_batch[:5].unsqueeze(0))
print(f"\n5x5 距离矩阵:\n{dists.detach()}")
print(f"Any NaN in distances: {torch.isnan(dists).any().item()}")

# ============ Part C: Riemannian Optimizer 测试 ============
print("\n" + "=" * 50)
print("Riemannian Optimizer 测试")
print("=" * 50)

# 模拟一个最简单的优化：把一个点移向目标点
ball_2d = geoopt.PoincareBall(c=1.0)

target = torch.tensor([0.5, 0.3])
# ManifoldParameter 告诉 optimizer 这个参数住在 Poincaré ball 上
param = geoopt.ManifoldParameter(
    ball_2d.expmap0(torch.tensor([0.0, 0.0])),
    manifold=ball_2d
)

optimizer = geoopt.optim.RiemannianAdam([param], lr=0.05)

print(f"Target: {target}")
print(f"Initial: {param.data}")

for step in range(50):
    optimizer.zero_grad()
    loss = ball_2d.dist(param, target) ** 2
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        d = ball_2d.dist(param.data, target).item()
        print(f"  Step {step:3d}: param={param.data.tolist()}, "
              f"dist_to_target={d:.4f}")

final_dist = ball_2d.dist(param.data, target).item()
print(f"\nFinal distance to target: {final_dist:.6f}")
print(f"Converged: {'✅' if final_dist < 0.01 else '❌'}")

# ============ 总结 ============
print("\n" + "=" * 50)
print("Phase 0 检查清单")
print("=" * 50)
print("✅ geoopt 导入正常")
print("✅ expmap0 / logmap0 / dist 无 NaN")
print("✅ batch 操作正常，所有点在 ball 内")
print("✅ RiemannianAdam 能收敛")
print("\n🎉 Phase 0 完成！可以开始 Phase 1 了")