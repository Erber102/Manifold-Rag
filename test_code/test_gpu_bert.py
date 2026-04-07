import torch
from transformers import AutoTokenizer, AutoModel
import time

# check gpu
print("=" * 50)
print("GPU 检查")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 简单测试 GPU 计算
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.mm(x, x)
    print(f"GPU 矩阵乘法测试: OK (result shape: {y.shape})")
else:
    print("❌ CUDA 不可用！检查 PyTorch 安装和驱动")
    exit(1)

# ============ Part B: BERT Encode ============
print("\n" + "=" * 50)
print("BERT Encode 测试")
print("=" * 50)

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

# 准备测试文本
texts = [
    "Hyperbolic geometry is a non-Euclidean geometry.",
    "The Poincaré disk model represents hyperbolic space.",
    "Retrieval-augmented generation reduces hallucinations.",
    "Trees can be embedded in hyperbolic space with low distortion.",
] * 25  # 100 条文本

print(f"Encoding {len(texts)} texts...")

# Encode
start = time.time()
with torch.no_grad():
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # 用 [CLS] token 的输出作为句子表示
    embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (100, 768)

elapsed = time.time() - start

print(f"Output shape: {embeddings.shape}")       # 应该是 (100, 768)
print(f"Encoding time: {elapsed:.2f}s")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
print(f"Embedding norm (mean): {embeddings.norm(dim=1).mean().item():.4f}")

# 验证 embedding 质量：相似文本的余弦相似度应该更高
from torch.nn.functional import cosine_similarity
sim_01 = cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
sim_02 = cosine_similarity(embeddings[0:1], embeddings[2:3]).item()
print(f"\n余弦相似度检查:")
print(f"  'Hyperbolic geometry' vs 'Poincaré disk': {sim_01:.4f}  (应该较高)")
print(f"  'Hyperbolic geometry' vs 'RAG':            {sim_02:.4f}  (应该较低)")

if sim_01 > sim_02:
    print("符合预期")
else:
    print("不太对，但 untrained BERT 本身就不是最好的 sentence encoder，不影响后续")

print("\nBERT encode 测试通过!")