import math
import matplotlib.pyplot as plt


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


fig, ax = plt.subplots(2, 2, figsize=(6, 4))
fontsize = 12
for k, n in enumerate([5, 10, 100, 1000]):
    # 注意如果是从3开始，那就是low-order远大于high-order，但GNN中S不能为空集
    x = [i / n for i in range(2, n + 1)]
    y = [(n - m + 1) / (n * (n - 1)) / nCr(n - 2, m - 2) for m in range(2, n + 1)]
    ax[k // 2, k % 2].plot(x, y)
    ax[k // 2, k % 2].tick_params(labelsize=fontsize)
    ax[k // 2, k % 2].set_title(f'n = {n}', fontsize=fontsize)
    ax[k // 2, k % 2].set_xlabel('Order / n', fontsize=fontsize)
    ax[k // 2, k % 2].set_ylabel('$F^{(m)}$', fontsize=fontsize)
fig.tight_layout()
plt.savefig(f'../../../../fm_.pdf', bbox_inches='tight')

