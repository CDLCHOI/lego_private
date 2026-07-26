import numpy as np
import matplotlib.pyplot as plt

def cosine_alpha_bar_schedule(timesteps, s=0.008):
    steps = timesteps + 1  # 计算从 t=0 到 t=T 的 \bar{\alpha}_t
    x = np.linspace(0, timesteps, steps)
    f_t = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    return alpha_bar

def compute_betas_from_alpha_bar(alpha_bar):
    betas = []
    for t in range(1, len(alpha_bar)):
        beta_t = 1 - alpha_bar[t] / alpha_bar[t - 1]
        betas.append(beta_t)
    return np.array(betas)

timesteps_list = [10, 50, 100, 1000]
target_xmax = 1000  # 统一映射到 [0, 1000]

plt.figure(figsize=(10, 6))

for T in timesteps_list:
    alpha_bar = cosine_alpha_bar_schedule(T)
    betas = compute_betas_from_alpha_bar(alpha_bar)

    # 把原始的 x 坐标线性映射到 [0, 1000]
    x_original = np.arange(len(betas))
    x_scaled = x_original / (len(betas) - 1) * target_xmax

    plt.plot(x_scaled, betas, label=f"T={T}")

plt.title("Beta values from Cosine Schedule (rescaled to T=1000)")
plt.xlabel("Virtual Timestep (0 ~ 1000)")
plt.ylabel("Beta (Variance)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
