import numpy as np

N_heatmaps = 10

gradient = np.abs(np.random.rand(1), dtype=np.float32)

heatmap_weights = np.abs(np.random.rand(N_heatmaps), dtype=np.float32)

heatmap_weights = heatmap_weights/heatmap_weights.mean(axis=-1, keepdims=True)

heatmap_noise = np.abs(np.random.rand(N_heatmaps), dtype=np.float32)

heatmap_noise = heatmap_noise-heatmap_noise.mean(axis=-1, keepdims=True)

f_input = (gradient * heatmap_weights) + heatmap_noise + 1

powers = np.arange(N_heatmaps)+1

f_out = f_input**powers

print(f_out)
print(gradient.squeeze()), 
print((f_out**(1/powers)).mean(axis=-1) - 1)