import numpy as np
N_heatmaps = 10
gradient = np.abs(np.random.rand(4, 1), dtype=np.float16)
heatmap_weights = np.abs(np.random.rand(4, N_heatmaps), dtype=np.float16)
heatmap_weights = heatmap_weights/heatmap_weights.mean(axis=-1, keepdims=True)
heatmap_noise = np.abs(np.random.rand(4, N_heatmaps), dtype=np.float16)
heatmap_noise = heatmap_noise-heatmap_noise.mean(axis=-1, keepdims=True)
f_input = (gradient * heatmap_weights) + heatmap_noise*3 + 3
powers = np.arange(N_heatmaps)+1
f_out = f_input**powers
print(f_out)
print(gradient.squeeze())
print((f_out**(1/powers)).mean(axis=-1) - 1)

# --- added: measure the error the eyeball can't ---
recovered = (f_out**(1/powers)).mean(axis=-1) - 3
truth = gradient.squeeze()
print("abs error per row:", np.abs(recovered - truth))
print("mean abs error:   ", np.abs(recovered - truth).mean())

# your exact OG construction, then:
ch0_only = f_input[:, 0] - 33                          # channel 0 estimate (power 1, so f^1)
full     = (f_out**(1/powers)).mean(axis=-1) - 1      # all channels
print("ch0-only err:", np.abs(ch0_only - gradient.squeeze()).mean())
print("full err:    ", np.abs(full - gradient.squeeze()).mean())
print("gradient:"+str(gradient))