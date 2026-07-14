from typing import Protocol, runtime_checkable
from  Utilities.HeatMap import HeatMappable, describeMap
import numpy as np 
import numpy.typing as npt
from numpy.typing import NDArray
import inspect


@runtime_checkable
class MultiHeatMappable(HeatMappable, Protocol):
    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        ...
    def getRange(self) -> tuple[float, float]:
        ...
    def generate_heatmap(self, noise=0, blockSize=1) -> np.ndarray:
        ...  
    def toString(this) -> str:
        ...
    def mapAll(self, cordArr:NDArray[np.int_]) -> NDArray[np.float32]:
        ...
    def mapOne(self, cordArr:NDArray[np.int_]) -> np.float32:
        ...
    def pregen(self):
        ...
    
class MultiPolynomialInverse():
    def __init__(self, N: np.int_, map, **kwargs):
        self.noiseScale = 2
        self.N = N
        inner_sig = inspect.signature(map)
        if any(p.kind == p.VAR_KEYWORD for p in inner_sig.parameters.values()):
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in inner_sig.parameters}
        self.inner = map(**filtered)
        self.heatmap_weights = np.abs(np.random.rand(N), dtype=np.float32)

        self.heatmap_weights = self.heatmap_weights/self.heatmap_weights.mean(axis=-1, keepdims=True)

        self.heatmap_noise = np.abs(np.random.rand(N), dtype=np.float32)

        self.heatmap_noise = self.heatmap_noise-self.heatmap_noise.mean(axis=-1, keepdims=True)

        self.heatmap_noise = self.heatmap_noise*self.noiseScale
    def pregen(self):
        self.heatmap_weights = np.abs(np.random.rand(self.N), dtype=np.float32)

        self.heatmap_weights = self.heatmap_weights/self.heatmap_weights.mean(axis=-1, keepdims=True)

        self.heatmap_noise = np.abs(np.random.rand(self.N), dtype=np.float32)

        self.heatmap_noise = self.heatmap_noise-self.heatmap_noise.mean(axis=-1, keepdims=True)

        self.heatmap_noise = self.heatmap_noise*self.noiseScale

    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        f_input = (self.inner.map(cordArr)*self.heatmap_weights)+self.heatmap_noise+1
        f_input = np.clip(f_input, 1e-3, None)

        powers = np.arange(self.N)+1
        return (f_input**powers)[0]
    def getRange(self) -> tuple[float, float]:
            v_lo, v_hi = self.inner.getRange()
            powers = np.arange(self.N) + 1                      # [1..N]
            # f = v*w + n + 1, per channel; w >= 0 so f is increasing in v
            f_lo = v_lo * self.heatmap_weights + self.heatmap_noise + 1
            f_hi = v_hi * self.heatmap_weights + self.heatmap_noise + 1
            f_lo = np.clip(v_lo * self.heatmap_weights + self.heatmap_noise + 1, 1e-3, None)
            f_hi = np.clip(v_hi * self.heatmap_weights + self.heatmap_noise + 1, 1e-3, None)
            # guard: if f can go negative, fractional/large powers misbehave — clip at 0
            f_lo = np.clip(f_lo, 0.0, None)
            f_hi = np.clip(f_hi, 0.0, None)
            c_lo = f_lo ** powers
            c_hi = f_hi ** powers
            # pooled across all channels -> ONE (low, high) for the whole block
            low  = float(np.minimum(c_lo, c_hi).min())
            high = float(np.maximum(c_lo, c_hi).max())
            return (low, high)
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

    def toString(self) -> str:
        # descriptive + unique: encodes the degree N and recurses into the wrapped map
        # (e.g. TargetWrap -> OptimalActionTarget), so different targets/degrees read distinctly
        return f"MultiPolyN{self.N}_{describeMap(self.inner)}"
    def mapAll(self, cordArr:NDArray[np.int_]) -> NDArray[np.float32]:
        f_input = (self.inner.map(cordArr)*self.heatmap_weights)+self.heatmap_noise+1
        f_input = np.clip(f_input, 1e-3, None)
        powers = np.arange(self.N)+1
        return (f_input**powers)
    def mapOne(self, cordArr:NDArray[np.int_], n) -> np.float32:
        f_input = (self.inner.map(cordArr)*self.heatmap_weights)+self.heatmap_noise+1
        f_input = np.clip(f_input, 1e-3, None)
        powers = np.arange(self.N)+1
        return (f_input**powers)[n]
    def getAmount(self) -> np.int_:
        return self.N