import inspect
from functools import partial

import numpy as np
import random
from numpy.typing import NDArray
from typing import Protocol, runtime_checkable
import numpy.typing as npt




def generate_heatmap_impl(obj, noise=0, blockSize=1): #returns a float16 heatmap #higher block size should mean less salient data
  #current implementation is probably slow
  shape = [int(obj.topRight[i] - obj.lowLeft[i] + 1) for i in range(obj.dimension)]
  width=shape[0]
  height=shape[1]


  heatmap = np.zeros(shape, dtype=np.float16)
  for idx in np.ndindex(*shape):
    coord = np.array([obj.lowLeft[i] + idx[i] for i in range(obj.dimension)], dtype=int)
    heatmap[idx] = obj.map(coord)

  heatmap_std = np.std(heatmap)

  #blocking
  if blockSize > 1:
    #0=bottom-left, 1=bottom-right, 2=top-left, 3=top-right
    corner = random.randint(0, 3)
    if corner == 0:
        x_step = 1
        y_step = 1
        x_start = 0
        y_start = 0
    elif corner == 1:
        x_step = -1
        y_step = 1
        x_start = width - 1
        y_start = 0
    elif corner == 2:
        x_step = 1
        y_step = -1
        x_start = 0
        y_start = height - 1
    else:
        x_step = -1
        y_step = -1
        x_start = width - 1
        y_start = height - 1

    blocked = np.zeros_like(heatmap)

    y = y_start
    while 0 <= y < height:
        x = x_start
        while 0 <= x < width:
            if x_step == 1:
                x_end = min(x + blockSize, width)
            else:
                x_end = x + 1
                x_start_block = max(x - blockSize + 1, 0)
            if y_step == 1:
                y_end = min(y + blockSize, height)
            else:
                y_end = y + 1
                y_start_block = max(y - blockSize + 1, 0)

            if x_step == 1:
                x_slice = slice(x, x_end)
            else:
                x_slice = slice(x_start_block, x_end) # type: ignore
            if y_step == 1:
                y_slice = slice(y, y_end)
            else:
                y_slice = slice(y_start_block, y_end) # type: ignore

            block_mean = np.mean(heatmap[x_slice, y_slice])+random.uniform(-noise*heatmap_std, noise*heatmap_std) #adds noise
            blocked[x_slice, y_slice] = block_mean

            if x_step == 1:
                x += blockSize
            else:
                x -= blockSize

        if y_step == 1:
            y += blockSize
        else:
            y -= blockSize

    heatmap = blocked
  else:
      #add noise
      for idx in np.ndindex(*shape):
        heatmap[idx]+=random.uniform(-noise*heatmap_std, noise*heatmap_std)
  return heatmap

def describeMap(m) -> str:
    """Call an instance's toString(), robust to the two conventions used across this
    file: a normal bound method (def toString(self)/(this)) and the legacy static form
    (def toString() with no self, valid to call on the CLASS but not an instance).
    Every wrapper's toString() should call this on its inner map(s) instead of calling
    inner.toString() directly, which crashes (TypeError) against the static form."""
    ts = getattr(m, "toString", None)
    if ts is None:
        return type(m).__name__
    try:
        return ts()
    except TypeError:
        return type(m).toString()

@runtime_checkable
class HeatMappable(Protocol):
    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        ...
    def getRange(self) -> tuple[float, float]:
        ...
    def generate_heatmap(self, noise=0, blockSize=1) -> np.ndarray:
        ...  
    def toString(this) -> str:
        ...

class DistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        min_dist = float('inf')
        x, y = int(cordArr[0]), int(cordArr[1])
        for tx, ty in self.targetCords:
            dx, dy = int(tx) - x, int(ty) - y
            d = (dx*dx + dy*dy) ** 0.5
            if d < min_dist:
                min_dist = d
        return min_dist
    def toString() -> str:
        return "L_2Distance"

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        max_dists = [np.min(np.linalg.norm(self.targetCords - c, axis=1)) for c in corners]
        return (0.0, float(np.max(max_dists)))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

class DijkstraDistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_], walls: np.ndarray):
        self.targetCords = np.array(targetCords)
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.walls = walls
        self.dimension = targetCords.shape[-1]
        self.num_cols = int(topRight[0] - lowLeft[0] + 1)
        self.num_rows = int(topRight[1] - lowLeft[1] + 1)

        # precompute shortest path distance from every cell to its nearest target
        self.dist_grid = self._precompute()

    def _precompute(self):
        import heapq

        INF = float('inf')
        dist = np.full((self.num_cols, self.num_rows), INF)

        # multi-source Dijkstra — seed all targets with distance 0
        heap = []
        for tx, ty in self.targetCords:
            xi = int(tx - self.lowLeft[0])
            yi = int(ty - self.lowLeft[1])
            if 0 <= xi < self.num_cols and 0 <= yi < self.num_rows:
                dist[xi, yi] = 0.0
                heapq.heappush(heap, (0.0, xi, yi))

        while heap:
            d, xi, yi = heapq.heappop(heap)

            if d > dist[xi, yi]:
                continue  # stale entry

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nxi, nyi = xi + dx, yi + dy
                if 0 <= nxi < self.num_cols and 0 <= nyi < self.num_rows:
                    if not self.walls[nxi, nyi]:
                        nd = d + 1.0
                        if nd < dist[nxi, nyi]:
                            dist[nxi, nyi] = nd
                            heapq.heappush(heap, (nd, nxi, nyi))

        return dist
    

    def map(self, cordArr: NDArray[np.int_]) -> float:
        xi = int(cordArr[0] - self.lowLeft[0])
        yi = int(cordArr[1] - self.lowLeft[1])
        if 0 <= xi < self.num_cols and 0 <= yi < self.num_rows:
            return float(self.dist_grid[xi, yi])
        return float('inf')

    def getRange(self) -> tuple[float, float]:
        finite = self.dist_grid[np.isfinite(self.dist_grid)]
        if len(finite) == 0:
            return (0.0, 0.0)
        return (0.0, float(np.max(finite)))

    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)
    
    def toString() -> str:
        return "DijkstraDistance"

class ManhattanDistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        min_dist = float('inf')
        x, y = int(cordArr[0]), int(cordArr[1])
        for tx, ty in self.targetCords:
            d = abs(int(tx) - x) + abs(int(ty) - y)
            if d < min_dist:
                min_dist = d
        return min_dist

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        max_dists = [np.min(np.sum(np.abs(self.targetCords - c), axis=1)) for c in corners]
        return (0.0, float(np.max(max_dists)))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)
    
    def toString() -> str:
        return "L_1Distance"

class LInftyDistanceTarget():
    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = targetCords
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        return np.min(np.amax(np.abs(self.targetCords - cordArr), axis=1))

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        max_dists = [np.min(np.amax(np.abs(self.targetCords - c), axis=1)) for c in corners]
        return (0.0, float(np.max(max_dists)))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)
    
    def toString() -> str:
        return "L_inftyDistance"

class OptimalActionTarget():
    """Returns the optimal action (0-3) for greedily reaching the nearest target.

    Action encoding matches SpaceEnv.step:
        0 = +Y (up), 1 = -Y (down), 2 = +X (right), 3 = -X (left).
    At each cell it picks the nearest target (L2) and returns the action whose
    one-step move most reduces the L2 distance to that target.

    To make it point at exactly one target instead of "nearest of all", wrap it
    with TargetWrap (which restricts targetCords to a single chosen index).
    """
    # one-step coordinate delta per action index (see SpaceEnv.step)
    _ACTION_DELTAS = ((0, 1), (0, -1), (1, 0), (-1, 0))

    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.targetCords = np.array(targetCords)
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = self.targetCords.shape[-1]

    def map(self, cordArr: NDArray[np.int_]) -> float:
        x, y = int(cordArr[0]), int(cordArr[1])
        # nearest target by squared L2
        diffs = self.targetCords - np.array([x, y])
        tx, ty = self.targetCords[int(np.argmin(np.sum(diffs * diffs, axis=1)))]
        # action whose resulting cell is closest to that target
        best_a, best_d = 0, float('inf')
        for a, (dx, dy) in enumerate(self._ACTION_DELTAS):
            ex, ey = int(tx) - (x + dx), int(ty) - (y + dy)
            d = ex * ex + ey * ey
            if d < best_d:
                best_d, best_a = d, a
        return float(best_a)

    def getRange(self) -> tuple[float, float]:
        # categorical action id in [0, 3]
        return (0.0, 3.0)

    @staticmethod
    def toString() -> str:
        return "OptimalAction"

    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

class LinearActionTarget():
    """Two-channel FLOAT encoding of the optimal action that requires the agent to
    LINEARLY COMBINE both channels to decode it (no trivial 4-value lookup).

    Each channel uses the SAME equation:   value = coeff * a + carrier
      - a       : optimal action toward the nearest target (0-3), as in OptimalActionTarget
      - carrier : continuous L2 distance to the nearest target (identical for both channels)
      - coeff   : 1 for component 0, 2 for component 1

    Because the carrier is continuous and shared, it cancels under subtraction:
        a == channel1 - channel0   (exact)
    yet neither channel alone reveals a (it's buried under an unknown continuous
    offset), so a single channel is NOT a trivial 4-valued observation. After
    getObs normalization the channels are scaled differently, but a fixed
    affine/linear combination still recovers a exactly (linear stays linear).

    Use two instances as two observation channels, e.g.:
        partial(LinearActionTarget, component=0)
        partial(LinearActionTarget, component=1)
    """
    _ACTION_DELTAS = ((0, 1), (0, -1), (1, 0), (-1, 0))
    _COEFFS = (1.0, 2.0)

    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_], component: int = 0, **kwargs):
        # **kwargs swallows extra env context (spawn, targetAwards) that buildMaps
        # passes to partials unfiltered — required since this is used via partial(component=...).
        self.targetCords = np.array(targetCords)
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = self.targetCords.shape[-1]
        self.component = int(component)

    def _nearest(self, x, y):
        diffs = self.targetCords - np.array([x, y])
        return self.targetCords[int(np.argmin(np.sum(diffs * diffs, axis=1)))]

    def map(self, cordArr: NDArray[np.int_]) -> float:
        x, y = int(cordArr[0]), int(cordArr[1])
        tx, ty = (int(v) for v in self._nearest(x, y))
        # optimal action: the move that most reduces L2 distance to that target
        best_a, best_d = 0, float('inf')
        for a, (dx, dy) in enumerate(self._ACTION_DELTAS):
            ex, ey = tx - (x + dx), ty - (y + dy)
            d = ex * ex + ey * ey
            if d < best_d:
                best_d, best_a = d, a
        carrier = ((tx - x) ** 2 + (ty - y) ** 2) ** 0.5  # continuous, shared across components
        return self._COEFFS[self.component] * best_a + carrier

    def getRange(self) -> tuple[float, float]:
        corners = np.array(np.meshgrid(*zip(self.lowLeft, self.topRight))).T.reshape(-1, self.dimension)
        maxDist = max(np.min(np.linalg.norm(self.targetCords - c, axis=1)) for c in corners)
        return (0.0, float(self._COEFFS[self.component] * 3 + maxDist))

    def toString(self) -> str:
        return "LinearAction" + str(self.component)

    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)
#USELESS because we now have polynomial heat map
class QuadraticActionTarget():
    """Two-channel FLOAT encoding where the optimal action is recoverable ONLY by a
    QUADRATIC combination of the channels — an affine decode provably cannot work.

    The continuous carrier enters one channel SQUARED:
        component 0:  h0 = carrier
        component 1:  h1 = a + carrier**2
    so the exact decode is   a = h1 - h0**2   (degree 2; needs the square term).
    No affine map alpha*h0 + beta*h1 + gamma equals a for all cells: to cancel the
    carrier you must subtract h0**2, which affine functions can't produce. Yet each
    channel alone is a continuous float that doesn't reveal a (contrast
    LinearActionTarget, whose decode a = h1 - h0 IS affine).

    carrier = L2 distance to the nearest target, normalized to [0,1] by a
    precomputed max distance so channel magnitudes stay small and float-precise.

    Use two instances (component=0, component=1) as two observation channels.
    """
    _ACTION_DELTAS = ((0, 1), (0, -1), (1, 0), (-1, 0))

    def __init__(self, targetCords: NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_], component: int = 0, **kwargs):
        self.targetCords = np.array(targetCords)
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.dimension = self.targetCords.shape[-1]
        self.component = int(component)
        corners = np.array(np.meshgrid(*zip(lowLeft, topRight))).T.reshape(-1, self.dimension)
        self._maxDist = float(max(np.min(np.linalg.norm(self.targetCords - c, axis=1)) for c in corners)) or 1.0

    def _nearest(self, x, y):
        diffs = self.targetCords - np.array([x, y])
        return self.targetCords[int(np.argmin(np.sum(diffs * diffs, axis=1)))]

    def map(self, cordArr: NDArray[np.int_]) -> float:
        x, y = int(cordArr[0]), int(cordArr[1])
        tx, ty = (int(v) for v in self._nearest(x, y))
        best_a, best_d = 0, float('inf')
        for a, (dx, dy) in enumerate(self._ACTION_DELTAS):
            ex, ey = tx - (x + dx), ty - (y + dy)
            d = ex * ex + ey * ey
            if d < best_d:
                best_d, best_a = d, a
        carrier = (((tx - x) ** 2 + (ty - y) ** 2) ** 0.5) / self._maxDist  # normalized [0,1]
        if self.component == 0:
            return float(carrier)
        return float(best_a + carrier * carrier)

    def getRange(self) -> tuple[float, float]:
        return (0.0, 1.0) if self.component == 0 else (0.0, 4.0)

    def toString(self) -> str:
        return "QuadraticAction" + str(self.component)

    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

class PolynomialWrap(HeatMappable):
    """Two-channel WRAPPER that hides an inner heatmap's value behind a configurable
    bivariate polynomial, so the agent must apply that polynomial to two channels to
    recover the value.

    Wraps:
      inner_map_type   : heatmap whose value  v = inner.map(coord)  gets encoded
                         (e.g. OptimalActionTarget to encode the action)
      carrier_map_type : heatmap whose normalized value is the hiding carrier
                         (default DistanceTarget — smooth and always available)

    Decode:    v = A(h0) * h1 + B(h0)
    Channels:  h0 = carrier_norm in [0,1]
               h1 = (v - B(carrier_norm)) / A(carrier_norm)

    a_coeffs / b_coeffs are A / B as constant-first coefficient lists. The decode is
    linear in h1 (generation is one division — no root solving) and an ARBITRARY
    polynomial in h0 (any degree; a cross term h0*h1 whenever A is non-constant).

    Examples (a_coeffs, b_coeffs):
        ([1],    [0, -1])     -> v = h1 - h0       (affine)
        ([1],    [0, 0, -1])  -> v = h1 - h0**2    (quadratic)
        ([1, 1], [0])         -> v = (1 + h0)*h1   (cross term h0*h1)
        ([1],    [0,0,0,-1])  -> v = h1 - h0**3    (cubic)

    Constraints: A(carrier) must be nonzero on [0,1] (it's a denominator) — keep a
    nonzero constant term in a_coeffs (enforced in __init__); and the decode is
    degree-1 in h1 (the efficient frontier — see note in chat).

    Use two instances (component=0, component=1) with identical config.
    """
    def __init__(self, inner_map_type, component: int = 0,
                 a_coeffs=(1.0,), b_coeffs=(0.0, -1.0),
                 carrier_map_type=DistanceTarget, **kwargs):
        self.component = int(component)
        self.a_coeffs = tuple(float(c) for c in a_coeffs)
        self.b_coeffs = tuple(float(c) for c in b_coeffs)
        self.inner = self._build(inner_map_type, kwargs)        # value to encode
        self.carrier = self._build(carrier_map_type, kwargs)    # hiding signal

        clow, chigh = self.carrier.getRange()
        self._clow = float(clow)
        self._cspan = float(chigh - clow) or 1.0

        # A(carrier) is a denominator -> must never vanish on carrier in [0,1].
        grid = np.linspace(0.0, 1.0, 128)
        if np.min(np.abs([self._poly(self.a_coeffs, g) for g in grid])) < 1e-6:
            raise ValueError("PolynomialWrap: A(carrier) hits ~0 on [0,1]; "
                             "give a_coeffs a nonzero constant term.")
        # Precompute component-1 value range (over the inner map's range) for getObs.
        vlow, vhigh = self.inner.getRange()
        vs = np.linspace(float(vlow), float(vhigh), 32)
        h1 = [(v - self._poly(self.b_coeffs, g)) / self._poly(self.a_coeffs, g)
              for v in vs for g in grid]
        self._h1_range = (float(min(h1)), float(max(h1)))

    @staticmethod
    def _build(map_type, kwargs):
        # Build an inner/carrier map from env context, filtering kwargs to what it
        # accepts (partials supported) — same pattern as DirectionWrap/TargetWrap.
        sig = inspect.signature(map_type)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return map_type(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return map_type(**filtered)

    @staticmethod
    def _poly(coeffs, x):
        return sum(c * (x ** k) for k, c in enumerate(coeffs))

    def _carrier_norm(self, cordArr):
        c = (self.carrier.map(cordArr) - self._clow) / self._cspan
        return min(1.0, max(0.0, c))  # clip into the guarded [0,1] (decode still exact)

    def map(self, cordArr: NDArray[np.int_]) -> float:
        c = self._carrier_norm(cordArr)
        if self.component == 0:
            return float(c)
        v = self.inner.map(cordArr)
        return float((v - self._poly(self.b_coeffs, c)) / self._poly(self.a_coeffs, c))

    def getRange(self) -> tuple[float, float]:
        return (0.0, 1.0) if self.component == 0 else self._h1_range

    def toString(self) -> str:
        # include the CARRIER too: two configs that differ only in carrier_map_type
        # must not collide on the same label (it's part of what's actually encoded).
        return (f"Poly{self.component}_{describeMap(self.inner)}"
                f"_carrier{describeMap(self.carrier)}")

    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

class DistanceFromMiddle():
    def __init__(self,targetCords:NDArray[np.int_], lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        if not (lowLeft.size == topRight.size):
            raise Exception("Wrong size inputs")
        if np.any(lowLeft >= topRight):
            raise Exception("Not proper corner points")
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.midPoint = lowLeft + (topRight - lowLeft) / 2
        self.dimension = lowLeft.size

    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        diffs = self.midPoint - cordArr
        return float(np.sqrt((diffs**2).sum()))

    def getRange(self) -> tuple[float, float]:
        return (0.0, float(np.linalg.norm(self.topRight - self.midPoint)))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)
    def toString() -> str:
        return "L_2DistanceFromMid"

class ManhattanDistanceFromMiddle():
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        if not (lowLeft.size == topRight.size):
            raise Exception("Wrong size inputs")
        if np.any(lowLeft >= topRight):
            raise Exception("Not proper corner points")
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.midPoint = lowLeft + (topRight - lowLeft) / 2
        self.dimension = lowLeft.size

    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        return int(np.abs(self.midPoint - cordArr).sum())



    def getRange(self) -> tuple[float, float]:
        return (0.0, float(np.sum(np.abs(self.topRight - self.midPoint))))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)
    
    def toString() -> str:
        return "L_1DistanceFromMiddle"

class LInftyDistanceFromMiddle():
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        if not (lowLeft.size == topRight.size):
            raise Exception("Wrong size inputs")
        if np.any(lowLeft >= topRight):
            raise Exception("Not proper corner points")
        self.lowLeft = lowLeft
        self.topRight = topRight
        self.midPoint = lowLeft + (topRight - lowLeft) / 2
        self.dimension = lowLeft.size

    def map(self, cordArr: NDArray[np.int_]) -> np.float16:
        if not (cordArr.size == self.dimension):
            raise Exception("Wrong array size")
        return np.amax(np.abs(cordArr - self.midPoint))

    def getRange(self) -> tuple[float, float]:
        return (0.0, float(np.amax(np.abs(self.topRight - self.midPoint))))
    
    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)
    
    def toString() -> str:
        return "L_InftyDistanceFromMiddle"
    
class DirectionWrap(HeatMappable):
    def __init__(self, inner_map_type, offset, **kwargs):
         # Filter kwargs to only what inner_map_type accepts.
         # Use signature(inner_map_type) (not .__init__) so partials work too.
        inner_sig = inspect.signature(inner_map_type)
        if any(p.kind == p.VAR_KEYWORD for p in inner_sig.parameters.values()):
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in inner_sig.parameters}
        self.inner = inner_map_type(**filtered)
        self.offset = np.array(offset)
    def map(self, coords):
        return self.inner.map(coords + self.offset)
    def getRange(self):
        return self.inner.getRange()

    def toString(this) -> str:
        return describeMap(this.inner) + "WithOffsetOf(" + str(this.offset[0]) + "," + str(this.offset[1]) + ")"
class NoiseWrap(HeatMappable):
    def __init__(self, inner_map_type=None, noiseLevel: float = 0.05, targetMap=None, **kwargs):
        """
        Wrap a heatmap with additive Gaussian noise.

        Two construction styles are supported:
          - new (used by SpaceEnv): inner_map_type is a HeatMap class/partial and
            the inner map is built from filtered env kwargs (mirrors DirectionWrap).
          - legacy: targetMap is an already-built HeatMap instance.
        """
        source = targetMap if targetMap is not None else inner_map_type
        if source is None:
            raise TypeError("NoiseWrap requires inner_map_type or targetMap")

        if isinstance(source, (type, partial)):
            # Build the inner map, filtering kwargs to what it accepts (partials supported).
            inner_sig = inspect.signature(source)
            if any(p.kind == p.VAR_KEYWORD for p in inner_sig.parameters.values()):
                filtered = kwargs
            else:
                filtered = {k: v for k, v in kwargs.items() if k in inner_sig.parameters}
            self.inner = source(**filtered)
        else:
            # Already an instance.
            self.inner = source
        self.noise_level = noiseLevel
        self.hash_table = dict()

    def map(self, cordArr: NDArray[np.int_]) -> float:
        # 1. Get the "clean" value from the original map
        clean_value = self.inner.map(cordArr)
        key = tuple(cordArr)

        #check if value has alread been computed
        if key not in self.hash_table:
            low, high = self.getRange()
            scale = (high - low) * self.noise_level
            noise = np.random.normal(0, scale)
            self.hash_table[key] = clean_value + noise

        return self.hash_table[key]

    def toString(this) -> str:
        return describeMap(this.inner) + "WithNoiseOf" + str(this.noise_level)

    def getRange(self) -> tuple[float, float]:
        # Return the range of the underlying map
        return self.inner.getRange()


class TargetWrap(HeatMappable):
    """Restricts an inner heatmap to a single target (by index).

    The env passes the full targetCords; this wrapper slices it down to one
    target before building the inner map, so e.g. OptimalActionTarget points at
    exactly that target instead of the nearest of all. Composes with the other
    wrappers (DirectionWrap, NoiseWrap) since it builds its inner the same way.
    """
    def __init__(self, inner_map_type, targetIndex=0, **kwargs):
        kwargs = dict(kwargs)
        if "targetCords" in kwargs and kwargs["targetCords"] is not None:
            tc = np.array(kwargs["targetCords"])
            self.targetIndex = int(targetIndex) % len(tc)
            kwargs["targetCords"] = tc[[self.targetIndex]]
        else:
            self.targetIndex = int(targetIndex)
        # Filter kwargs to what the inner accepts (partials supported), like DirectionWrap.
        inner_sig = inspect.signature(inner_map_type)
        if any(p.kind == p.VAR_KEYWORD for p in inner_sig.parameters.values()):
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in inner_sig.parameters}
        self.inner = inner_map_type(**filtered)

    def map(self, coords):
        return self.inner.map(coords)

    def getRange(self):
        return self.inner.getRange()

    def toString(this) -> str:
        return describeMap(this.inner) + "Target" + str(this.targetIndex)

#Heat maps used for testing and data collection

class xAscending(HeatMappable):
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.lowLeft = lowLeft
        self.topRight = topRight

    def map(self, cordArr: NDArray[np.int_]) -> float:
        return (cordArr-self.lowLeft)[0]


    def getRange(self) -> tuple[float, float]:
        return [0,(self.topRight-self.lowLeft)[0]]
    
    def toString(this) -> str:
        return "xAscending"



class yAscending(HeatMappable):
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.lowLeft = lowLeft
        self.topRight = topRight

    def map(self, cordArr: NDArray[np.int_]) -> float:
        return (cordArr-self.lowLeft)[1]


    def getRange(self) -> tuple[float, float]:
        return [0,(self.topRight-self.lowLeft)[1]]
    def toString(this) -> str:
        return "yAscending"


class xDescending(HeatMappable):
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.lowLeft = lowLeft
        self.topRight = topRight

    def map(self, cordArr: NDArray[np.int_]) -> float:
        return (self.topRight-cordArr)[0]


    def getRange(self) -> tuple[float, float]:
        return [0,(self.topRight-self.lowLeft)[0]]
    
    def toString(this) -> str:
        return "xDescending"



class yDescending(HeatMappable):
    def __init__(self, lowLeft: NDArray[np.int_], topRight: NDArray[np.int_]):
        self.lowLeft = lowLeft
        self.topRight = topRight

    def map(self, cordArr: NDArray[np.int_]) -> float:
        return (self.topRight-cordArr)[1]


    def getRange(self) -> tuple[float, float]:
        return [0,(self.topRight-self.lowLeft)[1]]
    
    def toString(this) -> str:
        return "yDescending"


class ZeroMap(HeatMappable):
    """Always reads 0. Used to pad an observation out to a fixed channel count when
    comparing configs whose real heatmaps produce different numbers of channels
    (e.g. different polynomial degrees) -- the padding channels carry no information."""
    def __init__(self, **kwargs):
        pass

    def map(self, cordArr: NDArray[np.int_]) -> float:
        return 0.0

    def getRange(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def generate_heatmap(self, noise=0, blockSize=1):
        return generate_heatmap_impl(self, noise, blockSize)

    def toString(this) -> str:
        return "Zero"

