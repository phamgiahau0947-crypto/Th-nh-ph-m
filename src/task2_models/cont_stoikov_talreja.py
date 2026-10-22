"""Cont, Stoikov & Talreja (2010) A stochastic model for order book dynamics (tài liệu/orderbook.pdf).

Hàng đợi ở giá mua/bán tốt nhất là 2 quá trình sinh–tử độc lập (Bổ đề 2):
    sinh (lệnh giới hạn tới)     tốc độ λ(1)
    tử   (lệnh thị trường + hủy) tốc độ μ + i·θ(1) khi hàng đợi có i lệnh
Với spread = 1 tick (gần như luôn đúng với BTC/ETH trên Binance), Mệnh đề 3 / pt. (11):
    P[mid tăng ở lần đổi giá kế tiếp | X_A(0)=a, X_B(0)=b] = P[σ_A < σ_B]
(hàng đợi bán cạn trước -> giá tăng; σ = thời điểm hàng đợi chạm 0).

Cách tính: bài báo dùng nghịch đảo Laplace của f̂_a(s)·f̂_b(−s). Ở đây tính TƯƠNG ĐƯƠNG trong miền thời gian
    P[σ_A < σ_B] = ∫ f_a(t)·S_b(t) dt      (f = mật độ, S = hàm sống của thời điểm chạm 0)
với S(t) lấy từ ma trận chuyển trạng thái exp(Q·Δt) của xích sinh–tử 1 chiều (ổn định số học, không cần
nghịch đảo Laplace). Phần khối lượng chưa chạm 0 sau T_max giây chia đôi (đối xứng).

Ước lượng tham số (mục 3.2) từ snapshot 1 giây (dữ liệu không có từng lệnh):
    λ  = KL lệnh thêm vào ở tầng 1 (cùng giá) / (T·S)
    μ  = KL lệnh thị trường / (T·S)                      (mỗi phía)
    θ  = KL hủy ở tầng 1 / (T·Q̄₁)                        (Q̄₁ = độ sâu tầng 1 trung bình; không phụ thuộc S)
với Δq + KL khớp = thêm − hủy  ->  thêm = max(·,0), hủy = max(−·,0) trong mỗi giây.
Đơn vị lệnh S: bài báo dùng KL trung bình của lệnh giới hạn – dữ liệu không có, nên S được chọn trên TRAIN
bằng hợp lý cực đại (log-loss nhỏ nhất) cho hướng đổi giá kế tiếp, trên lưới các bội số của độ sâu tầng 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import expm


@dataclass
class CSTModel:
    lam_vol: float          # KL thêm vào tầng 1 mỗi giây, mỗi phía
    mu_vol: float           # KL lệnh thị trường mỗi giây, mỗi phía
    theta: float            # tốc độ hủy trên mỗi đơn vị đang chờ (1/giây)
    unit: float             # đơn vị lệnh S
    n_max: int = 150
    t_max: float = 300.0
    dt: float = 0.1
    _table: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def lam(self) -> float:
        return self.lam_vol / self.unit

    @property
    def mu(self) -> float:
        return self.mu_vol / self.unit

    # ------------------------------------------------------------------ ước lượng
    @staticmethod
    def flow_stats(raw_days: list[pd.DataFrame]) -> dict:
        add_v = canc_v = mkt_v = T = q_sum = 0.0
        q_n = 0
        for d in raw_days:
            T += len(d)
            mkt_v += float(np.nansum(d["trade_qty"]))
            for p, q, hit in (("bids_p1", "bids_q1", "sell_qty"), ("asks_p1", "asks_q1", "buy_qty")):
                P, Q = d[p].to_numpy(float), d[q].to_numpy(float)
                same = np.isclose(P[1:], P[:-1], rtol=0, atol=1e-9)
                net = (Q[1:] - Q[:-1]) + d[hit].to_numpy(float)[1:]
                net = net[same & np.isfinite(net)]
                add_v += net[net > 0].sum()
                canc_v += -net[net < 0].sum()
                q_sum += np.nansum(Q)
                q_n += int(np.isfinite(Q).sum())
        return {"add_v": add_v, "canc_v": canc_v, "mkt_v": mkt_v, "T": T, "qbar": q_sum / max(q_n, 1)}

    @classmethod
    def estimate(cls, raw_days: list[pd.DataFrame], q_ask1, q_bid1, next_dir, n_max: int = 150,
                 unit_grid: tuple = (0.25, 0.5, 1, 2, 3, 5, 10, 20, 50), max_fit: int = 60000,
                 seed: int = 0) -> "CSTModel":
        """Ước lượng λ, μ, θ từ dòng lệnh; chọn đơn vị S trên TRAIN theo log-loss của hướng đổi giá kế tiếp.
        q_ask1, q_bid1, next_dir: mảng của các mẫu train (next_dir ∈ {+1, −1}, NaN = bỏ)."""
        st = cls.flow_stats(raw_days)
        T = st["T"]
        lam_v, mu_v, theta = st["add_v"] / 2 / T, st["mkt_v"] / 2 / T, st["canc_v"] / 2 / T / st["qbar"]
        qa, qb, y = (np.asarray(x, float) for x in (q_ask1, q_bid1, next_dir))
        ok = np.isfinite(qa) & np.isfinite(qb) & np.isfinite(y)
        idx = np.flatnonzero(ok)
        if len(idx) > max_fit:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_fit, replace=False))
        qa, qb, y = qa[idx], qb[idx], (y[idx] > 0).astype(float)
        med = float(np.median(np.concatenate([qa, qb])))
        best = None
        for g in unit_grid:
            m = cls(lam_vol=lam_v, mu_vol=mu_v, theta=theta, unit=med / g, n_max=n_max)
            p = np.clip(m.p_up(qa, qb), 1e-4, 1 - 1e-4)
            ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if best is None or ll < best[0]:
                best = (ll, m)
        model = best[1]
        model.fit_logloss = float(best[0])
        return model

    # ------------------------------------------------------------------ xác suất
    def _survival_path(self) -> np.ndarray:
        """S[k, a] = P(σ_a > k·dt), a = 1..N, cho xích sinh–tử 1 chiều (cắt cụt ở N, không sinh thêm)."""
        N = self.n_max
        i = np.arange(1, N + 1)
        birth = np.where(i < N, self.lam, 0.0)
        death = self.mu + i * self.theta
        Q = np.zeros((N, N))
        Q[i[:-1] - 1, i[:-1]] = birth[:-1]
        Q[i[1:] - 1, i[1:] - 2] = death[1:]
        Q[i - 1, i - 1] = -(birth + death)          # trạng thái 1 -> 0 là hấp thụ (rời khỏi lưới)
        E = expm(Q * self.dt)
        K = int(self.t_max / self.dt)
        S = np.empty((K + 1, N))
        v = np.ones(N)
        S[0] = v
        for k in range(1, K + 1):
            v = E @ v
            S[k] = v
        return np.clip(S, 0.0, 1.0)

    def table(self) -> np.ndarray:
        """P_up[a, b], a, b = 0..N (đơn vị lệnh). P(0, b) = 1, P(a, 0) = 0."""
        if self._table is not None:
            return self._table
        S = self._survival_path()
        f = S[:-1] - S[1:]                      # P(σ ∈ (k−1, k]·dt)
        Smid = (S[:-1] + S[1:]) / 2
        P = f.T @ Smid + 0.5 * np.outer(S[-1], S[-1])
        N = self.n_max
        t = np.zeros((N + 1, N + 1))
        t[0, 1:] = 1.0
        t[0, 0] = 0.5
        t[1:, 1:] = P
        self._table = t
        return t

    def p_up(self, q_ask1, q_bid1) -> np.ndarray:
        t = self.table()
        qa, qb = np.asarray(q_ask1, float), np.asarray(q_bid1, float)
        ok = np.isfinite(qa) & np.isfinite(qb)
        a = np.clip(np.rint(np.where(ok, qa, 0) / self.unit), 1, self.n_max).astype(int)
        b = np.clip(np.rint(np.where(ok, qb, 0) / self.unit), 1, self.n_max).astype(int)
        out = t[a, b].astype(float)
        out[~ok] = np.nan
        return out

    # ------------------------------------------------------------------ lưu / nạp
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_table", None)
        d["lam_units_per_s"], d["mu_units_per_s"] = self.lam, self.mu
        if hasattr(self, "fit_logloss"):
            d["fit_logloss"] = self.fit_logloss
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CSTModel":
        keys = ("lam_vol", "mu_vol", "theta", "unit", "n_max", "t_max", "dt")
        key = tuple((k, d[k]) for k in keys if k in d)
        if key not in _CACHE:
            _CACHE[key] = cls(**dict(key))
        return _CACHE[key]


_CACHE: dict = {}


def next_move_direction(mid: pd.Series) -> np.ndarray:
    """Hướng của lần đổi mid kế tiếp sau t (trong cùng đoạn dữ liệu): +1 / −1, NaN nếu không có."""
    m = mid.to_numpy(float)
    d = np.sign(np.diff(m))
    d = np.append(d, np.nan)              # d[t] = dấu (m[t+1] − m[t])
    out = np.full(len(m), np.nan)
    nxt = np.nan
    for t in range(len(m) - 1, -1, -1):   # quét ngược: hướng khác 0 gần nhất phía sau
        if np.isfinite(d[t]) and d[t] != 0:
            nxt = d[t]
        elif not np.isfinite(d[t]):
            nxt = np.nan                  # đứt dữ liệu -> không nối qua
        out[t] = nxt
    return out
