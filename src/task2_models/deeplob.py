"""DeepLOB (Zhang, Zohren & Roberts 2019) – mô hình nâng cao CNN + Inception + LSTM (đề bài 2.7, RQ3).

Thiết kế theo bài báo, điều chỉnh cho CPU & dữ liệu 1 giây:
  - Đầu vào: cửa sổ T = 50 giây × 40 đặc trưng = 10 tầng × [giá ask, KL ask, giá bid, KL bid].
    Giá biểu diễn bằng khoảng cách tới mid (bps) -> dừng (stationary) giữa các ngày; KL = log(1+q),
    chuẩn hóa z-score bằng thống kê của tập TRAIN.
  - Kiến trúc: 3 khối tích chập (1×2 gộp giá–KL, 1×2 gộp 2 phía, 1×10 gộp các tầng; mỗi khối thêm 2 lớp 4×1
    theo thời gian), module Inception (1×1→3×1, 1×1→5×1, maxpool→1×1), LSTM 64, softmax 3 lớp.
  - Nhãn & δ giống hệt mô hình bảng (δ chọn trên dev), h = 10s; trọng số lớp cân bằng.
  - Train trên 10 ngày dev (ngày dev cuối dùng làm validation cho early stopping), đánh giá MỘT LẦN trên 2 ngày
    test chưa đụng tới – cùng tập dòng với XGBoost/MNLogit để so sánh công bằng (đề bài: "phải chứng minh
    incremental performance trên untouched test period").
Không chạy walk-forward 5 fold cho DeepLOB vì chi phí CPU (ghi trong README_TASK_2.md).

    python -m src.task2_models.deeplob            # cần: pip install torch (bản CPU)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .common import FIG_DIR, MODEL_DIR, TABLE_DIR, ensure_dirs, get_logger, lob_days, load_config, read_day, \
    walk_forward_split
from .labeler import choose_delta, direction_label, forward_return_bps
from .metrics import block_bootstrap, block_ids, ci, direction_metrics
from .microstructure_math import regrid_1s

log = get_logger()
T_WIN, LEVELS = 50, 10


def lob_tensor(day_raw: pd.DataFrame) -> tuple[np.ndarray, pd.Index, np.ndarray]:
    """(n, 40) float32: với l = 1..10: [ask_dist_bps, log1p(ask_q), bid_dist_bps, log1p(bid_q)]; mid."""
    return lob_grid_tensor(regrid_1s(day_raw))


def lob_grid_tensor(g: pd.DataFrame) -> tuple[np.ndarray, pd.Index, np.ndarray]:
    """Như lob_tensor nhưng nhận sẵn lưới 1s (buffer live sau buffer_to_1s, hoặc 1 ngày đã regrid)."""
    mid = ((g["bids_p1"] + g["asks_p1"]) / 2).to_numpy()
    cols = []
    for l in range(1, LEVELS + 1):
        cols += [(g[f"asks_p{l}"].to_numpy() - mid) / mid * 1e4, np.log1p(g[f"asks_q{l}"].to_numpy()),
                 (g[f"bids_p{l}"].to_numpy() - mid) / mid * 1e4, np.log1p(g[f"bids_q{l}"].to_numpy())]
    return np.column_stack(cols).astype(np.float32), g.index, mid


def build_model():
    import torch
    from torch import nn

    class Inception(nn.Module):
        def __init__(self, c):
            super().__init__()
            act = nn.LeakyReLU(0.01)
            self.b1 = nn.Sequential(nn.Conv2d(c, 64, 1), act, nn.Conv2d(64, 64, (3, 1), padding=(1, 0)), act)
            self.b2 = nn.Sequential(nn.Conv2d(c, 64, 1), act, nn.Conv2d(64, 64, (5, 1), padding=(2, 0)), act)
            self.b3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), nn.Conv2d(c, 64, 1), act)

        def forward(self, x):
            return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

    def block(cin, k, s):
        act = nn.LeakyReLU(0.01)
        return nn.Sequential(nn.Conv2d(cin, 32, k, stride=s), act, nn.BatchNorm2d(32),
                             nn.Conv2d(32, 32, (4, 1), padding="same"), act, nn.BatchNorm2d(32),
                             nn.Conv2d(32, 32, (4, 1), padding="same"), act, nn.BatchNorm2d(32))

    class DeepLOB(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = block(1, (1, 2), (1, 2))      # 40 -> 20: gộp (giá, KL)
            self.c2 = block(32, (1, 2), (1, 2))     # 20 -> 10: gộp (ask, bid)
            self.c3 = block(32, (1, 10), (1, 1))    # 10 -> 1: gộp các tầng
            self.inc = Inception(32)
            self.lstm = nn.LSTM(192, 64, batch_first=True)
            self.fc = nn.Linear(64, 3)

        def forward(self, x):                      # x: (B, T, 40)
            z = self.c3(self.c2(self.c1(x.unsqueeze(1))))
            z = self.inc(z).squeeze(-1).permute(0, 2, 1)   # (B, T, 192)
            out, _ = self.lstm(z)
            return self.fc(out[:, -1])

    return DeepLOB()


def _windows(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.stack([X[i - T_WIN + 1: i + 1] for i in idx])


def run(symbols: list[str] | None = None, epochs: int = 5, stride: int = 4, batch: int = 256,
        threads: int | None = None) -> pd.DataFrame:
    import torch
    from torch import nn
    cfg = load_config()
    ensure_dirs()
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(cfg["random_state"])
    h = int(cfg["horizon_s"])
    rows, ci_rows = [], []
    for sym in symbols or cfg["symbols"]:
        days = lob_days(sym)
        _, dev, test = walk_forward_split([d for d, _ in days], cfg["split"]["n_folds"], cfg["split"]["test_days"])
        data = {}
        for d, f in days:
            X, ts, mid = lob_tensor(read_day(f))
            r = forward_return_bps(pd.Series(mid), h).to_numpy()
            data[d] = (X, ts.to_numpy(), r)
        delta = choose_delta(np.concatenate([data[d][2] for d in dev]), cfg["delta_flat_share"])
        Xtr_all = np.vstack([data[d][0] for d in dev[:-1]])
        mu, sd = np.nanmean(Xtr_all, 0), np.nanstd(Xtr_all, 0) + 1e-6

        def samples(day_list, step):
            Xs, ys, keys = [], [], []
            for d in day_list:
                X, ts, r = data[d]
                Xn = np.nan_to_num((X - mu) / sd)
                y = direction_label(r, delta)
                bad = ~np.isfinite(X).all(1)
                badc = np.convolve(bad.astype(int), np.ones(T_WIN, int))[: len(bad)] > 0   # cửa sổ có lỗ -> bỏ
                ok = np.flatnonzero(np.isfinite(y) & ~badc & (np.arange(len(y)) >= T_WIN - 1))
                ok = ok[::step]
                Xs.append((Xn, ok))
                ys.append(y[ok])
                keys.append(pd.DataFrame({"day": str(d), "timestamp": ts[ok]}))
            return Xs, np.concatenate(ys), pd.concat(keys, ignore_index=True)

        tr_X, tr_y, _ = samples(dev[:-1], stride)
        va_X, va_y, _ = samples(dev[-1:], stride * 2)
        te_X, te_y, te_keys = samples(test, 1)
        log.info(f"[DeepLOB] {sym}: δ={delta:.4f}bps | train {len(tr_y):,} cửa sổ, val {len(va_y):,}, "
                 f"test {len(te_y):,}")

        def batches(Xs, y, shuffle, rng=None):
            pairs = [(k, i) for k, (_, ok) in enumerate(Xs) for i in ok]
            order = rng.permutation(len(pairs)) if shuffle else np.arange(len(pairs))
            for s in range(0, len(order), batch):
                sel = order[s:s + batch]
                xb = np.stack([Xs[pairs[j][0]][0][pairs[j][1] - T_WIN + 1: pairs[j][1] + 1] for j in sel])
                yield torch.from_numpy(xb), torch.from_numpy((y[sel] + 1).astype(np.int64))

        model = build_model()
        n_params = sum(p.numel() for p in model.parameters())
        w = np.bincount((tr_y + 1).astype(int), minlength=3)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(len(tr_y) / (3 * np.maximum(w, 1)), dtype=torch.float32))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(cfg["random_state"])
        best, best_state, patience, t0 = np.inf, None, 0, time.time()
        for ep in range(epochs):
            model.train()
            tl, nb = 0.0, 0
            for xb, yb in batches(tr_X, tr_y, True, rng):   # xáo trộn THỨ TỰ mini-batch trong train là hợp lệ
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                tl += loss.item()
                nb += 1
            model.eval()
            with torch.no_grad():
                vl = np.mean([crit(model(xb), yb).item() for xb, yb in batches(va_X, va_y, False)])
            log.info(f"[DeepLOB] {sym} epoch {ep + 1}: train loss {tl / nb:.4f} | val loss {vl:.4f} "
                     f"| {time.time() - t0:.0f}s")
            if vl < best - 1e-4:
                best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 2:
                    break
        train_s = time.time() - t0
        model.load_state_dict(best_state)
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in batches(te_X, te_y, False):
                probs.append(torch.softmax(model(xb), 1).numpy())
        P = np.vstack(probs)
        xb1 = next(batches(te_X, te_y, False))[0][:1]
        lat = []
        with torch.no_grad():
            for _ in range(50):
                t1 = time.perf_counter()
                model(xb1)
                lat.append((time.perf_counter() - t1) * 1000)
        torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "delta_bps": delta, "T": T_WIN,
                    "levels": LEVELS}, MODEL_DIR / f"deeplob_{sym}.pt")
        met = direction_metrics(te_y, P)
        rows.append({"symbol": sym, "model": "DeepLOB", "split": "test", "n_params": n_params, "train_s": train_s,
                     "epochs_run": ep + 1, "latency_ms_p50": float(np.median(lat)), **met})
        # so sánh trên CÙNG các dòng test với mô hình bảng
        from .evaluate_walkforward import PRED_DIR
        pf = PRED_DIR / f"{sym}_preds.parquet"
        if pf.exists():
            pr = pd.read_parquet(pf)
            pr = pr[pr["split"] == "test"].copy()
            pr["day"] = pr["day"].astype(str)
            j = te_keys.assign(**{f"dl_{c}": P[:, k] for k, c in enumerate(("down", "flat", "up"))}, y=te_y)
            j = j.merge(pr, on=["day", "timestamp"], how="inner")
            j = j[np.isfinite(j["y_dir"])]
            cls = np.array([-1, 0, 1])
            from sklearn.metrics import f1_score
            f1 = lambda yy, pp: f1_score(yy, pp, average="macro", labels=cls, zero_division=0)
            blocks = block_ids(j["day"], j["timestamp"].to_numpy())
            yj = j["y_dir"].to_numpy().astype(int)
            pdl = cls[j[["dl_down", "dl_flat", "dl_up"]].to_numpy().argmax(1)]
            for other in ("XGBoost", "MNLogit", "RandomForest"):
                po = cls[j[[f"dir_{other}__{c}" for c in ("down", "flat", "up")]].to_numpy().argmax(1)]
                diff = block_bootstrap(lambda yy, a, b: f1(yy, a) - f1(yy, b), blocks, 200, 2, yj, pdl, po)
                ci_rows.append({"symbol": sym, "comparison": f"DeepLOB − {other}", "n": len(yj),
                                "macro_f1_deeplob": f1(yj, pdl), f"macro_f1_other": f1(yj, po),
                                "delta_macro_f1": f1(yj, pdl) - f1(yj, po), "ci_lo": ci(diff)[0],
                                "ci_hi": ci(diff)[1], "p_le_0": float((diff <= 0).mean())})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(TABLE_DIR / "deep_model_metrics.csv", index=False)
    if ci_rows:
        pd.DataFrame(ci_rows).round(4).to_csv(TABLE_DIR / "deep_vs_baseline_test.csv", index=False)
    log.info("Xong DeepLOB.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train/đánh giá DeepLOB (CPU)")
    ap.add_argument("--symbol", action="append")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.symbol, a.epochs, a.stride, threads=a.threads)


if __name__ == "__main__":
    main()
