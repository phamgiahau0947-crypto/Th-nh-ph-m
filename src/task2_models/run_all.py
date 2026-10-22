"""Chạy toàn bộ Task 2 theo thứ tự (mỗi bước chạy lại an toàn):

    python -m src.task2_models.run_all              # đầy đủ (không DeepLOB)
    python -m src.task2_models.run_all --deeplob    # + DeepLOB (cần torch, ~30–60 phút CPU)
    python -m src.task2_models.run_all --skip-eval  # chỉ chạy lại phân tích sau khi đã có mô hình

1. dataset              – đặc trưng + nhãn từ kho Task 1 (cache)
2. evaluate_walkforward – walk-forward 5 fold + test, chọn & lưu models/*.pkl, bảng/hình dự báo
3. mlofi_regression     – tái lập Xu et al. (2019), OLS vs Ridge, đồng thời vs dự báo
4. return_analysis      – hồi quy (13), stress vs lợi suất, Net PnL (14)
5. eda                  – bảng mô tả, LOB heatmap, OBI vs lợi suất, kiểm định mô hình cấu trúc
6. deeplob (tùy chọn)   – mô hình nâng cao CNN-LSTM trên test
7. spread_forecast      – (bổ sung) dự báo spread h giây tới
8. model_zoo            – (bổ sung) bộ mô hình cho bảng xếp hạng Dashboard (sau deeplob để có DeepLOB mới)
"""
from __future__ import annotations

import argparse
import time

from .common import get_logger

log = get_logger()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--deeplob", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    from . import dataset, eda, evaluate_walkforward, mlofi_regression, return_analysis
    dataset.build_cache()
    if not a.skip_eval:
        evaluate_walkforward.main([])
    mlofi_regression.run()
    return_analysis.run()
    eda.run()
    if a.deeplob:
        from . import deeplob
        deeplob.run()
    from . import model_zoo, spread_forecast
    spread_forecast.run()
    model_zoo.build()
    log.info(f"HOÀN TẤT Task 2 sau {(time.time() - t0) / 60:.1f} phút. Bảng: outputs/tables, hình: outputs/figures")


if __name__ == "__main__":
    main()
