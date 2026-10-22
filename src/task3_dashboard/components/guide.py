"""Hướng dẫn sử dụng & cách đọc kết quả: chú thích ngắn dưới từng khối + trang hướng dẫn đầy đủ (tab riêng)."""
from __future__ import annotations

import streamlit as st

from .theme import C


# ----------------------------------------------------------------------------- chú thích ngắn dưới từng khối
def note_html(key: str, warn: float = 0.30, caution: float = 0.15, h: int = 10) -> str:
    text = {
        "cards": "<b>Cách đọc:</b> Spread càng nhỏ càng dễ mua bán. OBI và MLOFI <b>dương (xanh)</b> = bên mua "
                 "đang mạnh hơn, <b>âm (đỏ)</b> = bên bán mạnh hơn; càng xa 0 thì lệch càng rõ.",
        "ladder": "<b>Cách đọc:</b> Đỏ phía trên = lệnh chờ BÁN, xanh phía dưới = lệnh chờ MUA. Thanh càng dài thì "
                  "tầng đó càng nhiều lệnh. Một phía dày hẳn lên thường đẩy giá về phía bên kia.",
        "depth": "<b>Cách đọc:</b> Đường xanh (mua) và đỏ (bán) cộng dồn khối lượng từ giá giữa ra ngoài. "
                 "Đường nào dốc lên nhanh hơn thì phía đó \"dày\" hơn, giá khó đi xuyên qua hơn.",
        "gauge": f"<b>Cách đọc:</b> Xác suất {h} giây tới việc mua/bán nhanh sẽ <b>đắt bất thường</b>. "
                 f"Bình thường chỉ khoảng 5%. <b style='color:{C['good']}'>Xanh</b> &lt; {caution:.0%} · "
                 f"<b style='color:{C['warning']}'>Vàng</b> {caution:.0%}–{warn:.0%} · "
                 f"<b style='color:{C['critical']}'>Đỏ</b> &gt; {warn:.0%}. Đỏ ≠ chắc chắn xảy ra, mà là "
                 f"nguy cơ cao gấp nhiều lần bình thường.",
        "probs": f"<b>Cách đọc:</b> 3 thanh cộng lại = 100%. Đoán bừa ≈ 33% mỗi thanh, nên thanh nào vượt hẳn "
                 f"~45–50% mới là tín hiệu rõ. Đây là dự báo cho đúng {h} giây tới, không phải xu hướng dài.",
        "slippage": "<b>Cách đọc:</b> Số bps bị thiệt nếu mua/bán ngay một lệnh cỡ vừa thay vì khớp đúng giá giữa "
                    "(1 bps = 0,01%). \"Cao hơn X%\" = hiện tại đắt hơn X% số giây gần đây; ≥ 95% là rất đắt.",
        "mid": "<b>Cách đọc:</b> Giá giữa (trung bình giá mua tốt nhất và giá bán tốt nhất) trong cửa sổ đang xem.",
        "heat": "<b>Cách đọc:</b> Trục ngang là thời gian, trục dọc là 20 tầng bán (trên) và 20 tầng mua (dưới). "
                "Màu sáng = nhiều lệnh chờ. Vùng <b>tối dần</b> = sổ lệnh mỏng đi → dễ trượt giá, hay đi trước "
                "lúc đồng hồ chuyển vàng/đỏ.",
        "returns": f"<b>Cách đọc:</b> Đường xanh = mô hình kỳ vọng giá đi bao nhiêu trong {h} giây; vùng cam = giá "
                   f"thực tế đã đi. Hai đường cùng phía (cùng trên hoặc cùng dưới 0) = mô hình đoán đúng hướng. "
                   f"Đường xanh thường nhỏ hơn nhiều vì đó là mức trung bình, không phải mức tối đa.",
        "stress_tl": "<b>Cách đọc:</b> Đường nhạt = từng giây, đường đậm = trung bình 10 giây. Chú ý khi đường đậm "
                     "<b>leo lên và nằm lâu</b> trong vùng vàng/đỏ, hơn là một cái gai đơn lẻ.",
        "spread": f"<b>Cách đọc:</b> Spread niêm yết BTC/ETH gần như luôn 1 tick, nên ô giữa cho biết khả năng nó "
                  f"<b>giãn ra</b> trong {h} giây tới (bình thường dưới 1%; hãy nhìn màu thay vì con số). Ô phải là <b>spread hiệu dụng</b> – "
                  f"chi phí thật nếu mua rồi bán ngay một lệnh cỡ vừa. Màu: xanh &lt; 2×, vàng 2–5×, đỏ ≥ 5× mức nền.",
        "spread_chart": f"<b>Cách đọc:</b> Đường xanh = spread hiệu dụng mô hình dự báo cho {h} giây tới; vùng cam = "
                        f"thực tế {h} giây sau. Hai đường càng bám nhau càng tốt. Tam giác vàng = lúc spread niêm yết "
                        f"thật sự giãn hơn 1 tick.",
        "hits": f"<b>Cách đọc:</b> Mỗi chấm là 1 giây: <b style='color:{C['bid']}'>xanh = đoán đúng hướng</b>, "
                f"<b style='color:{C['ask']}'>đỏ = sai</b> (chỉ có khi đã qua đủ {h} giây). Đường tím = tỉ lệ đúng "
                f"1 phút gần nhất; nằm trên vạch 33% (đoán bừa) là mô hình đang có ích.",
        "leaderboard": "<b>Cách đọc:</b> Mọi mô hình cùng đoán trên cùng dữ liệu, cùng lúc. Xếp theo <b>tỉ lệ đoán "
                       "đúng</b> (đoán bừa ≈ 33%); macro-F1 là điểm phụ của báo cáo. "
                       f"<b style='color:{C['good']}'>Xanh</b> = ngang mô hình đứng đầu (chênh nhỏ hơn sai số) · "
                       f"<b style='color:{C['warning']}'>Vàng</b> = kém hơn rõ · "
                       f"<b style='color:{C['critical']}'>Đỏ</b> = không hơn đoán bừa. Vài phút thì thứ hạng đổi liên "
                       f"tục là bình thường – chỉ tin khi chấm trên nhiều giờ.",
        "lb_chart": "<b>Cách đọc:</b> Mỗi đường là tỉ lệ đoán đúng trong 5 phút gần nhất của 1 mô hình (đường đậm = "
                    "mô hình đang dùng trên dashboard). Các đường đan xen nhau = các mô hình ngang sức; vạch chấm = "
                    "đoán bừa 33%.",
    }[key]
    return f"<div class='note'>{text}</div>"


# ----------------------------------------------------------------------------- trang hướng dẫn đầy đủ
def _box(title: str, body_md: str):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(body_md)


def guide_page(warn: float, caution: float, notional: dict, h: int = 10):
    nb, ne = notional.get("BTCUSDT", 25000), notional.get("ETHUSDT", 10000)
    st.markdown(
        f"<div class='lsm-header'><div><div class='lsm-title'>Hướng dẫn sử dụng & cách đọc kết quả</div>"
        f"<div class='lsm-sub'>Đọc 2 phút là dùng được · bấm nút <b>📊 Dashboard</b> ở trên để quay về"
        f"</div></div></div>", unsafe_allow_html=True)

    st.markdown("### 1. Dashboard này để làm gì?")
    st.markdown(
        f"Nhìn vào **sổ lệnh** (danh sách người đang chờ mua và chờ bán) của BTC/USDT hoặc ETH/USDT trên Binance "
        f"để trả lời 2 câu hỏi cho **{h} giây tới**:\n"
        f"1. **Giá sẽ tăng, đứng yên hay giảm?** → 3 thanh xác suất.\n"
        f"2. **Có sắp \"kẹt thanh khoản\" không?** – tức là muốn mua/bán nhanh một lệnh cỡ vừa "
        f"({nb:,.0f} USDT với BTC, {ne:,.0f} USDT với ETH) sẽ bị thiệt giá nhiều bất thường → đồng hồ "
        f"**Liquidity Stress Meter**.\n\n"
        f"Đây là công cụ nghiên cứu/minh họa, **không phải lời khuyên đầu tư**.")

    st.markdown("### 2. Hướng dẫn sử dụng")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _box("Mở / tắt", (
            "- **Mở:** bấm đúp `src\\task3_dashboard\\CHAY_TASK3.bat` → trình duyệt tự mở trang.\n"
            "- **Tắt:** đóng cửa sổ màu đen (hoặc nhấn Ctrl+C trong đó).\n"
            "- Không cần train lại mô hình, không ảnh hưởng việc tự thu dữ liệu nền.\n"
            "- 3 nút đầu trang: **📊 Dashboard** (màn hình chính) · **🏆 So sánh mô hình** (bảng xếp hạng) · "
            "**📖 Hướng dẫn** (trang này)."))
        _box("Thanh điều khiển bên trái", (
            "- **Nguồn dữ liệu** – chọn 1 trong 3 chế độ (bảng bên phải).\n"
            "- **Cặp giao dịch** – BTC/USDT hoặc ETH/USDT.\n"
            "- **Tốc độ làm mới** – số liệu nhảy bao lâu một lần (mặc định 0,5 giây). Máy yếu thì chọn 1–2 giây.\n"
            "- **Cửa sổ biểu đồ** – các biểu đồ hiện 5, 10 hay 15 phút gần nhất.\n"
            "- **Hiện chú thích dưới từng khối** – tắt đi khi đã quen để màn hình gọn hơn."))
    with c2:
        _box("3 chế độ dữ liệu", (
            "| Chế độ | Là gì | Khi nào dùng |\n|---|---|---|\n"
            "| **Giả lập** | Số liệu và xác suất **giả**, có các đợt căng thẳng giả | Thử giao diện, không có mạng |\n"
            "| **Live** | Dữ liệu Binance thật + mô hình thật | Xem thị trường ngay lúc này |\n"
            "| **Replay** | Phát lại 1 ngày đã lưu + mô hình thật | Thuyết trình, xem lại đợt căng thẳng thật |\n\n"
            "**Live** cần khoảng **10 phút đầu** để gom dữ liệu (có thanh tiến trình). "
            "Trong lúc đó dự báo chưa ổn định – nên mở trước khi trình bày."))
        _box("Điều khiển Replay", (
            "- **Ngày phát lại:** ưu tiên ngày ghi **NGOÀI mẫu** (01/09 và 01/10/2026) – mô hình chưa từng thấy "
            "ngày này nên kết quả là trung thực. Ngày **trong mẫu** mô hình đã học, chỉ để minh họa.\n"
            "- **Bắt đầu từ (giờ UTC):** giờ Việt Nam = UTC + 7.\n"
            "- **Tốc độ phát:** ×1 = như thật, ×60 = 1 phút thật xem trong 1 giây.\n"
            "- **⏯** phát/dừng · **⏮** quay về giờ bắt đầu."))

    st.markdown("### 3. Cách đọc từng khu vực")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f"<div class='gtag' style='color:{C['series1']}'>CỘT TRÁI · SỔ LỆNH</div>",
                    unsafe_allow_html=True)
        _box("4 thẻ chỉ số", (
            "| Thẻ | Nghĩa là gì | Đọc thế nào |\n|---|---|---|\n"
            "| **Giá mid** | Giá giữa giá mua tốt nhất và giá bán tốt nhất | Dòng nhỏ = đã đổi bao nhiêu bps trong cửa sổ |\n"
            "| **Spread** | Khoảng cách giá bán tốt nhất – giá mua tốt nhất | Nhỏ = dễ mua bán. BTC/ETH gần như luôn 1 tick (rất nhỏ). Thẻ vàng/đỏ khi spread giãn gấp 1,3 / 2 lần bình thường |\n"
            "| **OBI 5 tầng** | Lệch khối lượng chờ mua so với chờ bán ở 5 tầng gần giá, từ −1 đến +1 | +0,5 = bên mua nhiều hơn hẳn → giá hơi dễ tăng; −0,5 ngược lại |\n"
            "| **MLOFI 10 giây** | Dòng lệnh **mới vào/rút ra** khỏi sổ trong 10 giây qua | Dương = lệnh mua đổ vào / lệnh bán rút đi (áp lực mua). Phản ánh cái đang **diễn ra**, còn OBI là **ảnh chụp** hiện tại |"))
        _box("Bậc thang 20 tầng / Depth curve", (
            "- **Đỏ trên** = lệnh chờ bán, **xanh dưới** = lệnh chờ mua, dải giữa ghi giá mid và spread.\n"
            "- Thanh đậm = khối lượng của tầng đó; nền nhạt = khối lượng cộng dồn tới tầng đó.\n"
            "- Độ dài thanh vẽ theo căn bậc hai để tầng nhỏ vẫn nhìn thấy – nên so **tương đối**, đừng đo bằng thước.\n"
            "- Với BTC, cả 20 tầng chỉ trải trong khoảng ±1 bps quanh giá – sổ lệnh rất \"sát\"."))
        _box("Giá mid & LOB Heatmap", (
            "- **Giá mid:** đường giá trong cửa sổ.\n"
            "- **Heatmap:** mỗi hàng là 1 tầng (Ask 20 trên cùng → Bid 20 dưới cùng), mỗi cột là 1 thời điểm. "
            "Màu sáng = nhiều lệnh chờ.\n"
            "- **Dấu hiệu cần chú ý:** cả vùng tối dần = người ta rút lệnh, sổ mỏng → lệnh thị trường dễ trượt giá. "
            "Thường thấy trước hoặc cùng lúc đồng hồ stress tăng."))
        _box("Biểu đồ Spread hiệu dụng 10 giây tới", (
            f"- **Đường xanh** = mô hình dự báo chi phí mua + bán ngay một lệnh cỡ vừa trong {h} giây tới; "
            f"**vùng cam** = chi phí thật đo được {h} giây sau (chỉ vẽ phần đã qua).\n"
            "- Hai đường bám nhau = dự báo tốt. Mô hình thường **bắt được xu hướng lên xuống** nhưng "
            "**không bắt được các gai rất nhọn** (những gai đó gần như ngẫu nhiên).\n"
            "- **Tam giác vàng** = những giây spread niêm yết thật sự giãn hơn 1 tick."))
    with c2:
        st.markdown(f"<div class='gtag' style='color:{C['accent']}'>CỘT PHẢI · DỰ BÁO & RỦI RO</div>",
                    unsafe_allow_html=True)
        _box("Đồng hồ Liquidity Stress Meter", (
            f"- Con số = **xác suất** trong {h} giây tới chi phí mua/bán nhanh lọt vào nhóm **5% đắt nhất** "
            f"so với 1 giờ trước đó.\n"
            f"- Vì định nghĩa là \"top 5%\", nên **bình thường đồng hồ chỉ khoảng 5%** – đó là mức nền.\n"
            f"- **Xanh < {caution:.0%}** bình thường · **Vàng {caution:.0%}–{warn:.0%}** rủi ro tăng, theo dõi · "
            f"**Đỏ > {warn:.0%}** nguy cơ cao, tránh lệnh thị trường lớn.\n"
            f"- Hộp bên dưới cho biết đang **gấp mấy lần mức nền** (vd 6× = nguy cơ cao gấp 6 lần lúc bình thường).\n"
            f"- **StressRatio A–S:** chỉ số phụ theo mô hình Avellaneda–Stoikov – chi phí thực tế so với mức "
            f"\"hợp lý\" về lý thuyết; > 1,5 là bất thường. Chỉ để tham khảo, yếu hơn đồng hồ chính."))
        _box("3 thanh P(Tăng) / P(Đứng) / P(Giảm)", (
            f"- Xác suất giá mid sau {h} giây cao hơn / gần như bằng / thấp hơn bây giờ.\n"
            f"- \"Đứng\" với BTC = giá **không đổi chút nào**; với ETH = đổi rất ít (≤ 0,335 bps).\n"
            f"- **Đoán bừa = 33%** mỗi thanh. Thanh vượt ~45–50% mới là tín hiệu đáng kể; ba thanh xấp xỉ nhau "
            f"= mô hình \"không chắc\".\n"
            f"- **Lợi suất kỳ vọng** = mức giá trung bình mô hình dự kiến đi, thường chỉ vài phần mười bps."))
        _box("Live Slippage Alert (cảnh báo trượt giá)", (
            f"- **Mua ngay / Bán ngay:** mất bao nhiêu bps nếu khớp ngay lệnh {nb:,.0f} USDT (BTC) / "
            f"{ne:,.0f} USDT (ETH) so với giá mid. Ví dụ 0,5 bps của 25.000 USDT ≈ 1,25 USDT.\n"
            f"- **So với cửa sổ:** chi phí hiện tại đắt hơn bao nhiêu % số giây gần đây. ≥ 80% → vàng, ≥ 95% → đỏ.\n"
            f"- Màu của hộp lấy mức **nặng hơn** giữa dự báo của mô hình và chi phí tức thời.\n"
            f"- \"20 tầng không đủ khớp hết lệnh\" = sổ quá mỏng, chi phí thật còn cao hơn số hiển thị."))
        _box("Dự báo spread 10 giây tới (thêm mới – cột trái, dưới sổ lệnh)", (
            f"- **Spread hiện tại:** spread niêm yết tính bằng tick (bước giá nhỏ nhất, 0,01 USDT). BTC/ETH gần như "
            f"lúc nào cũng **1 tick**.\n"
            f"- **P(giãn > 1 tick):** khả năng trong {h} giây tới spread niêm yết rộng hơn 1 tick. Bình thường rất "
            f"thấp (dưới 1%); con số **gấp mấy lần mức nền** mới là thứ cần nhìn.\n"
            f"- **Độ tin cậy: vừa phải.** Trên 2 ngày test, lúc **đỏ** spread giãn thật ~3–4% số lần, lúc **xanh** "
            f"chỉ ~0,2–0,4% → màu xếp hạng rủi ro tốt. Nhưng con số % thường **cao hơn thực tế** vào ngày êm – "
            f"hãy đọc màu, đừng đọc con số.\n"
            f"- **Spread hiệu dụng:** chi phí thật nếu mua rồi bán ngay một lệnh {nb:,.0f} USDT (BTC) / "
            f"{ne:,.0f} USDT (ETH), dự báo trung bình cho {h} giây tới. Mũi tên ↑/↓ = so với 10 giây vừa qua.\n"
            f"- Màu: **xanh** < 2× mức nền · **vàng** 2–5× · **đỏ** ≥ 5×."))
        _box("Lợi suất 10 giây tới & Diễn biến P(stress)", (
            f"- **Lợi suất:** đường xanh = mô hình kỳ vọng, vùng cam = thực tế {h} giây sau (chỉ vẽ khi đã qua đủ "
            f"{h} giây – không nhìn trước). Cùng phía so với 0 = đoán đúng hướng.\n"
            f"- Dòng chữ bên dưới: **tỉ lệ đoán đúng hướng** trong cửa sổ. So với đoán bừa 33%.\n"
            f"- **Diễn biến P(stress):** đường đậm (trung bình 10 giây) leo lên và nằm lâu ở vùng vàng/đỏ mới đáng lo; "
            f"gai đơn lẻ 1–2 giây thường là nhiễu."))
        _box("Đúng / sai từng giây (thêm mới)", (
            f"- Dễ đọc hơn biểu đồ lợi suất: **mỗi chấm = 1 giây**, xanh = mô hình đoán đúng hướng {h} giây sau, "
            f"đỏ = sai.\n"
            f"- **Đường tím** = tỉ lệ đúng trong 1 phút gần nhất. Trên vạch **33%** (đoán bừa) = mô hình đang có ích; "
            f"thường dao động 40–60%.\n"
            f"- Có lúc rơi xuống dưới 33% vài chục giây là bình thường – nhìn cả cửa sổ, đừng nhìn 1 khoảnh khắc."))

    st.markdown("### 4. Trang 🏆 So sánh mô hình")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _box("Trang này làm gì?", (
            f"- **8 mô hình cùng đoán hướng giá {h} giây tới, trên cùng dữ liệu, cùng lúc**: XGBoost (đang dùng trên "
            "dashboard), Random Forest, Multinomial Logit, Logistic, DeepLOB (học sâu) và 3 mốc so sánh: CST (lý "
            "thuyết), OBI-rule (luật đơn giản), Majority (luôn đoán 1 đáp án).\n"
            "- Tất cả học trên **cùng 10 ngày**, giống hệt lúc chấm điểm trong báo cáo.\n"
            f"- Mỗi giây, mọi mô hình đưa ra dự báo; {h} giây sau có đáp án thì chấm. **Không mô hình nào được nhìn "
            "trước.**\n"
            "- Chạy được với **Live** và **Replay** (Giả lập không có mô hình thật). Replay lần đầu mở 1 ngày mất "
            "~1–3 phút để chạy cả ngày, sau đó lưu lại."))
        _box("Đọc bảng xếp hạng", (
            "| Cột | Nghĩa |\n|---|---|\n"
            "| **Tỉ lệ đoán đúng ± sai số** | Đúng hướng bao nhiêu % số giây (dùng để xếp hạng), kèm khoảng sai số "
            "do dữ liệu ít. Đoán bừa ≈ 33% |\n"
            "| **Macro-F1** | Điểm chính của báo cáo (chấm cả ngày). Trong vài phút, nếu giá hầu như không lúc nào "
            "\"đứng yên\" thì điểm này bị kéo thấp cho **mọi** mô hình – vì vậy không dùng để xếp hạng ở đây |\n"
            "| **Khi dám đoán Tăng/Giảm** | Trong những giây mô hình đoán có biến động, đúng bao nhiêu % |\n"
            "| **Đánh giá** | 🟢 ngang mô hình đứng đầu · 🟡 kém hơn rõ · 🔴 không hơn đoán bừa |\n\n"
            "Ô **Chấm trên** ở đầu trang: chọn chấm 5 / 15 / 30 / 60 phút gần nhất hoặc toàn bộ."))
    with c2:
        _box("Vì sao thứ hạng đổi liên tục? Có nên đổi mô hình không?", (
            "- Trên 2 ngày test của báo cáo, các mô hình nền **gần như hòa nhau** (chênh 0,001–0,01 điểm). "
            "Trong vài phút, chênh lệch đó nhỏ hơn nhiều so với **may rủi**, nên thứ hạng nhảy qua nhảy lại là "
            "chuyện bình thường.\n"
            "- Cột **± sai số** cho biết điều này: hai mô hình chênh nhau ít hơn sai số = **coi như ngang nhau**.\n"
            "- **Chỉ nên đổi mô hình dashboard** khi một mô hình khác thắng rõ trên **nhiều giờ** dữ liệu "
            "(việc B: chấm trên 15–20 giờ dữ liệu tự thu).\n"
            "- Những điều trang này cho thấy rõ: mô hình máy học **thắng các mốc đơn giản** (OBI-rule, CST, "
            "Majority); **DeepLOB phức tạp nhất nhưng không thắng** XGBoost."))
        _box("Biểu đồ theo thời gian", (
            "- Mỗi đường = tỉ lệ đúng trong 5 phút gần nhất của 1 mô hình; **đường đậm** = XGBoost (đang dùng).\n"
            "- Các đường **đan xen** = ngang sức. Đường nào luôn nằm dưới vạch 33% = không có ích.\n"
            "- Bấm tên mô hình ở chú giải để ẩn/hiện đường đó."))

    st.markdown("### 5. Kết quả này có ý nghĩa gì? Tin được đến đâu?")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _box("Mô hình giỏi tới mức nào (kiểm tra trên 2 ngày chưa từng thấy)", (
            "| | BTC | ETH | So sánh |\n|---|---|---|---|\n"
            "| Đoán đúng hướng giá 10 giây | **~54%** | **~49%** | Đoán bừa 33%; luật đơn giản theo OBI ~47% / 44% |\n"
            f"| Khi đồng hồ **đỏ** (> {warn:.0%}), stress xảy ra thật | **~24%** | **~21%** | Lúc bình thường chỉ ~5% → cao gấp **4–5 lần** |\n"
            f"| Thời gian đồng hồ đỏ | ~3% | ~5% | Cảnh báo hiếm, không \"kêu\" liên tục |\n"
            f"| Khi khối **dự báo spread** đỏ, spread giãn thật | ~3,9% | ~2,7% | Lúc xanh chỉ ~0,4% / 0,2% |\n\n"
            "→ Mô hình **tốt hơn rõ rệt so với đoán bừa và luật đơn giản**, nhưng **còn xa mới chắc chắn**: "
            "đồng hồ đỏ nghĩa là \"nguy cơ cao hơn hẳn\", không phải \"chắc chắn sẽ kẹt\"."))
        _box("Ý nghĩa kinh tế", (
            "- **Sổ lệnh có chứa thông tin về giá sắp tới:** bên nào đông lệnh chờ hơn thì giá hơi dễ đi về phía bên kia. "
            "Thông tin chủ yếu nằm ở tầng sát giá nhất.\n"
            "- **Căng thẳng thanh khoản làm tăng rủi ro, không làm tăng lãi:** lúc stress, giá dao động mạnh hơn "
            "(~12–19%) nhưng không nghiêng hẳn lên hay xuống.\n"
            "- **Khi sổ lệnh mỏng, cùng một lượng lệnh làm giá chạy xa hơn** (thấy rõ ở BTC).\n"
            "- **Không dùng để kiếm lời được:** lãi gộp chỉ ~0,5 bps/lệnh, trong khi phí sàn thường ~10 bps/chiều → "
            "trừ phí là lỗ. Giá trị thực tế là **biết lúc nào không nên đặt lệnh lớn** để tránh bị trượt giá."))
    with c2:
        _box("Những điều cần nhớ khi đọc", (
            "- **Thị trường càng biến động mạnh, mô hình càng kém** (BTC: ~54% lúc êm → ~43% lúc sôi động).\n"
            "- Chế độ **Giả lập**: mọi xác suất đều giả, chỉ để xem giao diện.\n"
            "- **Live 10 phút đầu** chưa đáng tin; **Replay ngày trong mẫu** trông \"đẹp\" hơn thực tế vì mô hình đã học ngày đó.\n"
            "- Mô hình học trên 12 ngày (mỗi tháng 1 ngày) – chưa phủ mọi kiểu thị trường.\n"
            "- Chỉ thấy 20 tầng gần giá → không đo được tác động của lệnh rất lớn (> ~100.000 USDT)."))
        _box("Ví dụ đọc nhanh 1 tình huống", (
            "> Đồng hồ **27% (vàng)**, hộp ghi **gấp 5,4× mức nền** · Slippage Alert **cao hơn 92%** · "
            "Heatmap vùng tầng bán tối dần · P(Tăng) 48%, P(Giảm) 22%.\n\n"
            "**Hiểu là:** sổ lệnh đang mỏng đi, nhất là phía bán; mua/bán nhanh lúc này đắt hơn hầu hết thời gian "
            "gần đây và có thể còn đắt hơn trong 10 giây tới. Giá hơi nghiêng về tăng nhưng không chắc. "
            "→ Nếu cần đặt lệnh lớn, nên chờ đồng hồ về xanh hoặc chia nhỏ lệnh."))
        _box("Từ ngữ hay gặp", (
            "- **bps** (basis point) = 0,01%. 1 bps của 25.000 USDT = 2,5 USDT.\n"
            "- **Tầng (level):** mỗi mức giá trong sổ lệnh; tầng 1 = giá tốt nhất.\n"
            "- **Thanh khoản:** mức dễ mua bán nhanh mà không bị thiệt giá.\n"
            "- **Trượt giá (slippage):** phần thiệt do lệnh phải \"ăn\" sang các tầng giá xấu hơn.\n"
            "- **Trong / ngoài mẫu:** ngày mô hình đã học / chưa từng thấy.\n"
            "- **UTC:** giờ quốc tế; giờ Việt Nam = UTC + 7."))
