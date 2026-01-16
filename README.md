# Chess AI Grandmaster (Python Version)

Một trò chơi Cờ Vua thông minh chạy trên máy tính với giao diện đồ họa, được viết bằng Python.

## Tính năng
- **AI thông minh:** Sử dụng thuật toán Minimax, Cắt tỉa Alpha-Beta, và Học máy (Learning) để ngày càng đánh hay hơn.
- **Chế độ chơi đa dạng:**
    - Người vs Máy (Chọn độ khó Easy/Medium/Hard).
    - Chọn màu quân (Trắng/Đen/Ngẫu nhiên).
    - Đồng hồ thi đấu (1 phút, 3 phút, 10 phút...).
- **Công cụ hỗ trợ:**
    - Hoàn tác (Undo).
    - Phân tích ván đấu sau khi chơi (AI Review).
    - Giao diện đồ họa đẹp mắt (Pygame).

## Yêu cầu hệ thống
- Máy tính cài sẵn **Python 3.x**.
    - Tải tại: [python.org](https://www.python.org/downloads/)

## Cách chạy game (Đơn giản nhất)
Bạn không cần gõ lệnh phức tạp. Chỉ cần làm theo bước sau:

1.  Tìm file **`setup_game.bat`** trong thư mục dự án.
2.  Nhấp đúp chuột (Double-click) vào file đó.

Lúc này, một cửa sổ đen (CMD) sẽ hiện lên để:
- Tự động tạo môi trường ảo (Virtual Environment) để không ảnh hưởng đến máy của bạn.
- Tự động cài thư viện đồ họa `pygame`.
- Tự động mở Game lên.

## Cách chạy thủ công (Dành cho Dev)
Nếu bạn muốn chạy bằng dòng lệnh:

```bash
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Cài đặt thư viện
pip install pygame

# 4. Chạy game
python chess_ai.py
```

Chúc bạn chơi vui vẻ!
