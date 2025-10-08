# 🚦 Dự án Nhận diện phương tiện giao thông

### 🏷️ Tên nhóm  
**Nhóm 10**

### 📝 Tên dự án  
**Nhận diện phương tiện giao thông**

### 👥 Thành viên nhóm  
| 👤 Họ và tên 🧑‍🎓    | 🆔 Mã sinh viên 🧾 | 🐙 Tên GitHub 🔗   |
|---------------------|---------------------|---------------------|
| Nguyễn Mai Hoàng Anh | 23001824 | 23001824-HoangAnh |
| Nguyễn Thị Lan Anh   | 23001826 | lanAnhne29 |
| Bùi Xuân Chung       | 23001838 | 23001838-hub |
| Đào Ngọc Cường       | 23001841 | daongocuong |
| Nguyễn Thị Huyền Linh| 23001899 | 23001899-af |

---

### 🗒️ Tóm tắt  
Dự án **Nhận diện phương tiện giao thông bằng Object Detection** hướng tới xây dựng một hệ thống có khả năng tự động phát hiện và phân loại các loại phương tiện giao thông như ô tô, xe máy, xe tải, xe buýt… xuất hiện trong ảnh chụp hoặc video từ camera giám sát.  

Dự án lựa chọn **YOLOv8** – một kiến trúc phát hiện đối tượng tiên tiến, được tối ưu cho tốc độ xử lý và độ chính xác cao, phù hợp với các ứng dụng cần xử lý theo thời gian thực.  

Bộ dữ liệu huấn luyện và kiểm thử được lấy từ các nguồn công khai trên **Kaggle**, bao gồm hàng chục nghìn ảnh chụp tình huống giao thông thực tế, trong đó mỗi phương tiện đều được gắn nhãn loại và vị trí bằng **bounding box**. Quy trình triển khai gồm: tiền xử lý dữ liệu, chuẩn hóa theo định dạng YOLO, huấn luyện hoặc fine-tune mô hình YOLOv8, đánh giá hiệu năng bằng các chỉ số mAP, và cuối cùng kiểm thử hệ thống trên ảnh/video mới.  

Dự án không chỉ giúp nhóm hiểu rõ hơn quy trình phát triển một ứng dụng AI từ dữ liệu đến triển khai, mà còn có giá trị thực tiễn cao khi có thể áp dụng vào hệ thống giám sát giao thông thông minh: hỗ trợ đếm, phân loại phương tiện, theo dõi mật độ lưu thông, phát hiện vi phạm, hoặc điều tiết luồng giao thông. Đây là một **mini-project tiêu biểu trong nhập môn AI** vì thể hiện đầy đủ chuỗi công việc từ xử lý dữ liệu, sử dụng mô hình học sâu đến ứng dụng thực tế.

---

### 🎯 Bối cảnh  
Giao thông đô thị tại Việt Nam hiện nay đang đối mặt với nhiều thách thức lớn như mật độ phương tiện ngày càng tăng, ùn tắc nghiêm trọng tại các thành phố lớn, cũng như tai nạn giao thông diễn biến phức tạp.  

Hệ thống quản lý giao thông truyền thống chủ yếu dựa vào con người giám sát hoặc xử lý dữ liệu thủ công, dẫn đến hiệu quả không cao và khó đáp ứng trong bối cảnh đô thị hóa nhanh chóng.  

Động lực để nhóm chọn chủ đề này đến từ nhu cầu áp dụng **AI và học sâu** vào quản lý giao thông thông minh. Công nghệ xử lý ảnh hiện nay đã phát triển mạnh, mở ra cơ hội xây dựng hệ thống tự động có khả năng nhận diện, phân loại và theo dõi phương tiện với độ chính xác cao.  

Chủ đề này vừa mang ý nghĩa xã hội, vừa hấp dẫn về kỹ thuật, đồng thời có tiềm năng ứng dụng trong **thành phố thông minh**, giúp giảm ùn tắc, phát hiện vi phạm và nâng cao an toàn giao thông.

---

### 🚀 Kế hoạch  

Quy trình thực hiện dự án gồm các bước chính sau:

1. **Thu thập & tiền xử lý dữ liệu**  
   - Tìm kiếm và tổng hợp tập dữ liệu hình ảnh/video giao thông từ **Kaggle** và các nguồn công khai khác.  
   - Tiến hành gán nhãn thủ công/bán tự động cho các loại phương tiện (ô tô, xe máy, xe tải, xe buýt, xe đạp…).  
   - Chuẩn hóa dữ liệu theo định dạng YOLO, kết hợp **resize** và **augmentation** để tăng tính đa dạng.  

2. **Xây dựng & huấn luyện mô hình**  
   - Sử dụng các mô hình học sâu trong thị giác máy tính như **YOLOv8, Faster R-CNN, CNN**.  
   - Fine-tune mô hình, điều chỉnh siêu tham số và tối ưu hóa để đạt hiệu năng cao.  

3. **Đánh giá mô hình**  
   - Sử dụng tập dữ liệu kiểm thử và các chỉ số: **accuracy, precision, recall, F1-score, mAP, FPS**.  

4. **Triển khai demo**  
   - Tích hợp mô hình vào ứng dụng nhận diện từ video trực tiếp hoặc video đã ghi.  
   - Hiển thị kết quả trực quan bằng bounding box và nhãn lớp.  

👉 Kế hoạch này vừa đảm bảo tính **nghiên cứu**, vừa hướng đến giá trị **ứng dụng thực tế** trong giám sát và quản lý giao thông thông minh.

---

### 📚 Tài liệu tham khảo  

1. Kaggle Datasets: [Traffic Detection Project](https://www.kaggle.com/datasets/yusufberksardoan/traffic-detection-project)  
2. **TensorFlow, PyTorch** – Thư viện xây dựng & huấn luyện mô hình học sâu.  
3. **OpenCV** – Thư viện xử lý ảnh, tiền xử lý dữ liệu.  

---
