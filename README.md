# Music-Genres-Classification
## Introduction:
Trong thời đại số như hiện nay, các nền tảng ứng dụng âm nhạc phát triển đi kèm với chúng là vô số bài nhạc từ cả thế kỷ trước cho đến nay, vì vậy các công ty sử dụng phân loại âm nhạc để dễ dàng tìm kiếm cũng như có thể đưa ra các đề xuất cho khách hàng của họ (chẳng hạn nhưu spotify, soundcloud,…). Xác định thể loại âm nhạc là bước đầu tiên đi theo hướng đó. Vì vậy nhóm dùng các kỹ thuật ML và DL để trích xuất các thể loại âm nhạc từ các tệp âm thanh khác nhau.

## Input/Output:
Bộ dữ liệu GZTAN bao gồm 1000 file audio, mỗi file 30s. Tổng cộng có 10 lớp, tương ứng với 10 thể loại nhạc, mỗi thể loại sẽ bao gồm 100 bài. Tất cả các bài sẽ có định dạng .wav.

Ngoài ra, bộ dữ liệu trên kaggle còn có 2 file csv bao gồm 58 thông tin đã được trích xuất sẵn, ví dụ như tâm quang phổ và băng thông, sắc độ, hài hòa, nhịp độ. Một file csv được trích xuất thông tin từ toàn bộ 30s của file audio và trả về một dòng trong bảng. File csv còn lại được trích xuất từ mỗi 3s trong từng bài hát. Nói cách khác, đoạn nhạc 30s sẽ được tách ra làm 10 phần, mỗi phần kéo dài 3s. Sau đó, mỗi đoạn nhỏ này sẽ được trích xuất MFCC features và trả lại 1 dòng ở trong bảng. Từ đó có thể thấy rằng bảng features_3s sẽ gấp 10 lần bảng features_30s.

## EDA
- Waveform: Dạng sóng là biểu diễn trực quan của âm thanh theo thời gian trên trục x và biên độ trên trục y. Cho phép ta quét nhanh dữ liệu âm thanh và so sánh và đối chiếu trực quan thể loại nào có thể giống hơn các thể loại khác

- Spectrogram: Biểu đồ phổ là một cách trực quan để biểu thị độ lớn tín hiệu của tín hiệu theo thời gian ở các tần số khác nhau có trong một dạng sóng cụ thể, với trục x là tần số (Hz) và trục y là thời gian.

- Spectral Rolloff: là tần số mà dưới đó một tỷ lệ phần trăm xác định của tổng năng lượng quang phổ

- Chroma Feature: Nó là một công cụ mạnh mẽ để phân tích các tính năng âm nhạc có cao độ có thể được phân loại một cách có ý nghĩa và có cách điều chỉnh gần đúng với thang âm bình đẳng. Một thuộc tính chính của các tính năng sắc độ là chúng nắm bắt các đặc tính hài hòa và giai điệu của âm nhạc trong khi vẫn mạnh mẽ trước những thay đổi về âm sắc và thiết bị đo đạc.

- Zero Crossing Rate: được cho là xảy ra nếu các mẫu liên tiếp có các dấu đại số khác nhau. Tốc độ xảy ra giao nhau bằng 0 là một thước đo đơn giản về nội dung tần số của tín hiệu. Tốc độ vượt qua 0 là thước đo số lần trong một khoảng thời gian/khung nhất định mà biên độ của tín hiệu giọng nói đi qua một giá trị bằng không.

- MFCC: một cách để trích xuất các đặc trưng giọng nói thường được sử dụng trong các model nhận dạng giọng nói (Speech Recognition) hay phân loại giọng nói (Speech Classification). Đúng như tên gọi của nó, MFCC sẽ cho ra kết quả là các hệ số coefficients của cepstral từ Mel filter trên phổ lấy được từ các file âm thanh chứa giọng nói.

- Spectral Centroid: Nó cho biết vị trí của “tâm khối lượng” đối với âm thanh và được tính bằng giá trị trung bình có trọng số của các tần số có trong âm thanh. Hãy xem xét hai bài hát, một thuộc thể loại blues và một thuộc thể loại metal. Bây giờ, so với bài hát thể loại blues với cùng thời lượng, bài hát metal có nhiều tần số hơn về cuối. Vì vậy, trung tâm quang phổ cho bài hát blues sẽ nằm ở đâu đó gần giữa phổ của nó trong khi trọng tâm của một bài hát metal sẽ nằm ở phần cuối của nó.

## Model
### Convert audio to image
B1: Pre-process

Sử dụng thư viện OpenCV để biến đổi audio thành ảnh

B2: Build model

Nhóm sử dụng model CNN

Sử dụng Adam optimizer để train model.

Tất cả các lớp ẩn sử dụng RELU activation function và lớp đầu ra được sử dụng softmax function.

Loss được tính bằng sparse_categorical_crossentropy.

### Convert audio to features
Việc chuyển đổi dữ liệu âm thanh sang định dạng số hoặc vectơ sẽ xác định lượng thông tin quan trọng được giữ lại khi dạng âm thanh bị mất. Chẳng hạn, nếu một định dạng dữ liệu không thể biểu thị âm lượng và nhịp độ của một bài hát rock, thì ngay cả những mô hình máy học tốt nhất cũng khó có thể tìm hiểu thể loại và phân loại mẫu. Ở phần này, nhóm đã sử dụng librosa, một thư viện dùng để trích xuất MFCC từ dữ liệu âm thanh.

Dưới đây là chi tiết từng bước trong hệ thống phân loại:

B1: Xử lý dữ liệu Ở bước này, nhóm sẽ sử dụng thư việc Librosa để trích xuất đặc trưng từ bộ dữ liệu trên.

B2: Build model Dataset bây giờ bao gồm các đặc trưng MFCC của từng bài hát. Bởi đây chỉ là cách biểu diễn khác của tín hiệu âm thanh, bộ dữ liệu có thể sử dụng như time series. Vậy nên, nhóm quyết định sử dụng LSTM để làm mô hình phân loại cho bài toán này. Dùng LSTM được vì sau khi đưa audio sẽ extract ma trận mfcc, mfcc là đặc trưng của âm thanh theo thời gian, sẽ dự đoán lần lượt những chỉ số đấy. Bởi mục tiêu của bài toán là phân loại 10 lớp, tương đương với 10 thể loại nhạc, nhóm đã sử dụng “sparse categorical cross entropy” để làm loss, và sử dụng “Adam” làm optimizer.

Ngoài ra, nhóm còn thử cả model CNN để thử nghiệm.

### Using csv file
Nhóm sử dụng file csv có sẵn và train bằng model neural network đơn giản

## Comparison
| Model | Audio to image | Audio to feature | File csv |
|--------------|-------|------|-------|
| CNN | 0.42 | 0.89 | 0.94 |
| LSTM | ... | 0.83 | ... | ... |


LSTM		0.83	
Nhìn kết quả thấy khi xử lý audio về dạng feature sẽ đưa ra kết quả tốt hơn rất nhiều so với convert từ audio sang ảnh. Vì khi convert sang ảnh theo đặc trưng của spectrogram , khá là khó để nhận diện được thì về bản chất những hình ảnh cho ra khá là giống nhau.
