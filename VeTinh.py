import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz

# Đường dẫn đến ảnh vệ tinh
image_path = 'vetinh.jpg'
# Đọc ảnh
image = cv2.imread(image_path)
# Chuyển ảnh sang không gian màu RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Lấy kích thước ảnh
h, w, c = image.shape
# Chuyển đổi ảnh thành mảng 2D
pixel_values = image.reshape((-1, 3))
# Đưa giá trị về kiểu float để thuận tiện cho tính toán
pixel_values = np.float32(pixel_values)

# Số cụm (tùy chỉnh dựa trên đối tượng muốn phân biệt)
k = 3

# Áp dụng phân cụm K-means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(pixel_values)
# Chuyển đổi nhãn thành ảnh phân cụm
segmented_image_kmeans = kmeans_labels.reshape((h, w))

# Áp dụng phân cụm Fuzzy C-Means (FCM)
fcm_centers, fcm_labels, _, _, _, _, _ = fuzz.cluster.cmeans(
    pixel_values.T, c=k, m=2, error=0.005, maxiter=1000, init=None)

# Chọn cụm có xác suất cao nhất cho mỗi pixel
fcm_labels = np.argmax(fcm_labels, axis=0)
segmented_image_fcm = fcm_labels.reshape((h, w))

# Hiển thị ảnh gốc và các ảnh phân cụm K-means và FCM
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.title("K-means Clustering")
plt.imshow(segmented_image_kmeans, cmap='viridis')

plt.subplot(1, 3, 3)
plt.title("FCM Clustering")
plt.imshow(segmented_image_fcm, cmap='viridis')

plt.show()
