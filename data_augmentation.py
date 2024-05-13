#%%
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Định nghĩa kích thước ảnh
img_height, img_width = 256, 256

# Định nghĩa kích thước batch
batch_size = 32

# Tạo một ImageDataGenerator với các tham số augmentation
datagen = ImageDataGenerator(
    rotation_range=45,  # Góc xoay từ -45 đến +45 độ
    width_shift_range=0.2,  # Dịch chuyển theo chiều rộng
    height_shift_range=0.2,  # Dịch chuyển theo chiều cao
    shear_range=0.2,  # Cắt biến dạng
    zoom_range=0.2,  # Phóng to hoặc thu nhỏ
    horizontal_flip=True,  # Lật ảnh ngang
    fill_mode='nearest'  # Chế độ điền pixel khi biên ảnh được đặt thành 'nearest'
)

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = 'G:/BrainTumor/Dataset/train'  # Đường dẫn đến thư mục chứa dữ liệu

# Tạo một thư mục mới để lưu các ảnh được xử lý
processed_dir = 'G:/BrainTumor/Dataset/processed_images'
os.makedirs(processed_dir, exist_ok=True)

# Tạo một generator từ thư mục chứa dữ liệu
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),  # Kích thước ảnh sau khi resize
    batch_size=batch_size,
    class_mode='categorical',  # Loại của mô hình, ở đây là categorical
    classes=['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']  # Danh sách các lớp (labels)
)

# Số batch muốn xử lý
num_batches_to_process = 100
batch_count = 0

# Lặp qua các batch và lưu các ảnh đã được xử lý vào thư mục mới
for i, (images, labels) in enumerate(train_generator):
    if batch_count >= num_batches_to_process:
        break  # Exit the loop if the desired number of batches have been processed
    for j in range(len(images)):
        # Tạo đường dẫn đến thư mục mới cho từng loại nhãn
        label = np.argmax(labels[j])  # Nhãn được chuyển về dạng số
        label_name = list(train_generator.class_indices.keys())[label]  # Tên của nhãn
        label_dir = os.path.join(processed_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)
        
        # Lưu ảnh vào thư mục mới
        image_path = os.path.join(label_dir, f'image_{i * batch_size + j}.jpg')
        cv2.imwrite(image_path, cv2.cvtColor(images[j], cv2.COLOR_RGB2BGR))  # Lưu ảnh dưới định dạng jpg
    
    batch_count += 1

# %%
