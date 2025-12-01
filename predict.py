"""
Script để dự đoán hình học từ một ảnh đơn lẻ
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import sys

# Thiết lập đường dẫn
MODEL_PATH = r"C:\xulianh\geometry_model.h5"

# Tên các lớp
CLASS_NAMES = ['circle', 'kite', 'parallelogram', 'rectangle', 
               'rhombus', 'square', 'trapezoid', 'triangle']

IMG_SIZE = 224

def predict_image(image_path):
    """
    Dự đoán hình học từ một ảnh
    
    Args:
        image_path: Đường dẫn đến file ảnh
        
    Returns:
        Tên lớp được dự đoán và xác suất
    """
    # Tải mô hình
    if not os.path.exists(MODEL_PATH):
        print(f"Không tìm thấy mô hình tại: {MODEL_PATH}")
        print("Vui lòng train mô hình trước bằng cách chạy: python train_model.py")
        return None
    
    model = keras.models.load_model(MODEL_PATH)
    
    # Kiểm tra file ảnh
    if not os.path.exists(image_path):
        print(f"Không tìm thấy file ảnh: {image_path}")
        return None
    
    # Load và preprocess ảnh
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Dự đoán
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Hiển thị kết quả
    print(f"\n{'='*50}")
    print(f"KẾT QUẢ DỰ ĐOÁN")
    print(f"{'='*50}")
    print(f"Ảnh: {image_path}")
    print(f"Hình học dự đoán: {predicted_class}")
    print(f"Độ tin cậy: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"\nXác suất cho tất cả các lớp:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:15s}: {predictions[0][i]:.4f} ({predictions[0][i]*100:.2f}%)")
    
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách sử dụng: python predict.py <đường_dẫn_ảnh>")
        print("Ví dụ: python predict.py C:\\xulianh\\test_image.jpg")
    else:
        image_path = sys.argv[1]
        predict_image(image_path)

