"""
UI đơn giản cho mô hình nhận dạng hình học toán học
Sử dụng Gradio
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import gradio as gr

# Thiết lập đường dẫn
MODEL_PATH = r"C:\xulianh\geometry_model.h5"

# Tên các lớp
CLASS_NAMES = ['circle', 'kite', 'parallelogram', 'rectangle', 
               'rhombus', 'square', 'trapezoid', 'triangle']

# Tên tiếng Việt
CLASS_NAMES_VI = {
    'circle': 'Hình tròn',
    'kite': 'Diều',
    'parallelogram': 'Hình bình hành',
    'rectangle': 'Hình chữ nhật',
    'rhombus': 'Hình thoi',
    'square': 'Hình vuông',
    'trapezoid': 'Hình thang',
    'triangle': 'Tam giác'
}

IMG_SIZE = 224

# Load mô hình (load một lần khi khởi động)
model = None

def load_model():
    """Load mô hình một lần"""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy mô hình tại: {MODEL_PATH}\n"
                "Vui lòng train mô hình trước bằng cách chạy: python train_model.py"
            )
        model = keras.models.load_model(MODEL_PATH)
    return model

def predict_geometry(img):
    """
    Dự đoán hình học từ ảnh
    
    Args:
        img: Ảnh đầu vào (PIL Image hoặc numpy array)
        
    Returns:
        Dictionary với kết quả dự đoán
    """
    try:
        # Load mô hình
        model = load_model()
        
        # Kiểm tra ảnh đầu vào
        if img is None:
            return "❌ Vui lòng upload ảnh để nhận dạng!"
        
        # Preprocess ảnh
        if isinstance(img, np.ndarray):
            # Nếu là numpy array, chuyển sang PIL Image
            from PIL import Image
            img = Image.fromarray(img)
        
        # Resize và normalize
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Dự đoán
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = CLASS_NAMES[predicted_class_idx]
        predicted_class_vi = CLASS_NAMES_VI[predicted_class]
        
        # Tạo kết quả
        result_text = f"🎯 **Hình học dự đoán: {predicted_class_vi}** ({predicted_class})\n\n"
        result_text += f"📊 **Độ tin cậy: {confidence*100:.2f}%**\n\n"
        
        if confidence >= 0.9:
            result_text += "✅ Độ tin cậy RẤT CAO - Kết quả đáng tin cậy\n\n"
        elif confidence >= 0.7:
            result_text += "✅ Độ tin cậy CAO - Kết quả tốt\n\n"
        elif confidence >= 0.5:
            result_text += "⚠️ Độ tin cậy TRUNG BÌNH - Có thể cần kiểm tra lại\n\n"
        else:
            result_text += "❌ Độ tin cậy THẤP - Kết quả không chắc chắn\n\n"
        
        # Xác suất cho tất cả các lớp
        result_text += "📈 **Xác suất cho tất cả các lớp:**\n\n"
        sorted_indices = np.argsort(predictions[0])[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            class_name = CLASS_NAMES[idx]
            class_name_vi = CLASS_NAMES_VI[class_name]
            prob = predictions[0][idx]
            marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            result_text += f"{marker} {class_name_vi:20s} ({class_name:15s}): {prob*100:6.2f}%\n"
        
        return result_text
        
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# Tạo giao diện Gradio
def create_interface():
    """Tạo giao diện Gradio"""
    
    # Mô tả
    description = """
    # 🔷 Nhận Dạng Hình Học Toán Học
    
    Upload ảnh chứa một trong các hình học sau để nhận dạng:
    - 🔵 Hình tròn (Circle)
    - 🪁 Diều (Kite)
    - ⬥ Hình bình hành (Parallelogram)
    - ▭ Hình chữ nhật (Rectangle)
    - ◇ Hình thoi (Rhombus)
    - ■ Hình vuông (Square)
    - ⏢ Hình thang (Trapezoid)
    - △ Tam giác (Triangle)
    
    **Lưu ý:** 
    - Ảnh nên có độ phân giải ≥ 224x224 pixels
    - Hình học nên rõ ràng, nổi bật trên nền
    - Nền đơn giản (trắng hoặc đơn màu) cho kết quả tốt nhất
    """
    
    # Tạo interface
    iface = gr.Interface(
        fn=predict_geometry,
        inputs=gr.Image(type="pil", label="Upload ảnh hình học"),
        outputs=gr.Textbox(label="Kết quả nhận dạng", lines=15),
        title="🔷 Nhận Dạng Hình Học Toán Học",
        description=description,
        examples=None,  # Có thể thêm examples sau
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return iface

if __name__ == "__main__":
    print("Đang khởi động ứng dụng...")
    print("Vui lòng đợi trong khi load mô hình...")
    
    # Load mô hình trước
    try:
        load_model()
        print("✅ Đã load mô hình thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load mô hình: {e}")
        print("Vui lòng đảm bảo mô hình đã được train và lưu tại:", MODEL_PATH)
        exit(1)
    
    # Tạo và launch interface
    iface = create_interface()
    iface.launch(
        server_name="127.0.0.1",  # Cho phép truy cập từ mạng
        server_port=7860,        # Port mặc định của Gradio
        share=False,             # Set True nếu muốn tạo public link
        show_error=True
    )

