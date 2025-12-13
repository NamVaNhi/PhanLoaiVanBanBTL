import pandas as pd
import tkinter as tk
from pyvi import ViTokenizer
from tkinter import messagebox, filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import warnings

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Lưu ý: Đảm bảo folder 'train_full' nằm cùng chỗ với file code
DATA_FOLDER_PATH = "Train_Full"

# ==============================================================================
# PHẦN 1: ĐỌC DỮ LIỆU (OUTPUT CHUẨN YÊU CẦU)
# ==============================================================================

def load_data_exact_output():
    print(f"📂 Đang quét dữ liệu từ: {os.path.abspath(DATA_FOLDER_PATH)}")
    # Kiểm tra thư mục
    if not os.path.exists(DATA_FOLDER_PATH):
        # Thử tìm ở đường dẫn tuyệt đối nếu chạy trong VS Code bị sai đường dẫn
        abs_path = os.path.join(os.getcwd(), DATA_FOLDER_PATH)
        if not os.path.exists(abs_path):
            messagebox.showerror("Lỗi", f"Không tìm thấy folder '{DATA_FOLDER_PATH}'")
            return create_dummy_data()
    
    data = []
    try:
        # 1. Lấy danh sách 10 chủ đề (Sắp xếp theo tên A-Z để in ra cho đẹp)
        sub_folders = sorted([f for f in os.listdir(DATA_FOLDER_PATH) if os.path.isdir(os.path.join(DATA_FOLDER_PATH, f))])
        
        # --- IN RA DÒNG BẠN MUỐN ---
        print(f"🔎 Tìm thấy {len(sub_folders)} chủ đề: {sub_folders}")
        
        for folder_name in sub_folders:
            folder_path = os.path.join(DATA_FOLDER_PATH, folder_name)
            
            # --- IN RA TIẾN TRÌNH ---
            print(f"   -> Đang đọc folder: '{folder_name}'...")
            
            # Quét file
            files = os.listdir(folder_path)
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    # Cơ chế đọc thử UTF-16 rồi đến UTF-8
                    try:
                        with open(file_path, 'r', encoding='utf-16') as f:
                            content = f.read()
                            if len(content) > 10: data.append({'text': content, 'category': folder_name})
                    except:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if len(content) > 10: data.append({'text': content, 'category': folder_name})
                        except:
                            pass # Bỏ qua file lỗi

        # Tạo bảng dữ liệu
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            print("⚠️ Không đọc được bài nào! Hãy kiểm tra lại folder.")
            return create_dummy_data()

        # --- IN KẾT QUẢ THỐNG KÊ CUỐI CÙNG ---
        print(f"✅ Đã đọc xong! Tổng số bài: {len(df)}")
        print("--- Số lượng bài theo từng chủ đề ---")
        # In ra bảng thống kê (Pandas sẽ tự sắp xếp giảm dần như ý bạn)
        print(df['category'].value_counts())
        
        return df

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return create_dummy_data()

def create_dummy_data():
    data = {'text': ["Demo"], 'category': ["Demo"]}
    return pd.DataFrame(data)


def simple_preprocess(text):
    return ViTokenizer.tokenize(str(text).lower())

def train_model():
    # Load dữ liệu với giao diện console chuẩn
    df = load_data_exact_output()
    # Xử lý & Huấn luyện
    df['text_clean'] = df['text'].apply(simple_preprocess)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
    )
    print("\n🧠 Đang huấn luyện mô hình Naive Bayes (Sẽ mất 1 lúc với 33k bài)...")
    # Tạo pipeline gồm TF-IDF và mô hình Naive Bayes
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    # In báo cáo chi tiết
    print("\n" + "="*60)
    print("BẢNG ĐÁNH GIÁ (CLASSIFICATION REPORT)")
    print("="*60)
    y_pred = model.predict(X_test)
    labels = sorted(list(set(y_test)))
    print(classification_report(y_test, y_pred, target_names=labels))
    print("="*60 + "\n")
    return model
# Chạy huấn luyện ngay khi bật
final_model = train_model()

# ==============================================================================
# PHẦN 2: GIAO DIỆN (FRONTEND)
# ==============================================================================

def on_click_predict():
    if final_model is None: return
    raw_text = txt_input.get("1.0", "end-1c")
    if len(raw_text.strip()) < 2: return
    
    text_clean = simple_preprocess(raw_text)
    prediction = final_model.predict([text_clean])[0]
    proba = final_model.predict_proba([text_clean]).max() * 100
    
    lbl_result.config(text=f"CHỦ ĐỀ: {prediction}", fg="red")
    lbl_conf.config(text=f"(Độ tin cậy: {proba:.1f}%)")

def on_click_upload():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        content = ""
        try:
            with open(file_path, "r", encoding="utf-16") as f: content = f.read()
        except:
            try:
                with open(file_path, "r", encoding="utf-8") as f: content = f.read()
            except:
                messagebox.showerror("Lỗi", "Không đọc được file này!")
                return
        
        txt_input.delete("1.0", tk.END)
        txt_input.insert(tk.END, content)

def on_click_clear():
    txt_input.delete("1.0", tk.END)
    lbl_result.config(text="...", fg="black")
    lbl_conf.config(text="")

# GUI Setup
root = tk.Tk()
root.title("Phân loại Tin tức (VNTC Full)")
root.geometry("650x600")

tk.Label(root, text="PHÂN LOẠI TIN TỨC (10 CHỦ ĐỀ)", font=("Arial", 16, "bold"), fg="blue").pack(pady=10)
tk.Label(root, text=f"Dữ liệu từ: {DATA_FOLDER_PATH}", font=("Arial", 10, "italic"), fg="green").pack()

txt_input = tk.Text(root, height=10, width=60, font=("Arial", 11)); txt_input.pack(pady=10)

frame_btn = tk.Frame(root); frame_btn.pack(pady=5)
tk.Button(frame_btn, text="📂 Tải file", command=on_click_upload).grid(row=0, column=0, padx=10)
tk.Button(frame_btn, text="🔍 PHÂN TÍCH", command=on_click_predict, bg="blue", fg="white", font=("Arial", 11, "bold")).grid(row=0, column=1, padx=10)
tk.Button(frame_btn, text="🗑 XÓA", command=on_click_clear, bg="red", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=10)

lbl_result = tk.Label(root, text="...", font=("Arial", 18, "bold")); lbl_result.pack(pady=10)
lbl_conf = tk.Label(root, text=""); lbl_conf.pack()

root.mainloop()