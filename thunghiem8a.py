import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
import zipfile
import urllib.request

# === Tải và xử lý dữ liệu từ GitHub ===
def load_and_prepare_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    df = pd.read_csv(url, header=None, names=columns)
    if df['stalk-root'].isin(['?']).any():
        mode = df['stalk-root'][df['stalk-root'] != '?'].mode()[0]
        df['stalk-root'] = df['stalk-root'].replace('?', mode)

    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# def load_and_prepare_data():
#     url = "https://github.com/DuongB2012074/mushroom-dataset/raw/main/mushroom.zip"
#     zip_path = "mushroom.zip"
#     extract_path = "mushroom"
#     csv_filename = "agaricus-lepiota.data"
#
#     if not os.path.exists(zip_path):
#         urllib.request.urlretrieve(url, zip_path)
#
#     if not os.path.exists(extract_path):
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_path)
#
#     file_path = os.path.join(extract_path, csv_filename)
#     columns = [
#         'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
#         'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
#         'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#         'stalk-surface-below-ring', 'stalk-color-above-ring',
#         'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
#         'ring-type', 'spore-print-color', 'population', 'habitat'
#     ]
#     df = pd.read_csv(file_path, header=None, names=columns)
#
#     # Xử lý duy nhất cột 'stalk-root' chứa '?'
#     if df['stalk-root'].isin(['?']).any():
#         mode = df['stalk-root'][df['stalk-root'] != '?'].mode()[0]
#         df['stalk-root'] = df['stalk-root'].replace('?', mode)
#
#     label_encoders = {}
#     for col in df.columns:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le
#
#     return df, label_encoders

# === Dự đoán dữ liệu mới ===
def predict_new_data(encoded_new_data, models, label_encoders):
    results = {}
    for name, model in models.items():
        pred = model.predict(encoded_new_data)
        labels = label_encoders['class'].inverse_transform(pred)
        results[name] = labels
    return results

# === Xử lý khi người dùng chọn file CSV để dự đoán ===
def on_select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    try:
        new_data = pd.read_csv(file_path)
        for col in new_data.columns:
            if new_data[col].isin(['?']).any():
                most_common = label_encoders[col].classes_[0]
                new_data[col] = new_data[col].replace('?', most_common)
        encoded_new = pd.DataFrame()
        for col in new_data.columns:
            encoded_new[col] = label_encoders[col].transform(new_data[col])
        predictions = predict_new_data(encoded_new, models, label_encoders)

        output_box.delete(*output_box.get_children())
        for i in range(len(new_data)):
            row = [f"Dòng {i+1}"]
            for name in models.keys():
                row.append(predictions[name][i])
            output_box.insert('', 'end', values=row)

    except Exception as e:
        messagebox.showerror("Lỗi", str(e))

# === Huấn luyện và hiển thị kết quả ===
def train_and_display():
    acc_dict = {}
    misclass_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        misclassified = (y_pred != y_test).sum()

        acc_dict[name] = acc
        misclass_dict[name] = misclassified

    for widget in chart_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(acc_dict.keys(), acc_dict.values(), color=['green', 'orange', 'blue'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Độ chính xác của các mô hình')
    for i, (k, v) in enumerate(acc_dict.items()):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    stat_box.delete(*stat_box.get_children())
    for name in models.keys():
        stat_box.insert('', 'end', values=(name, f"{acc_dict[name]:.4f}", misclass_dict[name]))

# === GUI chính ===
root = tk.Tk()
root.title("Phân loại Nấm với nhiều mô hình ML")
root.geometry("1200x650")

main_canvas = tk.Canvas(root)
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=main_canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

main_canvas.configure(yscrollcommand=scrollbar.set)
main_canvas.bind('<Configure>', lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

content_frame = tk.Frame(main_canvas)
main_canvas.create_window((0, 0), window=content_frame, anchor='nw')

# === Tải dữ liệu ===
df, label_encoders = load_and_prepare_data()
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = int(math.sqrt(len(X_train)))
if k % 2 == 0:
    k += 1

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    f"KNN (k={k})": KNeighborsClassifier(n_neighbors=k)
}

# === Giao diện nút điều khiển ===
top_frame = tk.Frame(content_frame)
top_frame.pack(pady=10)

btn_train = tk.Button(top_frame, text="Huấn luyện và Hiển thị", command=train_and_display)
btn_train.pack(side=tk.LEFT, padx=10)

btn_file = tk.Button(top_frame, text="Chọn file CSV để dự đoán", command=on_select_file)
btn_file.pack(side=tk.LEFT)

# === Hiển thị tổng số mẫu, train/test ===
summary_text = f"Tổng số mẫu: {len(df)} | Huấn luyện: {len(X_train)} mẫu | Kiểm thử: {len(X_test)} mẫu"
summary_label = tk.Label(content_frame, text=summary_text, font=("Arial", 11), fg="blue")
summary_label.pack(pady=5)

# === Bố cục chia cột: trái (biểu đồ + thống kê), phải (dự đoán) ===
main_body = tk.Frame(content_frame)
main_body.pack(fill="both", expand=True, padx=10)

# --- Cột trái ---
left_col = tk.Frame(main_body)
left_col.pack(side=tk.LEFT, fill="both", expand=True)

chart_frame = tk.Frame(left_col)
chart_frame.pack(pady=10)

stat_frame = tk.LabelFrame(left_col, text="Bảng thống kê")
stat_frame.pack(pady=5, fill="x")
stat_box = ttk.Treeview(stat_frame, columns=("model", "accuracy", "misclassified"), show='headings')
for col in stat_box["columns"]:
    stat_box.heading(col, text=col)
stat_box.pack(fill="x")

# --- Cột phải ---
right_col = tk.LabelFrame(main_body, text="Kết quả dự đoán dữ liệu mới")
right_col.pack(side=tk.LEFT, fill="both", expand=True, padx=10)

output_frame = tk.Frame(right_col)
output_frame.pack(fill="both", expand=True)

columns = ["Dòng"] + list(models.keys())
output_box = ttk.Treeview(output_frame, columns=columns, show='headings', height=20)

for col in columns:
    output_box.heading(col, text=col)
    output_box.column(col, width=120, anchor='center')

scrollbar_output = ttk.Scrollbar(output_frame, orient="vertical", command=output_box.yview)
output_box.configure(yscrollcommand=scrollbar_output.set)

output_box.grid(row=0, column=0, sticky='nsew')
scrollbar_output.grid(row=0, column=1, sticky='ns')

output_frame.grid_rowconfigure(0, weight=1)
output_frame.grid_columnconfigure(0, weight=1)

# === Bắt đầu giao diện ===
root.mainloop()
