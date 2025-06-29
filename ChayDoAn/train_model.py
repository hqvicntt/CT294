# ChayDoAn/train_model.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
from train_core import load_and_prepare_data, models

def train_and_display():
    acc_dict = {}
    misclass_dict = {}
    avg_accuracies = {}
    best_model_name = None
    best_model_score = 0
    best_model_object = None

    df, label_encoders, replace_map = load_and_prepare_data()
    X = df.drop('class', axis=1)
    y = df['class']

    for name, model in models.items():
        acc_sum = 0
        miss_sum = 0
        for i in range(10):
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=i)
            model.fit(X_train_i, y_train_i)
            y_pred_i = model.predict(X_test_i)
            acc_i = accuracy_score(y_test_i, y_pred_i)
            mis_i = (y_pred_i != y_test_i).sum()
            acc_sum += acc_i
            miss_sum += mis_i

        avg_acc = acc_sum / 10
        avg_mis = int(miss_sum / 10)

        avg_accuracies[name] = avg_acc
        acc_dict[name] = avg_acc
        misclass_dict[name] = avg_mis

        if avg_acc > best_model_score:
            best_model_score = avg_acc
            best_model_name = name
            best_model_object = model

    with open("best_model.pkl", "wb") as f:
        pickle.dump((best_model_object, label_encoders, replace_map), f)

    # Vẽ biểu đồ, hiển thị GUI
    for widget in chart_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(avg_accuracies.keys(), avg_accuracies.values(), color=['green', 'orange', 'blue'])
    ax.set_ylabel('Accuracy (avg of 10 runs)')
    ax.set_title('Độ chính xác trung bình của các mô hình')
    for i, (k, v) in enumerate(avg_accuracies.items()):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    stat_box.delete(*stat_box.get_children())
    for name in models.keys():
        stat_box.insert('', 'end', values=(name, f"{acc_dict[name]:.4f}", misclass_dict[name]))

    messagebox.showinfo("Hoàn tất", f"✅ Đã lưu mô hình tốt nhất: {best_model_name} với độ chính xác trung bình: {best_model_score:.4f}")

# GUI:
root = tk.Tk()
root.title("Huấn luyện mô hình nhận dạng nấm")

top_frame = tk.Frame(root)
top_frame.pack(pady=10)
btn_train = tk.Button(top_frame, text="Huấn luyện mô hình", command=train_and_display)
btn_train.pack(side=tk.LEFT, padx=5)

chart_frame = tk.Frame(root)
chart_frame.pack(padx=10, pady=10)

stat_frame = tk.LabelFrame(root, text="Bảng thống kê")
stat_frame.pack(padx=10, pady=10)

stat_box = ttk.Treeview(stat_frame, columns=("model", "accuracy", "misclassified"), show='headings')
stat_box.heading("model", text="Mô hình")
stat_box.heading("accuracy", text="Độ chính xác")
stat_box.heading("misclassified", text="Số mẫu sai")
stat_box.pack()

root.mainloop()
