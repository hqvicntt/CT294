from flask import Flask, request, redirect, url_for, flash, render_template_string
import os
import pandas as pd
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'secretkey'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
label_encoders = None
replace_map = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dự đoán mẫu nấm</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container mt-5">
    <h2 class="mb-4 text-center text-primary">Ứng dụng Dự đoán Nấm với Mô hình ML</h2>

    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
          <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
            {{ message }}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
          </div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <form action="{{ url_for('upload_model') }}" method="POST" enctype="multipart/form-data" class="mb-4 border p-3 bg-white">
        <h5>Tải mô hình lên (.pkl)</h5>
        <div class="input-group">
            <input type="file" name="model_file" class="form-control" required>
            <button type="submit" class="btn btn-success">Tải mô hình</button>
        </div>
    </form>

    <form action="{{ url_for('predict') }}" method="POST" enctype="multipart/form-data" class="mb-4 border p-3 bg-white">
        <h5>Tải dữ liệu CSV để dự đoán</h5>
        <div class="input-group">
            <input type="file" name="csv_file" class="form-control" required>
            <button type="submit" class="btn btn-primary">Dự đoán</button>
        </div>
    </form>

    {% if prediction_table %}
    <div class="border p-3 bg-white">
        <h5>Kết quả dự đoán:</h5>
        {{ prediction_table | safe }}
    </div>
    {% endif %}
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global model, label_encoders, replace_map
    file = request.files['model_file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            if isinstance(loaded, tuple) and len(loaded) == 3:
                model, label_encoders, replace_map = loaded
            elif isinstance(loaded, tuple) and len(loaded) == 2:
                model, label_encoders = loaded
                replace_map = {}
            else:
                model = loaded
                label_encoders = {}
                replace_map = {}
        flash('Mô hình đã được tải thành công!', 'success')
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    global model, label_encoders, replace_map
    if model is None:
        flash('Bạn cần tải mô hình trước!', 'danger')
        return redirect(url_for('index'))

    file = request.files['csv_file']
    if not file:
        flash('Hãy chọn file CSV để dự đoán.', 'danger')
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    try:
        new_data = pd.read_csv(filepath)

        # Xử lý '?' giống như khi huấn luyện
        for col in new_data.columns:
            if col in label_encoders:
                if new_data[col].isin(['?']).any():
                    if col in replace_map:
                        new_data[col] = new_data[col].replace('?', replace_map[col])
                    else:
                        fallback = label_encoders[col].classes_[0]
                        new_data[col] = new_data[col].replace('?', fallback)
                new_data[col] = label_encoders[col].transform(new_data[col])

        predictions = model.predict(new_data)

        # Giải mã nhãn dự đoán từ số → chữ (edible, poisonous)
        inv_label = label_encoders['class'].inverse_transform(predictions)
        label_meaning = {'e': 'Edible (Ăn được)', 'p': 'Poisonous (Độc)'}

        # Tạo bảng kết quả đơn giản
        results_html = '<table class="table table-bordered"><thead><tr><th>Dòng</th><th>Kết quả Dự đoán</th></tr></thead><tbody>'
        for idx, label in enumerate(inv_label, 1):
            readable = label_meaning.get(label, label)
            results_html += f'<tr><td>Dòng {idx}</td><td>{readable}</td></tr>'
        results_html += '</tbody></table>'

        return render_template_string(HTML_TEMPLATE, prediction_table=results_html)

    except Exception as e:
        flash(f'Lỗi khi dự đoán: {str(e)}', 'danger')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
