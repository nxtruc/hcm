import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openml
import os
import mlflow
import plotly.express as px
import shutil
import time
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split 

def mlflow_input():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    DAGSHUB_USERNAME = "Snxtruc"  # Thay bằng username của bạn
    DAGSHUB_REPO_NAME = "HocMayPython"
    DAGSHUB_TOKEN = "ca4b78ae4dd9d511c1e0c333e3b709b2cd789a19"  # Thay bằng Access Token của bạn

    # Đặt URI MLflow để trỏ đến DagsHub
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

    # Thiết lập authentication bằng Access Token
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    # Đặt thí nghiệm MLflow
    mlflow.set_experiment("Semi-supervised")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"

def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def display_mlflow_experiments():
    """Hiển thị danh sách Runs trong MLflow."""
    st.title("📊 MLflow Experiment Viewer")

    mlflow_input()

    experiment_name = "Neural Network"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách Runs
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    # Xử lý dữ liệu runs để hiển thị
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_data = mlflow.get_run(run_id)
        run_tags = run_data.data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")  # Lấy tên từ tags nếu có
        created_time = format_time_relative(run_data.info.start_time)
        duration = (run_data.info.end_time - run_data.info.start_time) / 1000 if run_data.info.end_time else "Đang chạy"
        source = run_tags.get("mlflow.source.name", "Unknown")

        run_info.append({
            "Run Name": run_name,
            "Run ID": run_id,
            "Created": created_time,
            "Duration (s)": duration if isinstance(duration, str) else f"{duration:.1f}s",
            "Source": source
        })

    # Sắp xếp run theo thời gian chạy (mới nhất trước)
    run_info_df = pd.DataFrame(run_info)
    run_info_df = run_info_df.sort_values(by="Created", ascending=False)

    # Hiển thị danh sách Runs trong bảng
    st.write("### 🏃‍♂️ Danh sách Runs:")
    st.dataframe(run_info_df, use_container_width=True)

    # Chọn Run từ dropdown
    run_names = run_info_df["Run Name"].tolist()
    selected_run_name = st.selectbox("🔍 Chọn một Run để xem chi tiết:", run_names)

    # Lấy Run ID tương ứng
    selected_run_id = run_info_df.loc[run_info_df["Run Name"] == selected_run_name, "Run ID"].values[0]

    # Lấy thông tin Run
    selected_run = mlflow.get_run(selected_run_id)

    # --- 📝 ĐỔI TÊN RUN ---
    st.write("### ✏️ Đổi tên Run")
    new_run_name = st.text_input("Nhập tên mới:", selected_run_name)
    if st.button("💾 Lưu tên mới"):
        try:
            mlflow.set_tag(selected_run_id, "mlflow.runName", new_run_name)
            st.success(f"✅ Đã đổi tên thành **{new_run_name}**. Hãy tải lại trang để thấy thay đổi!")
        except Exception as e:
            st.error(f"❌ Lỗi khi đổi tên: {e}")

    # --- 🗑️ XÓA RUN ---
    st.write("### ❌ Xóa Run")
    if st.button("🗑️ Xóa Run này"):
        try:
            mlflow.delete_run(selected_run_id)
            st.success(f"✅ Đã xóa run **{selected_run_name}**! Hãy tải lại trang để cập nhật danh sách.")
        except Exception as e:
            st.error(f"❌ Lỗi khi xóa run: {e}")

    # --- HIỂN THỊ CHI TIẾT RUN ---
    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")

        start_time_ms = selected_run.info.start_time
        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        # Hiển thị model artifact (nếu có)
        model_artifact_path = f"{st.session_state['mlflow_url']}/{selected_experiment.experiment_id}/{selected_run_id}/artifacts/model"
        st.write("### 📂 Model Artifact:")
        st.write(f"📥 [Tải mô hình]({model_artifact_path})")

    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")

def tong_quan():
    st.title("Tổng quan về tập dữ liệu MNIST")

    st.header("1. Giới thiệu")
    st.write("Tập dữ liệu MNIST (Modified National Institute of Standards and Technology) là một trong những tập dữ liệu phổ biến nhất trong lĩnh vực Machine Learning và Computer Vision, thường được dùng để huấn luyện và kiểm thử các mô hình phân loại chữ số viết tay.") 

    st.image("https://datasets.activeloop.ai/wp-content/uploads/2019/12/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp", use_container_width=True)

    st.subheader("Nội dung")
    st.write("- 70.000 ảnh grayscale (đen trắng) của các chữ số viết tay từ 0 đến 9.")
    st.write("- Kích thước ảnh: 28x28 pixel.")
    st.write("- Định dạng: Mỗi ảnh được biểu diễn bằng một ma trận 28x28 có giá trị pixel từ 0 (đen) đến 255 (trắng).")
    st.write("- Nhãn: Một số nguyên từ 0 đến 9 tương ứng với chữ số trong ảnh.")

    st.header("2. Nguồn gốc và ý nghĩa")
    st.write("- Được tạo ra từ bộ dữ liệu chữ số viết tay gốc của NIST, do LeCun, Cortes và Burges chuẩn bị.")
    st.write("- Dùng làm benchmark cho các thuật toán nhận diện hình ảnh, đặc biệt là mạng nơ-ron nhân tạo (ANN) và mạng nơ-ron tích chập (CNN).")
    st.write("- Rất hữu ích cho việc kiểm thử mô hình trên dữ liệu hình ảnh thực tế nhưng đơn giản.")

    st.header("3. Phân chia tập dữ liệu")
    st.write("- Tập huấn luyện: 60.000 ảnh.")
    st.write("- Tập kiểm thử: 10.000 ảnh.")
    st.write("- Mỗi tập có phân bố đồng đều về số lượng chữ số từ 0 đến 9.")

    st.header("4. Ứng dụng")
    st.write("- Huấn luyện và đánh giá các thuật toán nhận diện chữ số viết tay.")
    st.write("- Kiểm thử và so sánh hiệu suất của các mô hình học sâu (Deep Learning).")
    st.write("- Làm bài tập thực hành về xử lý ảnh, trích xuất đặc trưng, mô hình phân loại.")
    st.write("- Cung cấp một baseline đơn giản cho các bài toán liên quan đến Computer Vision.")

    st.header("5. Phương pháp tiếp cận phổ biến")
    st.write("- Trích xuất đặc trưng truyền thống: PCA, HOG, SIFT...")
    st.write("- Machine Learning: KNN, SVM, Random Forest, Logistic Regression...")
    st.write("- Deep Learning: MLP, CNN (LeNet-5, AlexNet, ResNet...), RNN")

    st.caption("Ứng dụng hiển thị thông tin về tập dữ liệu MNIST bằng Streamlit 🚀")

def up_load_db():
    st.header("📥 Tải Dữ Liệu")
    
    if "data" in st.session_state and st.session_state.data is not None:
        st.warning("🔸 **Dữ liệu đã được tải lên rồi!** Bạn có thể tiếp tục với các bước tiếp theo.")
    else:
        option = st.radio("Chọn nguồn dữ liệu:", ["Tải từ OpenML", "Upload dữ liệu"], key="data_source_radio")
        
        if "data" not in st.session_state:
            st.session_state.data = None
        
        if option == "Tải từ OpenML":
            st.markdown("#### 📂 Tải dữ liệu MNIST từ OpenML")
            if st.button("Tải dữ liệu MNIST", key="download_mnist_button"):
                with st.status("🔄 Đang tải dữ liệu MNIST từ OpenML...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    for percent_complete in range(0, 101, 20):
                        time.sleep(0.5)
                        progress_bar.progress(percent_complete)
                        status.update(label=f"🔄 Đang tải... ({percent_complete}%)")
                    
                    X = np.load("X.npy")
                    y = np.load("y.npy")
                    
                    status.update(label="✅ Tải dữ liệu thành công!", state="complete")
                    
                    st.session_state.data = (X, y)
        
        else:
            st.markdown("#### 📤 Upload dữ liệu của bạn")
            uploaded_file = st.file_uploader("Chọn một file ảnh", type=["png", "jpg", "jpeg"], key="file_upload")
            
            if uploaded_file is not None:
                with st.status("🔄 Đang xử lý ảnh...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    for percent_complete in range(0, 101, 25):
                        time.sleep(0.3)
                        progress_bar.progress(percent_complete)
                        status.update(label=f"🔄 Đang xử lý... ({percent_complete}%)")
                    
                    image = Image.open(uploaded_file).convert('L')
                    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
                    
                    if image.size != (28, 28):
                        status.update(label="❌ Ảnh không đúng kích thước 28x28 pixel.", state="error")
                    else:
                        status.update(label="✅ Ảnh hợp lệ!", state="complete")
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                        image_tensor = transform(image).unsqueeze(0)
                        st.session_state.data = image_tensor
    
    if st.session_state.data is not None:
        st.markdown("#### ✅ Dữ liệu đã sẵn sàng!")
    else:
        st.warning("🔸 Vui lòng tải dữ liệu trước khi tiếp tục làm việc.")
    
    st.markdown("""
    🔹 **Lưu ý:**
    - Ứng dụng chỉ sử dụng dữ liệu ảnh dạng **28x28 pixel (grayscale)**.
    - Dữ liệu phải có cột **'label'** chứa nhãn (số từ 0 đến 9) khi tải từ OpenML.
    - Nếu dữ liệu của bạn không đúng định dạng, vui lòng sử dụng dữ liệu MNIST từ OpenML.
    """)

def chia_du_lieu():
    st.title("📌 Chia dữ liệu Train/Test")

    # Kiểm tra dữ liệu đã có chưa
    if "data" not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Vui lòng tải dữ liệu trước khi thực hiện chia dữ liệu.")
        return

    # Lấy dữ liệu từ session_state
    X, y = st.session_state.data
    total_samples = X.shape[0]

    # Nếu chưa có flag "data_split_done", mặc định là False
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.number_input("📌 Nhập số lượng ảnh để train:", min_value=1000, max_value=total_samples, value=20000, step=1000)
    
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    train_size = 100 - test_size
    st.write(f"📌 **Tỷ lệ phân chia:** Train={train_size}%, Test={test_size}%")

    # Placeholder cho bảng kết quả và thanh tiến trình
    progress_bar = st.empty()
    table_placeholder = st.empty()

    # Nút xác nhận chia dữ liệu
    if st.button("✅ Xác nhận & Lưu", key="luu"):
        progress_bar.progress(10)  # Bắt đầu tiến trình
        st.session_state.data_split_done = True  # Đánh dấu đã chia dữ liệu
        
        # Chọn số lượng dữ liệu cần lấy
        if num_samples == total_samples:
            X_selected, y_selected = X, y
        else:
            X_selected, _, y_selected, _ = train_test_split(
                X, y, train_size=num_samples, stratify=y, random_state=42
            )

        progress_bar.progress(40)  # Tiến trình 40%

        # Chia train/test
        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        progress_bar.progress(70)  # Tiến trình 70%

        # Lưu vào session_state
        st.session_state.total_samples = num_samples
        st.session_state["neural_X_train"] = X_train
        st.session_state["neural_X_test"] = X_test
        st.session_state["neural_y_train"] = y_train
        st.session_state["neural_y_test"] = y_test
        st.session_state.test_size = X_test.shape[0]
        st.session_state.train_size = X_train.shape[0]

        progress_bar.progress(90)  # Gần hoàn tất

        # Lưu bảng kết quả
        st.session_state.summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_test.shape[0]]
        })

        st.success("✅ Dữ liệu đã được chia thành công!")
        table_placeholder.table(st.session_state.summary_df)
        progress_bar.progress(100)  # Hoàn thành

    # Nếu đã chia dữ liệu trước đó
    if st.session_state.data_split_done:
        if "summary_df" in st.session_state:
            table_placeholder.table(st.session_state.summary_df)
        
        st.info("✅ Dữ liệu đã được chia. Nhấn nút dưới đây để chia lại nếu muốn thay đổi.")
        
        if st.button("🔄 Chia lại dữ liệu", key="chia_lai"):
            progress_bar.progress(10)  # Bắt đầu lại tiến trình
            table_placeholder.empty()

            if num_samples == total_samples:
                X_selected, y_selected = X, y
            else:
                X_selected, _, y_selected, _ = train_test_split(
                    X, y, train_size=num_samples, stratify=y, random_state=42
                )

            progress_bar.progress(40)

            # Chia train/test
            stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
            )

            progress_bar.progress(70)

            # Lưu vào session_state
            st.session_state.total_samples = num_samples
            st.session_state["neural_X_train"] = X_train
            st.session_state["neural_X_test"] = X_test
            st.session_state["neural_y_train"] = y_train
            st.session_state["neural_y_test"] = y_test
            st.session_state.test_size = X_test.shape[0]
            st.session_state.train_size = X_train.shape[0]

            progress_bar.progress(90)

            # Cập nhật bảng kết quả mới
            st.session_state.summary_df = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Test"],
                "Số lượng mẫu": [X_train.shape[0], X_test.shape[0]]
            })
            st.success("✅ Dữ liệu đã được chia lại thành công!")
            table_placeholder.table(st.session_state.summary_df)
            progress_bar.progress(100)  # Hoàn thành


#Callback phuc vu train2
class ProgressBarCallback:
    def __init__(self, total_epochs, progress_bar, status_text, max_train_progress=80):
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.max_train_progress = max_train_progress

    def on_epoch_begin(self, epoch):
        progress = (epoch + 1) / self.total_epochs * self.max_train_progress
        self.progress_bar.progress(min(int(progress), self.max_train_progress))
        self.status_text.text(f"🛠️ Đang huấn luyện mô hình... Epoch {epoch + 1}/{self.total_epochs}")

    def on_train_end(self):
        self.progress_bar.progress(self.max_train_progress)
        self.status_text.text("✅ Huấn luyện mô hình hoàn tất, đang chuẩn bị logging...")

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_layers, num_nodes, activation):
        super(NeuralNet, self).__init__()
        layers = [nn.Linear(input_size, num_nodes), self.get_activation(activation)]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(self.get_activation(activation))
        
        layers.append(nn.Linear(num_nodes, 10))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_activation(self, activation):
        """ Trả về hàm kích hoạt phù hợp """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Không hỗ trợ activation: {activation}")

def train2():
    st.header("⚙️ Pseudo Labelling với Neural Network (PyTorch)")
    
    if "neural_X_train" not in st.session_state or "neural_X_test" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return
    
    X_train_full = torch.tensor(st.session_state["neural_X_train"].reshape(-1, 28 * 28) / 255.0, dtype=torch.float32)
    y_train_full = torch.tensor(st.session_state["neural_y_train"], dtype=torch.long)
    X_test = torch.tensor(st.session_state["neural_X_test"].reshape(-1, 28 * 28) / 255.0, dtype=torch.float32)
    y_test = torch.tensor(st.session_state["neural_y_test"], dtype=torch.long)
    
    # Lấy 1% dữ liệu ban đầu
    X_initial, y_initial = [], []
    for digit in range(10):
        indices = torch.where(y_train_full == digit)[0]
        num_samples = max(1, int(0.01 * len(indices)))
        selected_indices = indices[torch.randperm(len(indices))[:num_samples]]
        X_initial.append(X_train_full[selected_indices])
        y_initial.append(y_train_full[selected_indices])
    
    X_initial = torch.cat(X_initial, dim=0)
    y_initial = torch.cat(y_initial, dim=0)
    mask = torch.ones(len(X_train_full), dtype=torch.bool)
    mask[selected_indices] = False
    X_unlabeled = X_train_full[mask]
    
    # Chọn chế độ lặp
    loop_mode = st.selectbox("Chọn chế độ lặp:", ["Số vòng lặp cố định", "Gán hết toàn bộ tập train"], key="pseudo_iteration_mode")
    
    if loop_mode == "Số vòng lặp cố định":
        max_iterations = st.slider("Số vòng lặp tối đa", 1, 10, 5, key="pseudo_max_iter")
    else:
        st.warning("⚠️ Thời gian sẽ lâu do có thể lặp nhiều khi chọn 'Gán hết toàn bộ tập train'!")
        max_iterations = float('inf')
    
    # Hyperparameters
    num_layers = st.slider("Số lớp ẩn", 1, 5, 2)
    num_nodes = st.slider("Số node mỗi lớp", 32, 256, 128)
    activation = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"])
    epochs = st.slider("Số epoch mỗi vòng", 1, 50, 10)
    threshold = st.slider("Ngưỡng gán nhãn", 0.5, 1.0, 0.95, step=0.01)
    learn_rate = st.number_input("Tốc độ học", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
    
    run_name = st.text_input("🔹 Nhập tên Run:", "Pseudo_Default_Run")
    
    if st.button("Bắt đầu Pseudo Labelling"):
        mlflow.start_run(run_name=f"Pseudo_{run_name}")
        model = NeuralNet(28 * 28, num_layers, num_nodes, activation)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.CrossEntropyLoss()
        
        X_labeled, y_labeled = X_initial.clone(), y_initial.clone()
        X_unlabeled_remaining = X_unlabeled.clone()
        total_samples = len(X_train_full)
        
        iteration = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while iteration < max_iterations:
            st.write(f"### Vòng lặp {iteration + 1}")
            train_dataset = TensorDataset(X_labeled, y_labeled)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(epochs):
                status_text.text(f"🚀 Đang train - Epoch {epoch + 1}/{epochs}...")
                progress_bar.progress(int((epoch + 1) / epochs * 50))
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                time.sleep(0.5)  # Độ trễ mỗi epoch
            
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                test_acc = (outputs.argmax(dim=1) == y_test).float().mean().item()
            
            st.write(f"📊 Độ chính xác trên tập test: {test_acc:.4f}")
            mlflow.log_metric("pseudo_test_accuracy", test_acc, step=iteration)
            
            if len(X_unlabeled_remaining) == 0:
                break
            
            status_text.text("🔍 Đang dự đoán nhãn cho dữ liệu chưa gán...")
            with torch.no_grad():
                outputs = model(X_unlabeled_remaining)
                probs, predicted_labels = outputs.softmax(dim=1).max(dim=1)
            
            confident_mask = probs >= threshold
            X_confident = X_unlabeled_remaining[confident_mask]
            y_confident = predicted_labels[confident_mask]
            
            st.write(f"Số mẫu được gán nhãn giả: {X_confident.shape[0]} (ngưỡng: {threshold})")
            st.write(f"Số mẫu chưa gán nhãn còn lại: {X_unlabeled_remaining.shape[0] - X_confident.shape[0]}")
            
            if len(X_confident) == 0:
                break
            
            X_labeled = torch.cat([X_labeled, X_confident])
            y_labeled = torch.cat([y_labeled, y_confident])
            X_unlabeled_remaining = X_unlabeled_remaining[~confident_mask]
            
            labeled_fraction = X_labeled.shape[0] / total_samples
            progress_bar.progress(min(int(50 + 50 * labeled_fraction), 100))
            status_text.text(f"📈 Đã gán nhãn: {X_labeled.shape[0]}/{total_samples} mẫu ({labeled_fraction:.2%})")
            
            iteration += 1
            if loop_mode == "Gán hết toàn bộ tập train" and len(X_unlabeled_remaining) == 0:
                break
        
        torch.save(model.state_dict(), "pseudo_model_final.pth")
        mlflow.log_artifact("pseudo_model_final.pth")
        mlflow.end_run()
        
        st.success("✅ Quá trình Pseudo Labelling hoàn tất!")
        st.markdown(f"[🔗 Xem MLflow trên DAGsHub]({st.session_state['mlflow_url']})")


def Semi_supervised():
    st.markdown(
        """
        <style>
        .stTabs [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
            display: flex;
            scrollbar-width: thin;
            scrollbar-color: #888 #f0f0f0;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar {
            height: 6px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-thumb {
            background-color: #888;
            border-radius: 3px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-track {
            background: #f0f0f0;
        }
        .stTabs [role="tab"]:hover {
            background-color: #f0f0f0;
            transition: background-color 0.3s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    ) 
    st.markdown(" ### 🖊️ MNIST NN & Semi-supervised App")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Tổng quan", 
    "Tải dữ liệu",
    "Chia dữ liệu",
    "Huấn luyện", 
    "Thông tin huấn luyện"])

    with tab1: 
        tong_quan()
    with tab2: 
        up_load_db()
    with tab3: 
        chia_du_lieu()
    with tab4: 
        train2()
    with tab5:
        display_mlflow_experiments()
        
def run():
    Semi_supervised() 

if __name__ == "__main__":
    run()
