import sys
import os
import time
import traceback
import numpy as np
import cv2
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from shapely.geometry import Point
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox,
                             QTextEdit, QProgressBar, QGroupBox, QTabWidget, QMessageBox, QSplitter)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage

# 设置Matplotlib后端为Agg
mpl.use('Agg')


# ==========================================
# 核心算法类
# ==========================================

class FisherClassifier:
    def __init__(self):
        self.projectors = []
        self.class_means = []

    def train(self, samples, labels):
        num_classes = len(np.unique(labels))
        num_features = samples.shape[1]
        class_samples = []
        self.class_means = []

        for i in range(1, num_classes + 1):
            mask = (labels == i)
            class_i_samples = samples[mask]
            class_samples.append(class_i_samples)
            self.class_means.append(np.mean(class_i_samples, axis=0))

        total_mean = np.mean(samples, axis=0)                              # 计算类别中心和全局中心
        Sw = np.zeros((num_features, num_features))
        Sb = np.zeros((num_features, num_features))

        for i in range(num_classes):
            class_i_samples = class_samples[i]
            mean_i = self.class_means[i]
            centered = class_i_samples - mean_i
            Sw += np.dot(centered.T, centered)                              # 计算散度矩阵Sb和Sw
            mean_diff = mean_i - total_mean
            Sb += len(class_i_samples) * np.outer(mean_diff, mean_diff)

        try:
            Sw_inv = np.linalg.inv(Sw)
            eig_vals, eig_vecs = np.linalg.eig(np.dot(Sw_inv, Sb))
        except np.linalg.LinAlgError:
            Sw_inv = np.linalg.pinv(Sw)
            eig_vals, eig_vecs = np.linalg.eig(np.dot(Sw_inv, Sb))          # 求解广义特征值

        sorted_indices = np.argsort(eig_vals)[::-1]                         # 选择特征值最大的投影方向
        sorted_eig_vecs = eig_vecs[:, sorted_indices]
        k = min(num_classes - 1, num_features)
        self.projectors = sorted_eig_vecs[:, :k].real

    def predict(self, samples):
        projected = np.dot(samples, self.projectors)
        projected_means = []
        for mean in self.class_means:
            projected_means.append(np.dot(mean, self.projectors))            # 中心点映射到低维空间

        predictions = np.zeros(samples.shape[0], dtype=int)
        batch_size = 10000

        for i in range(0, len(samples), batch_size):
            end = min(i + batch_size, len(samples))
            batch = projected[i:end]
            distances = np.zeros((end - i, len(projected_means)))

            for c, p_mean in enumerate(projected_means):
                diff = batch - p_mean
                distances[:, c] = np.sqrt(np.sum(diff ** 2, axis=1))          # 根据欧几里得距离判定预测类别

            closest_class = np.argmin(distances, axis=1) + 1
            predictions[i:end] = closest_class
        return predictions


class BayesianClassifier:
    def __init__(self):
        self.means = []
        self.covs = []
        self.priors = []
        self.inv_covs = []
        self.det_covs = []

    def train(self, samples, labels):
        classes = np.unique(labels)
        num_classes = len(classes)
        num_samples = len(samples)
        self.means = []
        self.covs = []
        self.priors = []
        self.inv_covs = []
        self.det_covs = []

        for i in range(1, num_classes + 1):
            mask = (labels == i)
            class_samples = samples[mask]
            prior = len(class_samples) / num_samples          # 计算先验概率（样本/总体）
            self.priors.append(prior)
            mean = np.mean(class_samples, axis=0)
            self.means.append(mean)
            cov = np.cov(class_samples, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6
            self.covs.append(cov)

            try:
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)                   # 预计算逆矩阵和行列式
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)
                det_cov = np.linalg.det(cov)
                if det_cov == 0:
                    det_cov = 1e-10

            self.inv_covs.append(inv_cov)
            self.det_covs.append(det_cov)

    def _calculate_log_probability(self, x, mean, inv_cov, det_cov):
        n = len(mean)                                                            # 计算单样本在高斯分布下对数似然概率
        diff = x - mean
        exponent = -0.5 * np.dot(np.dot(diff, inv_cov), diff)
        const = -0.5 * (n * np.log(2 * np.pi) + np.log(max(det_cov, 1e-10)))
        return const + exponent

    def predict(self, samples):
        num_samples = len(samples)
        num_classes = len(self.means)
        predictions = np.zeros(num_samples, dtype=int)
        batch_size = 10000

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = samples[start:end]
            log_posteriors = np.zeros((end - start, num_classes))

            for c in range(num_classes):
                for i, sample in enumerate(batch):
                    log_posteriors[i, c] = self._calculate_log_probability(           # 计算每个样本对每个类的后验概率
                        sample, self.means[c], self.inv_covs[c], self.det_covs[c]
                    ) + np.log(max(self.priors[c], 1e-10))

            best_classes = np.argmax(log_posteriors, axis=1) + 1
            predictions[start:end] = best_classes

        return predictions


# ==========================================
# 工作线程
# ==========================================
class ProcessingThread(QThread):
    log_signal = pyqtSignal(str)  # 发送日志文本
    progress_signal = pyqtSignal(int)  # 发送进度条数值 (0-100)
    finished_signal = pyqtSignal(dict)  # 完成信号，携带结果路径字典
    error_signal = pyqtSignal(str)  # 错误信号

    def __init__(self, image_path, shp_path, output_dir, selected_models):
        super().__init__()
        self.image_path = image_path
        self.shp_path = shp_path
        self.output_dir = output_dir
        self.selected_models = selected_models
        self.is_running = True

    def calculate_class_accuracy(self, y_true, y_pred, class_dict):
        id_to_name = {v: k for k, v in class_dict.items()}
        cm = confusion_matrix(y_true, y_pred, labels=list(class_dict.values()))
        class_metrics = {}
        for i, class_id in enumerate(sorted(class_dict.values())):
            class_name = id_to_name[class_id]
            total_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            po_class = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            pe_class = (cm[i, :].sum() * cm[:, i].sum()) / (cm.sum() * cm.sum()) if cm.sum() > 0 else 0
            kappa_class = (po_class - pe_class) / (1 - pe_class) if (1 - pe_class) > 0 else 0
            precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            f1_class = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            class_metrics[class_name] = {
                'total_accuracy': total_accuracy, 'kappa': kappa_class, 'f1_score': f1_class,
                'sample_count': cm[i, :].sum(), 'correct_count': cm[i, i]
            }
        return class_metrics, cm

    def save_accuracy_report(self, class_metrics, name, overall_acc, overall_kappa, overall_f1):
        df_data = []
        for c_name, m in class_metrics.items():
            df_data.append({
                '类别': c_name, '样本数': m['sample_count'], '正确数': m['correct_count'],
                '精度': f"{m['total_accuracy']:.4f}", 'Kappa': f"{m['kappa']:.4f}", 'F1': f"{m['f1_score']:.4f}"
            })
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(self.output_dir, f"{name.lower()}_accuracy.csv"), index=False, encoding='utf-8-sig')

        report_path = os.path.join(self.output_dir, f"{name.lower()}_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{name} Results\nOA: {overall_acc:.4f}, Kappa: {overall_kappa:.4f}, F1: {overall_f1:.4f}\n\n")
            f.write(df.to_string())

        return report_path

    def predict_image(self, classifier, name, scaler=None):
        self.log_signal.emit(f"开始 {name} 全图预测...")
        with rasterio.open(self.image_path) as src:
            profile = src.profile.copy()
            image = src.read()
            image = np.transpose(image, (1, 2, 0))  # H, W, C
            h, w = image.shape[:2]
            result = np.zeros((h, w), dtype=np.uint8)

            block_size = 512
            total_blocks = ((h + block_size - 1) // block_size) * ((w + block_size - 1) // block_size)
            processed_blocks = 0

            for y in range(0, h, block_size):
                end_y = min(y + block_size, h)
                for x in range(0, w, block_size):
                    end_x = min(x + block_size, w)

                    block = image[y:end_y, x:end_x, :]
                    block_pixels = block.reshape(-1, block.shape[-1])

                    if scaler:
                        block_pixels = scaler.transform(block_pixels)

                    block_pred = classifier.predict(block_pixels)
                    result[y:end_y, x:end_x] = block_pred.reshape(end_y - y, end_x - x)

                    processed_blocks += 1
                    # 更新进度条
                    # 为了简化，只发送信号

        # 保存TIF
        profile.update(count=1, dtype='uint8')
        tif_path = os.path.join(self.output_dir, f"{name.lower()}_result.tif")
        with rasterio.open(tif_path, 'w', **profile) as dst:
            dst.write(result[np.newaxis, :, :])

        # 保存可视化PNG
        colors = [[0, 0, 0], [128, 128, 128], [0, 0, 255], [0, 255, 0], [255, 255, 0]]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(1, 5):
            vis[result == i] = colors[i]
        png_path = os.path.join(self.output_dir, f"{name.lower()}_vis.png")
        cv2.imwrite(png_path, vis)

        return result, png_path

    def run(self):
        try:
            results_storage = {}
            self.log_signal.emit("正在读取影像和矢量文件...")
            self.progress_signal.emit(5)

            # 1. 提取样本
            with rasterio.open(self.image_path) as src:
                image_array = src.read()
                image_array = np.transpose(image_array, (1, 2, 0))

            gdf = gpd.read_file(self.shp_path)
            class_dict = {'surface': 1, 'water': 2, 'plant': 3, 'land': 4}
            all_samples = []
            all_labels = []

            self.log_signal.emit(f"正在提取样本点 (共 {len(gdf)} 个多边形)...")

            count = 0
            total_polys = len(gdf)
            for idx, row in gdf.iterrows():
                class_name = row['CLASS_NAME']
                if class_name not in class_dict: continue

                class_id = class_dict[class_name]
                geometry = row.geometry
                minx, miny, maxx, maxy = geometry.bounds
                col_start = max(0, int(np.floor(minx)))
                col_end = min(image_array.shape[1], int(np.ceil(maxx)))
                row_start = max(0, int(np.floor(miny)))
                row_end = min(image_array.shape[0], int(np.ceil(maxy)))

                for r in range(row_start, row_end):
                    for c in range(col_start, col_end):
                        point = Point(c + 0.5, r + 0.5)
                        if geometry.contains(point):
                            pixel_value = image_array[r, c]
                            all_samples.append(pixel_value)
                            all_labels.append(class_id)

                count += 1
                if count % 10 == 0:
                    prog = 5 + int((count / total_polys) * 15)
                    self.progress_signal.emit(prog)

            all_samples = np.array(all_samples)
            all_labels = np.array(all_labels)

            if len(all_samples) == 0:
                raise ValueError("未提取到任何样本，请检查Shapefile与影像是否坐标匹配。")

            self.log_signal.emit(f"样本提取完成。总样本数: {len(all_samples)}")

            indices = np.random.RandomState(42).permutation(len(all_samples))
            all_samples = all_samples[indices]
            all_labels = all_labels[indices]

            X_train, X_val, y_train, y_val = train_test_split(
                all_samples, all_labels, test_size=0.3, random_state=42, stratify=all_labels
            )
            self.progress_signal.emit(20)

            total_stages = len(self.selected_models)
            current_stage = 0

            # 2. 运行各个分类器

            # Fisher
            if 'Fisher' in self.selected_models:
                self.log_signal.emit("=== 正在训练 Fisher 分类器 ===")
                clf = FisherClassifier()
                clf.train(X_train, y_train)
                y_pred = clf.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                kappa = cohen_kappa_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                class_metrics, _ = self.calculate_class_accuracy(y_val, y_pred, class_dict)

                report_path = self.save_accuracy_report(class_metrics, "Fisher", acc, kappa, f1)
                res_img, png_path = self.predict_image(clf, "Fisher")

                results_storage['fisher'] = {
                    'acc': acc, 'kappa': kappa, 'f1': f1, 'png': png_path, 'report': report_path,
                    'class_metrics': class_metrics, 'result_matrix': res_img
                }
                current_stage += 1
                self.progress_signal.emit(20 + int(current_stage / total_stages * 70))

            # Bayes
            if 'Bayes' in self.selected_models:
                self.log_signal.emit("=== 正在训练 Bayes 分类器 ===")
                clf = BayesianClassifier()
                clf.train(X_train, y_train)
                y_pred = clf.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                kappa = cohen_kappa_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                class_metrics, _ = self.calculate_class_accuracy(y_val, y_pred, class_dict)

                report_path = self.save_accuracy_report(class_metrics, "Bayes", acc, kappa, f1)
                res_img, png_path = self.predict_image(clf, "Bayes")

                results_storage['bayes'] = {
                    'acc': acc, 'kappa': kappa, 'f1': f1, 'png': png_path, 'report': report_path,
                    'class_metrics': class_metrics, 'result_matrix': res_img
                }
                current_stage += 1
                self.progress_signal.emit(20 + int(current_stage / total_stages * 70))

            # SVM
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            if 'SVM' in self.selected_models:
                self.log_signal.emit("=== 正在训练 SVM 分类器 ===")
                clf = svm.SVC(gamma='scale', cache_size=1000, C=10)
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_val_scaled)
                acc = accuracy_score(y_val, y_pred)
                kappa = cohen_kappa_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                class_metrics, _ = self.calculate_class_accuracy(y_val, y_pred, class_dict)

                report_path = self.save_accuracy_report(class_metrics, "SVM", acc, kappa, f1)
                res_img, png_path = self.predict_image(clf, "SVM", scaler)

                results_storage['svm'] = {
                    'acc': acc, 'kappa': kappa, 'f1': f1, 'png': png_path, 'report': report_path,
                    'class_metrics': class_metrics, 'result_matrix': res_img
                }
                current_stage += 1
                self.progress_signal.emit(20 + int(current_stage / total_stages * 70))

            # BP神经网络
            if 'BP' in self.selected_models:
                self.log_signal.emit("=== 正在训练 BP 神经网络 ===")
                mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                                    learning_rate_init=0.001, solver='adam')
                mlp.fit(X_train_scaled, y_train)
                y_pred = mlp.predict(X_val_scaled)
                acc = accuracy_score(y_val, y_pred)
                kappa = cohen_kappa_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                class_metrics, _ = self.calculate_class_accuracy(y_val, y_pred, class_dict)

                report_path = self.save_accuracy_report(class_metrics, "BP", acc, kappa, f1)
                res_img, png_path = self.predict_image(mlp, "BP", scaler)

                results_storage['bp'] = {
                    'acc': acc, 'kappa': kappa, 'f1': f1, 'png': png_path, 'report': report_path,
                    'class_metrics': class_metrics, 'result_matrix': res_img
                }
                current_stage += 1
                self.progress_signal.emit(20 + int(current_stage / total_stages * 70))

            # 3. 汇总对比
            self.log_signal.emit("正在生成汇总报告...")
            self.generate_comparison(results_storage)

            self.progress_signal.emit(100)
            self.finished_signal.emit(results_storage)

        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))

    def generate_comparison(self, results_dict):
        # 1. 文本对比
        comp_path = os.path.join(self.output_dir, "accuracy_summary.csv")
        data = []
        for name, res in results_dict.items():
            data.append({
                'Model': name.upper(),
                'Accuracy': f"{res['acc']:.4f}",
                'Kappa': f"{res['kappa']:.4f}",
                'F1': f"{res['f1']:.4f}"
            })
        pd.DataFrame(data).to_csv(comp_path, index=False)

        # 2. 绘图对比
        num = len(results_dict)
        if num > 0:
            cols = 2 if num > 1 else 1
            rows = (num + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

            # 无论 subplots 返回的是单个对象还是数组，统一转为扁平的一维数组/列表
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]

            cmap = mpl.colormaps['tab10'].resampled(5)

            # 遍历每一个结果进行绘图
            for idx, (name, res) in enumerate(results_dict.items()):
                ax = axes[idx]
                ax.imshow(res['result_matrix'], cmap=cmap, vmin=0, vmax=4)
                ax.set_title(f"{name.upper()}")
                ax.axis('off')

            for i in range(num, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "comparison.png"), dpi=150)
            plt.close()


# ==========================================
# 主界面类
# ==========================================
def get_application_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("遥感影像智能分类系统v1.0.1")
        self.resize(1000, 800)

        # 默认路径
        self.img_path = ""
        self.shp_path = ""
        self.out_dir = get_application_path()

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 输入文件区
        grp_input = QGroupBox("输入数据设置")
        layout_input = QVBoxLayout()

        # 影像
        l1 = QHBoxLayout()
        self.line_img = QLineEdit()
        self.line_img.setPlaceholderText("选择遥感影像 (.tif)")
        btn_img = QPushButton("浏览...")
        btn_img.clicked.connect(self.select_image)
        l1.addWidget(QLabel("遥感影像:"))
        l1.addWidget(self.line_img)
        l1.addWidget(btn_img)
        layout_input.addLayout(l1)

        # 样本
        l2 = QHBoxLayout()
        self.line_shp = QLineEdit()
        self.line_shp.setPlaceholderText("选择样本矢量 (.shp)")
        btn_shp = QPushButton("浏览...")
        btn_shp.clicked.connect(self.select_shp)
        l2.addWidget(QLabel("训练样本:"))
        l2.addWidget(self.line_shp)
        l2.addWidget(btn_shp)
        layout_input.addLayout(l2)

        # 输出
        l3 = QHBoxLayout()
        self.line_out = QLineEdit(self.out_dir)
        btn_out = QPushButton("浏览...")
        btn_out.clicked.connect(self.select_out)
        l3.addWidget(QLabel("输出目录:"))
        l3.addWidget(self.line_out)
        l3.addWidget(btn_out)
        layout_input.addLayout(l3)

        grp_input.setLayout(layout_input)
        main_layout.addWidget(grp_input)

        # 2. 模型选择区
        grp_model = QGroupBox("分类模型选择")
        layout_model = QHBoxLayout()
        self.chk_fisher = QCheckBox("Fisher")
        self.chk_fisher.setChecked(True)
        self.chk_bayes = QCheckBox("Bayes")
        self.chk_bayes.setChecked(True)
        self.chk_svm = QCheckBox("SVM")
        self.chk_svm.setChecked(True)
        self.chk_bp = QCheckBox("BP神经网络")
        self.chk_bp.setChecked(True)

        layout_model.addWidget(self.chk_fisher)
        layout_model.addWidget(self.chk_bayes)
        layout_model.addWidget(self.chk_svm)
        layout_model.addWidget(self.chk_bp)
        grp_model.setLayout(layout_model)
        main_layout.addWidget(grp_model)

        # 3. 运行按钮
        self.btn_run = QPushButton("开始分类")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.btn_run.clicked.connect(self.start_processing)
        main_layout.addWidget(self.btn_run)

        # 4. 进度与日志区
        splitter = QSplitter(Qt.Vertical)

        # 日志
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("运行日志将显示在这里...")
        splitter.addWidget(self.txt_log)

        # 结果预览 Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(QLabel("等待运行结果..."), "预览")
        splitter.addWidget(self.tabs)

        main_layout.addWidget(splitter)

        # 进度条
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        main_layout.addWidget(self.pbar)

        # 5. 底部状态栏
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择影像", self.out_dir, "TIFF Files (*.tif)")
        if path:
            self.line_img.setText(path)

    def select_shp(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择Shapefile", self.out_dir, "Shapefile (*.shp)")
        if path:
            self.line_shp.setText(path)

    def select_out(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.out_dir)
        if path:
            self.line_out.setText(path)

    def log(self, msg):
        self.txt_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        # 滚动到底部
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_processing(self):
        img = self.line_img.text()
        shp = self.line_shp.text()
        out = self.line_out.text()

        if not (os.path.exists(img) and os.path.exists(shp) and os.path.exists(out)):
            QMessageBox.warning(self, "错误", "请检查文件路径是否存在！")
            return

        models = []
        if self.chk_fisher.isChecked(): models.append('Fisher')
        if self.chk_bayes.isChecked(): models.append('Bayes')
        if self.chk_svm.isChecked(): models.append('SVM')
        if self.chk_bp.isChecked(): models.append('BP')

        if not models:
            QMessageBox.warning(self, "提示", "请至少选择一个模型！")
            return

        # 禁用UI
        self.btn_run.setEnabled(False)
        self.txt_log.clear()
        self.tabs.clear()
        self.pbar.setValue(0)

        self.worker = ProcessingThread(img, shp, out, models)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.pbar.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)

        self.worker.start()
        self.log("任务线程已启动...")

    def on_error(self, err_msg):
        self.log(f"错误: {err_msg}")
        QMessageBox.critical(self, "运行出错", err_msg)
        self.btn_run.setEnabled(True)

    def on_finished(self, results):
        self.log("任务全部完成！")
        self.btn_run.setEnabled(True)
        self.status_label.setText("完成")
        self.display_results(results)

    def display_results(self, results):
        self.tabs.clear()

        # 1. 总体对比图
        comp_img_path = os.path.join(self.line_out.text(), "comparison.png")
        if os.path.exists(comp_img_path):
            lbl = QLabel()
            pix = QPixmap(comp_img_path)
            if not pix.isNull():
                lbl.setPixmap(pix.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                lbl.setAlignment(Qt.AlignCenter)
                self.tabs.addTab(lbl, "综合对比")

        # 2. 各个模型的分页
        for model_key, data in results.items():
            model_name = model_key.upper()

            tab_widget = QWidget()
            layout = QHBoxLayout(tab_widget)

            # 左侧：图片
            lbl_img = QLabel()
            pix = QPixmap(data['png'])
            if not pix.isNull():
                lbl_img.setPixmap(pix.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lbl_img.setAlignment(Qt.AlignCenter)

            # 右侧：精度文本
            txt_info = QTextEdit()
            txt_info.setReadOnly(True)
            with open(data['report'], 'r', encoding='utf-8') as f:
                txt_info.setText(f.read())

            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(lbl_img)
            splitter.addWidget(txt_info)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)

            layout.addWidget(splitter)
            self.tabs.addTab(tab_widget, model_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())