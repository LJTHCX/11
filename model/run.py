import os
import sys
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 以上为依赖包引入部分, 请根据实际情况引入
# 引入的包需要安装的, 请在requirements.txt里列明, 最好请列明版本

# 以下为逻辑函数, main函数的入参和最终的结果输出不可修改
def main(to_pred_dir, result_save_path):
    # 定义路径和模型目录
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)

    to_pred_dir = os.path.abspath(to_pred_dir)
    testa_csv_path = os.path.join(to_pred_dir, "testa_x", "testa_x.csv")

    # 加载数据
    def load_data(csv_path):
        data = pd.read_csv(csv_path)
        texts = data['Title'].astype(str).values
        labels = data['label'].values
        return data, texts, labels

    # 数据预处理
    def preprocess_data(texts, max_features=5000):
        vectorizer = TfidfVectorizer(max_features=max_features)
        data = vectorizer.fit_transform(texts).toarray()
        return data, vectorizer

    # 构建模型
    def build_model(input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # 加载和处理数据
    testa, texts, labels = load_data(testa_csv_path)
    data, vectorizer = preprocess_data(texts)

    # 分割数据集
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # 构建和训练模型
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # 生成预测
    predictions = model.predict(data)
    predictions = (predictions > 0.5).astype(int).flatten()
    testa['label'] = predictions  # 将预测标签填入数据集

    # 提取预测结果
    test_result = testa[["id", "label"]]

    # 结果输出到result_save_path
    test_result.to_csv(result_save_path, index=None)

if __name__ == "__main__":
    # 以下代码请勿修改, 若因此造成的提交失败由选手自负
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)
