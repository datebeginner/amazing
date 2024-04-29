from hyperopt import STATUS_OK, fmin, tpe, hp, Trials
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# 层次分析法（AHP）相关函数

def validate_user_input(user_input):
    try:
        numbers = list(map(float, user_input.split(',')))
        if len(numbers) != 3:
            raise ValueError("每行应输入3个数字，用逗号分隔")
        return numbers
    except ValueError as e:
        st.error(f"输入格式错误: {e}")
        raise

def check_consistency(matrix):
    weights = np.mean(matrix / matrix.sum(axis=0), axis=1)
    cr = np.max(np.abs(np.dot(matrix, weights) - np.einsum('ij,j->i', matrix, weights))) / (len(matrix) - 1)
    if cr > 0.1:
        return False, None
    return True, weights

def get_user_matrices():
    # 初始化计数器以生成唯一键
    criteria_row_counter = 0
    within_criteria_counters = {criterion: 0 for criterion in ['B1', 'B2', 'B3']}

    # 收集准则层间的比较矩阵
    criteria_matrices = []
    for _ in range(3):
        unique_key = f"criterion_row_{criteria_row_counter}"
        row = st.text_input(f"请输入准则层间比较矩阵的一行数字，用逗号分隔:", key=unique_key)
        if row:
            criteria_matrices.append(row)
        criteria_row_counter += 1

    # 收集同一准则层下自变量的比较矩阵
    within_criteria_matrices = {criterion: [] for criterion in ['B1', 'B2', 'B3']}
    for criterion in ['B1', 'B2', 'B3']:
        for _ in range(3):
            unique_key = f"{criterion}_row_{within_criteria_counters[criterion]}"
            row = st.text_input(f"请输入{criterion}下自变量比较矩阵的一行数字，用逗号分隔:", key=unique_key)
            if row:
                within_criteria_matrices[criterion].append(row)
            within_criteria_counters[criterion] += 1

    # 其余逻辑保持不变...
    # ...

    weights_within_criteria = []
    for matrix in within_criteria_matrices:
        consistent, weights = check_consistency(matrix)
        if not consistent:
            st.warning(f"{criterion}下自变量比较矩阵的一致性比率大于0.1，请重新输入。")
            return None, None
        weights_within_criteria.append(weights)

    return criteria_matrix, within_criteria_matrices, weights_criteria, weights_within_criteria

# 数据预处理函数

def preprocess_data(data):
    features = data[['B1C1', 'B1C2', 'B2C1', 'B2C2', 'B3C1', 'B3C2', 'B3C3']].copy()
    labels = data['物流行业经济适应度'].copy()

    features.dropna(inplace=True)
    labels.dropna(inplace=True)

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    features = pd.DataFrame(features_scaled, columns=features.columns)

    return train_test_split(features, labels, test_size=0.2, random_state=42)

# 模型训练相关函数

def objective_function(params, x_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100,), 
                         activation='relu', 
                         solver='adam', 
                         alpha=params['alpha'], 
                         learning_rate_init=params['learning_rate_init'], 
                         early_stopping=True, 
                         max_iter=1000, 
                         random_state=42)
    model.fit(x_train, y_train)
    mse = mean_squared_error(y_train, model.predict(x_train))
    return {'loss': mse, 'status': STATUS_OK, 'model': model}

def train_model(x_train, y_train, x_test, y_test):
    space = {
        'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(1)),
        'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.0001), np.log(1)),
    }

    trials = Trials()
    best = fmin(fn=lambda params: objective_function(params, x_train, y_train), 
                space=space, 
                algo=tpe.suggest, 
                max_evals=500, 
                trials=trials)

    optimized_params = space_eval(space, best)

    model = MLPRegressor(hidden_layer_sizes=(100,), 
                         activation='relu', 
                         solver='adam', 
                         alpha=optimized_params['alpha'], 
                         learning_rate_init=optimized_params['learning_rate_init'], 
                         early_stopping=True, 
                         random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    # 绘制模型评估结果
    fig1 = st.line_chart({'Actual': y_test.tolist(), 'Predicted': y_pred.tolist()})
    st.write(f"模型均方误差（MSE）: {mse}")

    return model

def main():
    st.title("数据分析与模型训练")

    uploaded_file = st.file_uploader("选择CSV数据文件", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.header("输入层次分析矩阵")
        criteria_matrix, within_criteria_matrices, weights_criteria, weights_within_criteria = get_user_matrices()
        if all(w is not None for w in [weights_criteria] + weights_within_criteria):
            # 应用权重到特征
            for col, weight in zip(data.columns, weights_within_criteria[0]):
                data[col] *= weight

            x_train, x_test, y_train, y_test = preprocess_data(data)

            # 训练模型
            model = train_model(x_train, y_train, x_test, y_test)

            st.header("使用模型进行预测")
            user_input = st.text_input("输入预测数据（以逗号分隔的数值）")
            if user_input:
                try:
                    user_input = validate_user_input(user_input)
                    prediction = model.predict([user_input])
                    st.write(f"预测结果：{prediction[0]}")
                except ValueError:
                    st.write("输入数据格式不正确，请输入逗号分隔的数值。")

if __name__ == "__main__":
    main()
