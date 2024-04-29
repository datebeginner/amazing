from hyperopt import STATUS_OK
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from hyperopt import hp, fmin, tpe, space_eval
import plotly.express as px
import plotly.graph_objs as go
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def preprocess_data(data):
    features = data[['B1C1', 'B1C2', 'B2C1', 'B2C2', 'B3C1', 'B3C2', 'B3C3']].copy()
    labels = data['物流行业经济适应度'].copy()

    features.dropna(inplace=True)
    labels.dropna(inplace=True)

    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    return train_test_split(features, labels, test_size=0.2, random_state=42)

# 修改get_user_matrix函数以适应四个不同的比较矩阵
def get_user_matrices():
    # 准则层之间的比较矩阵
    criteria_matrix = []
    for i in range(3):
        row = st.text_input(f"请输入准则层间比较矩阵第{i+1}行的3个数字，用逗号分隔:")
        if row:
            try:
                criteria_matrix.append(validate_user_input(row))
            except ValueError as e:
                st.error(f"输入格式错误: {e}")
                return None
    
within_criteria_matrices = []
    for criterion in ['B1', 'B2', 'B3']:
        matrix = []
        for i in range(3):
            row = st.text_input(f"请输入{criterion}下自变量比较矩阵第{i+1}行的3个数字，用逗号分隔:")
            if row:
                try:
                    matrix.append(validate_user_input(row))
                except ValueError as e:
                    st.error(f"输入格式错误: {e}")
                    return None
        within_criteria_matrices.append(np.array(matrix))
    
    # 一致性检验
    consistent_criteria, weights_criteria = check_consistency(np.array(criteria_matrix))
    consistent_within = all(check_consistency(matrix)[0] for matrix in within_criteria_matrices)
    
    if consistent_criteria and consistent_within:
        return user_matrix, weights
    else:
        st.warning("一致性比率大于0.1，请重新输入比较矩阵。")
        return None, None

        
def check_consistency(matrix):
    weights = np.mean(matrix / matrix.sum(axis=0), axis=1)
    cr = np.max(np.abs(np.dot(matrix, weights) - np.sum(weights))) / (len(matrix) - 1)
    if cr > 0.1:
        return False, None
    return True, weights

# 模型训练相关函数
def objective_function(params, x_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=params['alpha'], learning_rate_init=params['learning_rate_init'], early_stopping=True, max_iter=1, warm_start=True, random_state=42)
    model.fit(x_train, y_train)
    mse = mean_squared_error(y_train, model.predict(x_train))
    return {'loss': mse, 'status': STATUS_OK, 'model': model}

def train_model(x_train, y_train, x_test, y_test):
    space = {
        'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(1)),
        'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.0001), np.log(1)),
    }

    trials = Trials()
    best = fmin(fn=lambda params: objective_function(params, x_train, y_train), space=space, algo=tpe.suggest, max_evals=500, trials=trials)

    mse_history = [x['result']['loss'] for x in trials.trials]

    optimized_params = space_eval(space, best)

    model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=optimized_params['alpha'], learning_rate_init=optimized_params['learning_rate_init'], early_stopping=True, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual'))
    fig1.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted'))
    fig1.update_layout(title_text=f"模型评估<br>MSE: {mse:.4f}")

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=x_train, name='Train', nbinsx=20, opacity=0.5))
    fig2.add_trace(go.Histogram(x=x_test, name='Test', nbinsx=20, opacity=0.5))
    fig2.update_layout(title_text="训练和测试数据分布", barmode='overlay')

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=mse_history, mode='lines'))
    fig3.update_layout(title_text="模型优化过程")

    return fig1, fig2, fig3, model, mse
def main():
    st.title("数据分析与模型训练")

    uploaded_file = st.file_uploader("选择CSV数据文件", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.header("输入层次分析矩阵")
        user_matrix, weights = get_user_matrix()
        if weights is not None:
            weights_df = pd.DataFrame(weights, index=data.columns[:-1], columns=['weight'])
            x_train, x_test, y_train, y_test = preprocess_data(data)
            x_train = x_train * weights_df.T.values
            x_test = x_test * weights_df.T.values

            fig1, fig2, fig3, model, mse = train_model(x_train, y_train, x_test, y_test)

            st.subheader("模型训练结果")
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
            st.plotly_chart(fig3)

            st.write(f"均方误差（MSE）: {mse}")

            st.header("使用模型进行预测")
            user_input = st.text_input("输入预测数据（以逗号分隔的数值）")
            if user_input:
                try:
                    user_input = validate_user_input(user_input)
                    prediction = model.predict([user_input])
                    st.write(f"预测结果：{prediction}")
                except ValueError:
                    st.write("输入数据格式不正确，请输入逗号分隔的数值。")

if __name__ == "__main__":
    main()
