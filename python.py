from hyperopt import STATUS_OK
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from hyperopt import hp, fmin, tpe, space_eval
import plotly.express as px
import plotly.graph_objs as go

# 数据预处理函数
def preprocess_data(data):
    features = data[['B1C1', 'B1C2', 'B1C3', 'B2C1', 'B2C2', 'B2C3', 'B3C1', 'B3C2', 'B3C3']].copy()
    labels = data['物流行业经济适应度'].copy()

    features.dropna(inplace=True)
    labels.dropna(inplace=True)

    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    return train_test_split(features, labels, test_size=0.2, random_state=42)


# 层次分析法（AHP）相关函数
def validate_user_input(user_input):
    try:
        numbers = []

        for num_str in user_input.split(','):
            if '/' in num_str:
                numerator, denominator = map(int, num_str.split('/'))
                numbers.append(numerator / denominator)
            else:
                numbers.append(int(num_str))

        if len(numbers) != 9:
            raise ValueError("每行应输入9个数值（整数或分数），以逗号分隔")

        return numbers

    except ValueError as e:
        st.error(f"输入格式错误: {e}")
        raise


def get_user_matrix():
    user_matrix = []

    for i in range(3):
        row = st.text_input(f"请输入第{i+1}行的3个数字，用逗号分隔:")
        if row:
            user_matrix.append(validate_user_input(row))

    if len(user_matrix) == 3:
        user_matrix = np.array(user_matrix)
        return user_matrix
    else:
        return None  # 如果用户还没有输入完所有的数据，就返回None


def calculate_weights(criteria_judgment_matrix):
    eig_vals, eig_vecs = np.linalg.eig(criteria_judgment_matrix)
    max_eig_val_index = np.argmax(eig_vals.real)
    weight_vector = eig_vecs[:, max_eig_val_index].real
    weight_vector /= np.sum(weight_vector)
    return weight_vector

# 模型训练相关函数
def objective_function(params, x_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=params['alpha'], learning_rate_init=params['learning_rate_init'], early_stopping=True, random_state=42)
    model.fit(x_train, y_train)
    mse = mean_squared_error(y_train, model.predict(x_train))
    return {'loss': mse, 'status': STATUS_OK}

def train_model(x_train, y_train, x_test, y_test):
    space = {
        'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(1)),
        'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.0001), np.log(1)),
    }

    best_params = fmin(fn=lambda params: objective_function(params, x_train, y_train), space=space, algo=tpe.suggest, max_evals=500)
    optimized_params = space_eval(space, best_params)

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

    return fig1, fig2, model, mse

def main():
    st.title("数据分析与模型训练")

    uploaded_file = st.file_uploader("选择CSV数据文件", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.header("输入层次分析矩阵")
        user_matrix = get_user_matrix()  # 仅获取用户矩阵
        if user_matrix is not None:
            weights = calculate_weights(user_matrix)  # 单独计算权重
            weights_df = pd.DataFrame(weights, index=data.columns[:9], columns=['weight'])
            x_train, x_test, y_train, y_test = preprocess_data(data)
            x_train = x_train * weights_df.T.values
            x_test = x_test * weights_df.T.values

            fig1, fig2, model, mse = train_model(x_train, y_train, x_test, y_test)

            st.subheader("模型训练结果")
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)

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
