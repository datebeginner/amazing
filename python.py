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
    features = data[['B1C1', 'B1C2', 'B2C1', 'B2C2', 'B2C3', 'B3C1', 'B3C2']].copy()
    labels = data['物流行业经济适应度'].copy()

    features.dropna(inplace=True)
    labels.dropna(inplace=True)

    scaler = MinMaxScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    return train_test_split(features, labels, test_size=0.2, random_state=42)


# 层次分析法（AHP）相关函数
def validate_user_input(user_input):
    try:
        numbers = list(map(float, user_input.split(',')))
        if len(numbers) != 7:
            raise ValueError("每行应输入7个数字，用逗号分隔")
        return numbers
    except ValueError as e:
        st.error(f"输入格式错误: {e}")
        raise

def get_user_matrix():
    user_matrix = []
    for i in range(7):
        row = st.text_input(f"请输入第{i+1}行的7个数字，用逗号分隔:")
        if row:  # 只有当用户输入了数据后才添加到列表中
            user_matrix.append(validate_user_input(row))

    if len(user_matrix) == 7:  # 只有当用户输入了所有7行数据后才进行处理
        user_matrix = np.array(user_matrix)
        consistent, weights = check_consistency(user_matrix)
        if consistent:
            return user_matrix, weights
        else:
            st.warning("一致性比率大于0.1，请重新输入比较矩阵。")
            return None, None
    else:
        return None, None  # 如果用户还没有输入完所有的数据，就返回None, None

def check_consistency(matrix):
    weights = np.mean(matrix / matrix.sum(axis=0), axis=1)
    cr = np.max(np.abs(np.dot(matrix, weights) - np.sum(weights))) / (len(matrix) - 1)
    if cr > 0.1:
        return False, None
    return True, weights

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

    # 修改模型评估图表
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual'))
    fig1.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted'))
    fig1.update_layout(title_text=f"Model Evaluation<br>MSE: {mse:.4f}")

    # 修改训练和测试数据分布图表
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=x_train, name='Train', nbinsx=20, opacity=0.5))
    fig2.add_trace(go.Histogram(x=x_test, name='Test', nbinsx=20, opacity=0.5))
    fig2.update_layout(title_text="Training and Test Data Distribution", barmode='overlay')

    return fig1, fig2, model, mse, model_predict  # 更新返回值，添加 model_predict

# Streamlit应用主体
def main():
    st.title("数据分析与模型训练")

    # 加载数据
    uploaded_file = st.file_uploader("选择CSV数据文件", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # 输入层次分析矩阵
        st.header("输入层次分析矩阵")
        user_matrix, weights = get_user_matrix()
        if weights is not None:
            weights_df = pd.DataFrame(weights, index=data.columns[:-1], columns=['weight'])
            x_train, x_test, y_train, y_test = preprocess_data(data)
            x_train = x_train * weights_df.T.values
            x_test = x_test * weights_df.T.values

            # 训练模型
            fig1, fig2, model, mse, model_predict = train_model(x_train, y_train, x_test, y_test)

            # 可视化结果
            st.subheader("模型训练结果")
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)

   # 显示模型评估指标（如需要，可以添加其他评估指标）
            st.write(f"均方误差（MSE）: {mse}")

            # 使用模型进行预测
            st.header("使用模型进行预测")
            user_input = st.text_input("输入预测数据（以逗号分隔的数值）")
            if user_input:
                try:
                    user_input = validate_user_input(user_input)
                    user_input = user_input * weights_df.T.values
                    prediction = model_predict(user_input)
                    st.write(f"预测结果：{prediction}")
                except ValueError:
                    st.write("输入数据格式不正确，请输入逗号分隔的数值。")

if __name__ == "__main__":
    main()
