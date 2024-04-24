from hyperopt import STATUS_OK
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from hyperopt import hp, fmin, tpe, space_eval
import shap

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

    fig1, ax1 = plt.subplots()
    ax1.plot(y_test, label='Actual')
    ax1.plot(y_pred, label='Predicted')
    ax1.set_title(f"Model Evaluation\nMSE: {mse:.4f}")
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.hist(x_train, bins=20, alpha=0.5, label='Train')
    ax2.hist(x_test, bins=20, alpha=0.5, label='Test')
    ax2.set_title("Training and Test Data Distribution")
    ax2.legend()

    # 在这里定义 model_predict 函数
    def model_predict(data):
        return model.predict(data)

    return fig1, fig2, model, mse, model_predict  # 更新返回值，添加 model_predict

# Streamlit应用主体
from hyperopt import STATUS_OK
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from hyperopt import hp, fmin, tpe, space_eval
import shap

# 定义数据预处理函数等...

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
            st.pyplot(fig1)
            st.pyplot(fig2)

            # 显示模型评估指标（如需要，可以添加其他评估指标）
            st.write(f"测试集MSE: {mse:.4f}")

            # 添加新的图形
            st.subheader("特征重要性")
            masker = shap.maskers.Independent(data=x_test, max_samples=x_test.shape[0])
            explainer = shap.Explainer(model_predict, masker)
            shap_values = explainer(x_test)
            fig_importance, ax_importance = plt.subplots()
            shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
            plt.show()
            st.pyplot(fig_importance)

            st.subheader("散点图")
            fig_scatter, ax_scatter = plt.subplots()
            ax_scatter.scatter(data['B1C1'], data['B1C2'])
            ax_scatter.set_xlabel('B1C1')
            ax_scatter.set_ylabel('B1C2')
            ax_scatter.set_title('B1C1 vs B1C2')
            st.pyplot(fig_scatter)
        else:
            st.warning("请先输入完整的层次分析矩阵以进行模型训练和评估.")
    else:
        st.info("请上传CSV数据文件以开始分析.")

if __name__ == "__main__":
    main()
