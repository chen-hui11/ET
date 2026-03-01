import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components


st.set_page_config(page_title="胰腺癌术后早期肝转移预测系统", layout="centered")
st.title("胰腺癌术后早期肝转移预测系统")


@st.cache_resource
def load_model():
    return joblib.load("best_model_et_bayes.pkl")


@st.cache_data
def load_test():
    return pd.read_excel("test.xlsx")


model = load_model()
X_test = load_test()

feature_names = ["Chemotherapy", "AV.invasion", "Grade", "HBV", "LYM", "ALB", "CA199", "T", "AKP", "Tumor.size"]

# ----------- UI 输入 -----------
Chemotherapy = st.selectbox("化疗", options=[0, 1], format_func=lambda x: "有" if x == 1 else "无")
AV_invasion = st.selectbox("腹腔干/门静脉侵犯", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
Grade = st.selectbox("分化程度", options=[0, 1], format_func=lambda x: "低分化" if x == 1 else "中高分化")
HBV = st.selectbox("乙肝", options=[0, 1], format_func=lambda x: "有" if x == 1 else "无")

LYM = st.number_input("淋巴细胞计数", min_value=0.0, max_value=10.0, value=3.0)
ALB = st.number_input("白蛋白", min_value=10.0, max_value=50.0, value=33.0)
CA199 = st.number_input("CA199", min_value=0.0, max_value=5000.0, value=120.0)
AKP = st.number_input("碱性磷酸酶", min_value=0.0, max_value=2000.0, value=200.0)

T = st.selectbox("T 分期", options=[1, 2, 3, 4], format_func=lambda x: f"T{x}")
Tumor_size = st.number_input("肿瘤大小（cm）", min_value=0.0, max_value=20.0, value=3.0)

feature_values = [Chemotherapy, AV_invasion, Grade, HBV, LYM, ALB, CA199, T, AKP, Tumor_size]
x_input = np.array(feature_values, dtype=float).reshape(1, -1)
df_input = pd.DataFrame([feature_values], columns=feature_names)

# ----------- 预测按钮 -----------
if st.button("Predict"):
    # 预测
    predicted_class = int(model.predict(x_input)[0])
    proba = model.predict_proba(x_input)[0]  # shape: (2,)
    risk = float(proba[predicted_class]) * 100.0

    st.write(f"**Predicted Class:** {predicted_class}（1：高风险，0：低风险）")
    st.write(f"**Predicted Probabilities:** {proba}")

    if predicted_class == 1:
        st.warning(f"模型预测为 **高风险**，对应概率约 **{risk:.1f}%**。建议结合临床情况进一步评估。")
    else:
        st.success(f"模型预测为 **低风险**，对应概率约 **{risk:.1f}%**。建议规律随访。")

    # ----------- SHAP 解释 -----------
    st.subheader("SHAP Explanation (Force Plot)")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)

        # 兼容不同 shap 版本返回结构：
        # - 有的返回 list（分类）：[class0_array, class1_array]
        # - 有的返回 ndarray：(n, features, classes)
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class]  # (1, features)
            ev = explainer.expected_value[predicted_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            # ndarray: (1, features, classes)
            sv = shap_values[:, :, predicted_class]
            ev = explainer.expected_value[predicted_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

        plt.figure()
        shap.force_plot(ev, sv, df_input, matplotlib=True, show=False)
        st.pyplot(plt.gcf())
        plt.close()

    except Exception as e:
        st.error("SHAP 绘图失败（可能是 shap 版本/模型类型导致的兼容问题）。")
        st.exception(e)

    # ----------- LIME 解释 -----------
    st.subheader("LIME Explanation")

    try:
        # 如果 test.xlsx 的列和 feature_names 不一致，优先用 feature_names 的顺序
        use_named_cols = all(c in X_test.columns for c in feature_names)
        training_data = X_test[feature_names].values if use_named_cols else X_test.values
        lime_feature_names = feature_names if use_named_cols else X_test.columns.tolist()

        lime_explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=lime_feature_names,
            class_names=["No Metastasis", "Metastasis"],
            mode="classification"
        )

        exp = lime_explainer.explain_instance(
            data_row=x_input.flatten(),
            predict_fn=model.predict_proba,
            num_features=min(10, x_input.shape[1])
        )

        lime_html = exp.as_html(show_table=False)
        components.html(lime_html, height=700, scrolling=True)

    except Exception as e:
        st.error("LIME 解释失败（常见原因：特征列不一致/输入维度不匹配）。")
        st.exception(e)
