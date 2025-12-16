# https://decision-tree-builder.streamlit.app/
# https://github.com/lukestin/decision-tree-builder/blob/main/app.py#L1C1-L246C45
# https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


fm.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = 'Taipei Sans TC Beta'
pd.options.mode.copy_on_write = True




@st.dialog("資料預覽", width='large')
def data_preview(df : pd.DataFrame):
    if st.toggle('Head', True) :
        df = df.head(10)
    st.dataframe(df)


def reset_toggle():
    if st.session_state.preview_toggle:
        data_preview(data)
        st.session_state.preview_toggle = False



st.set_page_config(page_title='檢測多重共線性 version 0.11', page_icon="🌳", layout='centered')

st.header('🌳 檢測多重共線性', divider='gray')
"""
多重共線性是指迴歸模型中兩個或多個預測變數(X)高度相關，以至於難以區分它們各自對因變數(y)的影響。可透過以下方式，檢測多重共線性：
📌領域知識與經驗
📌相關矩陣
📌變異數膨脹因子(VIF)
"""

form_side = st.container()
main_side = st.container()

if 'preview_toggle' not in st.session_state:
    st.session_state.preview_toggle = False

uploaded_file = form_side.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    form_side.toggle('查看檔案', value=st.session_state.preview_toggle, key='preview_toggle', on_change=reset_toggle)
 
    # main_side.divider()
    # accuracy, ineractive_graph, static_image, tree_code, prediction = main_side.tabs(['Accuracy', 'Interactive Graph', 'Satic Image', 'Python Code', 'Prediction'])
    

    with st.sidebar :
        "## 藉由 領域知識與經驗"
        target_column = st.selectbox("選擇Target Column", data.columns, help='選擇用於預測的對象(y)')
        target_df = data[target_column]
    
        initial_feature_columns = [col for col in data.select_dtypes(include=['number']).columns if col != target_column]
        feature_columns = initial_feature_columns
        
        if st.toggle('選擇數值Feature Columns', False, help='若不調整，y之外的數值欄位均用作特徵(X)') :
            feature_columns = st.multiselect("", initial_feature_columns)

        feature_df = data[feature_columns]
        select_df = pd.concat([feature_df, target_df], axis=1)





    # auto_launch = form_side.toggle('Auto launch', False)

    form_side.divider()
    _col1, _col2, _col3 = form_side.columns(3)
    if _col1.button('目前處理資料預覽', type='primary', use_container_width=True) :
        data_preview(select_df)
        # data_preview(target_df)

    if _col2.button('相關矩陣', use_container_width=True) :
        correlation_matrix = data[feature_columns].corr()
        
        fig, ax = plt.subplots()
        ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig)
        # correlation_matrix = data[feature_columns].select_dtypes(include=['number']).corr()
        # plt.figure() 
        # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") 
        # plt.title('Feature Columns的相關矩陣')
        # plt.show()

    if _col3.button("變異數膨脹因子", use_container_width=True):
        st.image("vif.png")
        
        X_vif = add_constant(data[feature_columns])
        vif_data = pd.DataFrame()
        vif_data["數值特徵"] = X_vif.columns
        vif_data["變異數膨脹因子"] = [variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])]

        st.dataframe(vif_data)
