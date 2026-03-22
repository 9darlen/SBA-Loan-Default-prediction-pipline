import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 載入模型
model = joblib.load("best_pipeline.joblib")
now = pd.Timestamp.now()
REF_DATE = pd.Timestamp('1970-01-01')
days_since_appv = (now - REF_DATE).days

st.set_page_config(page_title="SBA 貸款違約預測系統", layout="wide")

# 🔥 新增：頁面切換
page = st.sidebar.radio("功能選單", [
    "📊 貸款風險評估",
    "📈 模型分析"
])

# ================================
# 🟢 PAGE 1：貸款風險評估
# ================================
if page == "📊 貸款風險評估":

    with st.sidebar:
        st.header("🏢 企業基本資訊")
        identify = st.text_input("企業識別碼 (LoanNr_ChkDgt)", value="1234567890")
        name = st.text_input("企業名稱 (Name)", value="Test Corp")
        state = st.selectbox("州別 (State)", ['IN', 'OK', 'FL', 'CT', 'NJ',
                                              'NC', 'IL', 'RI', 'TX', 'VA', 'TN', 'AR', 'MN', 'MO', 'MA',
                                              'CA', 'SC', 'LA', 'IA', 'OH', 'KY', 'MS', 'NY', 'MD', 'PA', 
                                              'OR', 'ME', 'KS', 'MI', 'AK', 'WA', 'CO', 'MT', 'WY', 'UT', 
                                              'NH', 'WV', 'ID', 'AZ', 'NV', 'WI', 'NM', 'GA', 'ND', 'VT', 
                                              'AL', 'NE', 'SD', 'HI', 'DE', 'DC', 'n'])
        bankstate = st.selectbox("銀行所在州別 (BankState)", ['OH', 'IN', 'OK', 'FL', 'DE', 
                                                            'SD', 'AL', 'CT', 'GA', 'OR', 'MN', 'RI', 'NC', 
                                                            'TX', 'MD', 'NY', 'TN', 'SC', 'MS', 'MA', 'LA', 
                                                            'IA', 'VA', 'CA', 'IL', 'KY', 'PA', 'MO', 'WA', 
                                                            'MI', 'UT', 'KS', 'WV', 'WI', 'AZ', 'NJ', 'CO', 
                                                            'ME', 'NH', 'AR', 'ND', 'MT', 'ID', 'WY', 'NM', 
                                                            'DC', 'NV', 'NE', 'PR', 'HI', 'VT', 'AK', 'GU', 'AN', 'EN', 'VI'])
        naics = st.text_input("NAICS 行業代碼 (前六碼)", value="236115")
        new_exist = st.radio("企業類型", options=[1.0, 2.0],
                             format_func=lambda x: "新創" if x == 2 else "現有")

    st.subheader("💰 財務申貸資訊")
    col1, col2 = st.columns(2)

    with col1:
        gr_appv = st.number_input("銀行批准總金額 (GrAppv)", min_value=0, value=50000, step=1000)
        sba_appv = st.number_input("SBA 保證金額 (SBA_Appv)", min_value=0, value=40000, step=1000)
        term = st.number_input("貸款期限 (Term, 月)", min_value=0, value=60, step=0)
        no_emp = st.number_input("員工人數 (NoEmp)", min_value=0, value=10, step=1)

    with col2:
        rev_line = st.selectbox("循環信度 (RevLineCr)", ["Y", "N", "0"])
        low_doc = st.selectbox("低文件程序 (LowDoc)", ["Y", "N", "0"])
        franchise_code = st.text_input("特許經營代碼 (FranchiseCode)", value="0")
        urban_rural = st.selectbox("地區屬性 (UrbanRural)", [0, 1, 2], format_func=lambda x: "未知" if x == 0 else ("都市" if x == 1 else "鄉村"))
    
    if st.button("🔍 開始風險評估"):

        # ⚠️ 改：不要轉字串，保持 numeric
        input_df = pd.DataFrame([{
            "State": state,
            "BankState": bankstate,  
            "NAICS": naics,
            "NewExist": new_exist,
            "GrAppv": gr_appv,
            "SBA_Appv": sba_appv,
            "Term": term,
            "NoEmp": no_emp,  
            "RevLineCr": rev_line,
            "LowDoc": low_doc,
            "FranchiseCode": franchise_code,
            "UrbanRural":   urban_rural,

            "ApprovalDate": now,
            "ApprovalFY": now.year,
            "Days_Since_Appv": days_since_appv,
        }])

        prob = model.predict_proba(input_df)[0][1]

        st.divider()

        # ================================
        # 🎯 決策結果（最重要）
        # ================================
        st.header("📌 核貸建議")

        if prob > 0.7:
            st.error("❌ 不建議核貸")
            decision_level = "high"
        elif prob > 0.3:
            st.warning("⚠️ 需人工審核")
            decision_level = "medium"
        else:
            st.success("✅ 建議核貸")
            decision_level = "low"

        # ================================
        # 📊 KPI
        # ================================
        st.subheader("📊 風險指標")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("違約機率", f"{prob:.2%}")

        with col2:
            expected_loss = prob * gr_appv
            st.metric("預期損失", f"${expected_loss:,.0f}")

        # ================================
        # 🔍 行動建議（🔥關鍵升級）
        # ================================
        st.subheader("💡 建議措施")

        if decision_level == "high":
            st.markdown("""
            - 降低貸款金額
            - 要求擔保品
            - 提高利率
            """)
        elif decision_level == "medium":
            st.markdown("""
            - 補充財務文件
            - 加強信用審查
            """)
        else:
            st.markdown("""
            - 可快速核貸
            """)

# ================================
# 🔵 PAGE 2：模型分析
# ================================
elif page == "📈 模型分析":

    st.header("📈 模型分析 Dashboard")

    uploaded_file = st.file_uploader("上傳測試資料 (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if len(df) > 10000:
            df_sample = df.sample(10000, random_state=42)
        else:
            df_sample = df
        st.write("資料預覽")
        st.dataframe(df_sample.head())

        if st.button("執行模型分析"):
          with st.spinner("模型計算中..."):
            preds = model.predict(df_sample)
            probs = model.predict_proba(df_sample)[:, 1]

            st.subheader("📊 基本統計")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("平均違約率", f"{np.mean(probs):.2%}")

            with col2:
                st.metric("高風險比例 (>0.7)", f"{np.mean(probs > 0.7):.2%}")

            # 分布圖
            st.subheader("📊 違約機率分布")
            fig, ax = plt.subplots()
            ax.hist(probs, bins=30)
            st.pyplot(fig)