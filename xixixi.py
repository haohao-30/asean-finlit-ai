import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import google.generativeai as genai
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import hashlib

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ASEAN FinLit AI - Financial Literacy & Risk Protection System",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Gemini API Configuration ====================
# Method 1: Direct API key (for testing)
GEMINI_API_KEY = "AIzaSyBEOiRm1PPbvSGvgZzCg6UtDzyT0EGG_ZA"  # Your API key

# Method 2: Use Streamlit secrets (more secure, recommended)
# GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Custom CSS styling
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
    .asean-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
    }
    .risk-low { background: linear-gradient(135deg, #43a047 0%, #1e5f20 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; }
    .risk-medium { background: linear-gradient(135deg, #fb8c00 0%, #bb5e00 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; }
    .risk-high { background: linear-gradient(135deg, #e53935 0%, #ab000d 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; }
    .protection-gap { background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; }
    .chat-message { padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; }
    .user-message { background-color: #e3f2fd; text-align: right; }
    .bot-message { background-color: #f5f5f5; text-align: left; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==================== Sidebar Navigation ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/asean.png", width=100)
    st.title("ASEAN FinLit AI")
    st.markdown("*Building Financial Resilience in Southeast Asia*")
    st.markdown("---")
    
    page = st.radio(
        "Navigation Menu",
        ["🏠 Home | ASEAN Financial Challenge", 
         "📊 Financial Literacy Analysis", 
         "🔧 Risk Education Tools", 
         "🤖 AI Financial Advisor", 
         "📈 Protection Gap Assessment"]
    )
    
    st.markdown("---")
    st.subheader("🌏 ASEAN Financial Snapshot")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Digital Payments", "$1.5T", "by 2030")
    with col2:
        st.metric("Protection Gap", "65%", "uncovered")
    st.markdown("---")
    st.caption("© 2024 ASEAN FinLit AI | Hackathon Project")

# ==================== Initialize Session State ====================
# ==================== 初始化session state ====================
if 'df' not in st.session_state:
    # 读取你准备好的真实数据文件
    csv_file_path = 'synthetic_personal_finance_dataset.csv'  # 你的真实数据文件名
    
    try:
        # 尝试读取真实数据
        st.session_state.df = pd.read_csv(csv_file_path)
        st.success("✅ Real ASEAN financial dataset loaded successfully")
        st.session_state.using_real_data = True
    except FileNotFoundError:
        # 如果文件不存在，使用备用的示例数据（防止程序崩溃）
        st.warning("⚠️ Real data file not found, using sample data for demonstration")
        example_data = pd.DataFrame({
            'user_id': ['U001', 'U002', 'U003'],
            'region': ['Indonesia', 'Vietnam', 'Thailand'],
            'age': [28, 35, 42],
            'monthly_income_usd': [500, 1200, 800],
            # ... 其他字段
        })
        st.session_state.df = example_data
        st.session_state.using_real_data = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'financial_goals' not in st.session_state:
    st.session_state.financial_goals = {}

# ==================== ASEAN Countries Data ====================
ASEAN_COUNTRIES = {
    'Singapore': {'digital_adoption': 0.95, 'fin_literacy': 0.75, 'protection_gap': 0.30},
    'Malaysia': {'digital_adoption': 0.85, 'fin_literacy': 0.60, 'protection_gap': 0.45},
    'Thailand': {'digital_adoption': 0.80, 'fin_literacy': 0.55, 'protection_gap': 0.55},
    'Indonesia': {'digital_adoption': 0.70, 'fin_literacy': 0.45, 'protection_gap': 0.65},
    'Vietnam': {'digital_adoption': 0.65, 'fin_literacy': 0.40, 'protection_gap': 0.70},
    'Philippines': {'digital_adoption': 0.60, 'fin_literacy': 0.35, 'protection_gap': 0.75},
    'Myanmar': {'digital_adoption': 0.40, 'fin_literacy': 0.25, 'protection_gap': 0.85},
    'Cambodia': {'digital_adoption': 0.35, 'fin_literacy': 0.20, 'protection_gap': 0.90},
    'Laos': {'digital_adoption': 0.30, 'fin_literacy': 0.18, 'protection_gap': 0.92},
    'Brunei': {'digital_adoption': 0.90, 'fin_literacy': 0.70, 'protection_gap': 0.35}
}

# ==================== Utility Functions ====================
def calculate_financial_literacy_score(row):
    """Calculate financial literacy score based on actual dataset columns"""
    score = 0
    
    # 1. 是否有贷款知识 (20分)
    if pd.notna(row.get('has_loan')):
        # 有贷款的人对金融产品有更多了解
        if row['has_loan'] == 'Yes':
            score += 20
        elif row['has_loan'] == 'No':
            # 没有贷款可能更保守，但也可能不了解
            score += 5
    
    # 2. 负债率知识 (20分)
    if pd.notna(row.get('debt_to_income_ratio')):
        dti = float(row['debt_to_income_ratio'])
        if dti < 0.3:  # 负债率低，可能更懂理财
            score += 20
        elif dti < 0.6:  # 中等负债率
            score += 10
        else:  # 高负债率
            score += 5
    
    # 3. 信用分知识 (20分)
    if pd.notna(row.get('credit_score')):
        credit = float(row['credit_score'])
        if credit >= 750:  # 优
            score += 20
        elif credit >= 650:  # 良
            score += 15
        elif credit >= 550:  # 中
            score += 10
        else:  # 差
            score += 5
    
    # 4. 储蓄习惯 (20分)
    if pd.notna(row.get('savings_usd')) and pd.notna(row.get('monthly_income_usd')):
        savings = float(row['savings_usd'])
        income = float(row['monthly_income_usd'])
        if income > 0:
            savings_ratio = savings / income
            if savings_ratio > 12:  # 超过1年收入
                score += 20
            elif savings_ratio > 6:  # 6个月以上
                score += 15
            elif savings_ratio > 3:  # 3个月以上
                score += 10
            elif savings_ratio > 0:
                score += 5
    
    # 5. 年龄因素 (20分)
    if pd.notna(row.get('age')):
        age = float(row['age'])
        if age >= 40:  # 中年人经验丰富
            score += 20
        elif age >= 30:
            score += 15
        elif age >= 25:
            score += 10
        else:  # 年轻人
            score += 5
    
    # 确保得分在0-100之间
    return min(max(score, 0), 100)

def assess_protection_gap(row):
    """Assess protection gap"""
    base_gap = 100
    if row.get('has_insurance', 0) == 1: base_gap -= 30
    emergency_fund = row.get('emergency_fund_months', 0)
    base_gap -= min(emergency_fund * 5, 30)
    if row.get('income_stability', 'Low') == 'High': base_gap -= 20
    elif row.get('income_stability', 'Low') == 'Medium': base_gap -= 10
    fin_literacy = row.get('financial_literacy_score', 50)
    base_gap -= fin_literacy * 0.2
    return max(base_gap, 0)

def chat_with_gemini(message, user_profile=None, df=None):
    """Use Google Gemini API for real AI conversation"""
    
    try:
        # Build data summary
        data_summary = ""
        if df is not None and len(df) > 0:
            # Calculate basic statistics
            total_users = len(df)
            avg_income = df['monthly_income_usd'].mean() if 'monthly_income_usd' in df.columns else 0
            avg_credit = df['credit_score'].mean() if 'credit_score' in df.columns else 0
            avg_dti = df['debt_to_income_ratio'].mean() if 'debt_to_income_ratio' in df.columns else 0
            
            # Loan user statistics
            if 'has_loan' in df.columns:
                loan_users = len(df[df['has_loan'] == 'Yes'])
                no_loan_users = len(df[df['has_loan'] == 'No'])
            else:
                loan_users = "Unknown"
                no_loan_users = "Unknown"
            
            # High-risk users
            if 'debt_to_income_ratio' in df.columns:
                high_risk = len(df[df['debt_to_income_ratio'] > 0.6])
            else:
                high_risk = "Unknown"
            
            # Country distribution
            if 'region' in df.columns:
                regions = df['region'].value_counts().to_dict()
                region_text = ", ".join([f"{k}: {v} users" for k, v in regions.items() if k != 'Other'][:5])
            else:
                region_text = "No region data"
            
            data_summary = f"""
            Current Dataset Overview:
            - Total Users: {total_users}
            - Average Monthly Income: ${avg_income:.2f}
            - Average Credit Score: {avg_credit:.1f}
            - Average Debt-to-Income Ratio: {avg_dti:.2%}
            - Users with Loans: {loan_users}
            - Users without Loans: {no_loan_users}
            - High-Risk Users (DTI>60%): {high_risk}
            - Main Regions: {region_text}
            
            Available analysis: income distribution, credit scores, loan status, regional comparison, age analysis, debt ratios, etc.
            """
        
        # User information
        user_info = ""
        if user_profile:
            user_info = f"""
            Current User Information:
            - Age: {user_profile.get('age', 'Unknown')}
            - Country: {user_profile.get('country', 'Unknown')}
            - Occupation: {user_profile.get('occupation', 'Unknown')}
            - Monthly Income: ${user_profile.get('monthly_income', 0)}
            - Uses BNPL: {'Yes' if user_profile.get('uses_bnpl') else 'No'}
            - Has Insurance: {'Yes' if user_profile.get('has_insurance') else 'No'}
            - Financial Goal: {user_profile.get('goal', 'Unknown')}
            """
        
        # Build prompt
        prompt = f"""You are a professional ASEAN region financial advisor, expert in personal finance, risk management, investment planning, and insurance planning.
        
        {data_summary}
        
        {user_info}
        
        User Question: {message}
        
        Please answer based on the real data provided. Requirements:
        1. If the question involves data analysis, provide specific numbers based on the data
        2. If the question involves financial advice, give practical suggestions considering ASEAN characteristics
        3. Answers should be professional, friendly, and constructive
        4. If data is insufficient, provide general advice but indicate it's based on general knowledge
        5. Suggest uploading more data for more precise analysis when appropriate
        """
        
        # Call Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash-lite')  # Free model
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"❌ AI service temporarily unavailable: {str(e)}\n\nPlease check:\n1. API key is correct\n2. Internet connection is normal\n3. Gemini API is enabled"

# ==================== 风险预测模型 ====================
# Risk Prediction Model Functions

@st.cache_data
def train_risk_prediction_model(df):
    """
    Train a machine learning model to predict user risk level
    训练机器学习模型预测用户风险等级
    """
    try:
        # 准备特征和标签
        # Features: age, monthly_income, debt_to_income_ratio, credit_score, savings_ratio
        features = []
        labels = []
        
        # 创建特征矩阵
        for idx, row in df.iterrows():
            feature = []
            
            # 年龄 (归一化到0-1)
            if pd.notna(row.get('age')):
                feature.append(min(row['age'] / 100, 1.0))
            else:
                feature.append(0.5)
            
            # 月收入 (归一化到0-1，假设最大收入10000)
            if pd.notna(row.get('monthly_income_usd')):
                feature.append(min(row['monthly_income_usd'] / 10000, 1.0))
            else:
                feature.append(0.3)
            
            # 负债率 (已经是0-1范围)
            if pd.notna(row.get('debt_to_income_ratio')):
                feature.append(min(row['debt_to_income_ratio'], 1.0))
            else:
                feature.append(0.5)
            
            # 信用分 (归一化到0-1，300-850范围)
            if pd.notna(row.get('credit_score')):
                credit_norm = (row['credit_score'] - 300) / 550  # 550 = 850-300
                feature.append(min(max(credit_norm, 0), 1.0))
            else:
                feature.append(0.5)
            
            # 储蓄率 (计算储蓄/收入比，归一化)
            if pd.notna(row.get('savings_usd')) and pd.notna(row.get('monthly_income_usd')):
                if row['monthly_income_usd'] > 0:
                    savings_ratio = row['savings_usd'] / (row['monthly_income_usd'] * 12)  # 年收入倍数
                    feature.append(min(savings_ratio / 10, 1.0))  # 最多10倍年收入
                else:
                    feature.append(0)
            else:
                feature.append(0)
            
            features.append(feature)
            
            # 创建标签：基于保护缺口和负债率定义风险
            risk_score = 0
            if pd.notna(row.get('debt_to_income_ratio')):
                if row['debt_to_income_ratio'] > 0.6:
                    risk_score += 2
                elif row['debt_to_income_ratio'] > 0.4:
                    risk_score += 1
            
            if pd.notna(row.get('credit_score')):
                if row['credit_score'] < 550:
                    risk_score += 2
                elif row['credit_score'] < 650:
                    risk_score += 1
            
            if pd.notna(row.get('savings_usd')) and pd.notna(row.get('monthly_income_usd')):
                if row['monthly_income_usd'] > 0:
                    savings_months = row['savings_usd'] / row['monthly_income_usd']
                    if savings_months < 3:
                        risk_score += 2
                    elif savings_months < 6:
                        risk_score += 1
            
            # 风险等级：0-低风险，1-中风险，2-高风险
            if risk_score >= 4:
                label = 2  # 高风险
            elif risk_score >= 2:
                label = 1  # 中风险
            else:
                label = 0  # 低风险
            
            labels.append(label)
        
        X = np.array(features)
        y = np.array(labels)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练随机森林模型
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 计算特征重要性
        feature_names = ['Age', 'Income', 'Debt Ratio', 'Credit Score', 'Savings Ratio']
        importance = dict(zip(feature_names, model.feature_importances_))
        
        return {
            'model': model,
            'accuracy': accuracy,
            'importance': importance,
            'n_samples': len(df)
        }
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def predict_user_risk(model, user_data):
    """
    Predict risk level for a single user
    预测单个用户的风险等级
    """
    try:
        # 准备特征
        features = []
        
        # 年龄
        age = user_data.get('age', 30)
        features.append(min(age / 100, 1.0))
        
        # 月收入
        income = user_data.get('monthly_income', 1000)
        features.append(min(income / 10000, 1.0))
        
        # 负债率 (如果没有，用默认值0.3)
        dti = user_data.get('debt_to_income_ratio', 0.3)
        features.append(min(dti, 1.0))
        
        # 信用分 (如果没有，用默认值650)
        credit = user_data.get('credit_score', 650)
        credit_norm = (credit - 300) / 550
        features.append(min(max(credit_norm, 0), 1.0))
        
        # 储蓄率
        if 'savings' in user_data and 'monthly_income' in user_data:
            if user_data['monthly_income'] > 0:
                savings_ratio = user_data['savings'] / (user_data['monthly_income'] * 12)
                features.append(min(savings_ratio / 10, 1.0))
            else:
                features.append(0)
        else:
            features.append(0.2)  # 默认值
        
        # 预测
        X = np.array([features])
        risk_level = model.predict(X)[0]
        risk_proba = model.predict_proba(X)[0]
        
        return {
            'level': int(risk_level),
            'probability': risk_proba.tolist(),
            'low_risk_prob': risk_proba[0] if len(risk_proba) > 0 else 0,
            'medium_risk_prob': risk_proba[1] if len(risk_proba) > 1 else 0,
            'high_risk_prob': risk_proba[2] if len(risk_proba) > 2 else 0
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# ==================== Home Page ====================
if page == "🏠 Home | ASEAN Financial Challenge":
    st.markdown("""
    <div class='asean-header'>
        <h1>🌏 ASEAN FinLit AI</h1>
        <h3>Building Financial Resilience in Southeast Asia: Bridging Digital Finance & Financial Literacy</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📱 Digital Payments</h3>
            <h2>$1.5T</h2>
            <p>ASEAN Market Size by 2030</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='protection-gap'>
            <h3>🛡️ Protection Gap</h3>
            <h2>65%</h2>
            <p>Disaster losses uncovered</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='risk-high'>
            <h3>📚 Financial Literacy</h3>
            <h2>Only 36%</h2>
            <p>Adults with basic financial knowledge</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🌏 ASEAN Countries Financial Health Comparison")
    
    country_data = pd.DataFrame(ASEAN_COUNTRIES).T.reset_index()
    country_data.columns = ['Country', 'Digital Adoption', 'Financial Literacy', 'Protection Gap']
    
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Digital Adoption", "Financial Literacy", "Protection Gap"))
    fig.add_trace(go.Bar(x=country_data['Country'], y=country_data['Digital Adoption'], marker_color='royalblue'), row=1, col=1)
    fig.add_trace(go.Bar(x=country_data['Country'], y=country_data['Financial Literacy'], marker_color='green'), row=1, col=2)
    fig.add_trace(go.Bar(x=country_data['Country'], y=country_data['Protection Gap'], marker_color='crimson'), row=1, col=3)
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📁 Upload ASEAN User Data")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! {st.session_state.df.shape[0]} users")
        st.info("💡 Now go to 'AI Financial Advisor' page for intelligent data analysis!")
        
        with st.expander("📋 Data Preview"):
            st.dataframe(st.session_state.df.head())
    
    # Sample data
    st.subheader("📋 Download Sample Data")
    example_data = pd.DataFrame({
        'user_id': ['U001', 'U002', 'U003'],
        'region': ['Indonesia', 'Vietnam', 'Thailand'],
        'age': [28, 35, 42],
        'gender': ['Female', 'Male', 'Female'],
        'occupation': ['Student', 'Entrepreneur', 'Teacher'],
        'monthly_income_usd': [500, 1200, 800],
        'monthly_expenses_usd': [300, 800, 500],
        'savings_usd': [1000, 5000, 2000],
        'has_loan': ['No', 'Yes', 'No'],
        'loan_type': ['None', 'Business', 'None'],
        'credit_score': [650, 700, 600],
        'debt_to_income_ratio': [0.1, 0.4, 0.2]
    })
    csv = example_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample Data", data=csv, file_name="asean_sample.csv", mime="text/csv")

# ==================== Financial Literacy Analysis ====================
elif page == "📊 Financial Literacy Analysis":
    st.title("📊 ASEAN Financial Literacy Analysis")
    st.markdown("---")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data on the Home page first")
        st.stop()
    
    df = st.session_state.df.copy()
    
    # Calculate financial literacy score
    if 'financial_literacy_score' not in df.columns:
        df['financial_literacy_score'] = df.apply(calculate_financial_literacy_score, axis=1)
        df['literacy_level'] = pd.cut(df['financial_literacy_score'], 
                                      bins=[0, 40, 70, 100], 
                                      labels=['Low Literacy', 'Medium Literacy', 'High Literacy'])
        st.session_state.df = df
    
    col1, col2 = st.columns(2)
    with col1:
        level_counts = df['literacy_level'].value_counts()
        fig = px.pie(values=level_counts.values, names=level_counts.index,
                    title="Financial Literacy Level Distribution",
                    color=level_counts.index,
                    color_discrete_map={'Low Literacy': 'red', 'Medium Literacy': 'orange', 'High Literacy': 'green'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='financial_literacy_score', nbins=20,
                          title="Literacy Score Distribution",
                          color_discrete_sequence=['#1E88E5'])
        st.plotly_chart(fig, use_container_width=True)
    
    if 'region' in df.columns:
        country_lit = df.groupby('region')['financial_literacy_score'].mean().reset_index()
        fig = px.bar(country_lit, x='region', y='financial_literacy_score',
                    title="Average Financial Literacy by Country", color='financial_literacy_score')
        st.plotly_chart(fig, use_container_width=True)

# ==================== Risk Education Tools ====================
elif page == "🔧 Risk Education Tools":
    st.title("🔧 Financial Risk Education Tools")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["💳 BNPL Risk Simulator", "📊 Compound Interest Calculator", "🛡️ Protection Gap Assessment"])
    
    with tab1:
        st.subheader("Buy Now Pay Later (BNPL) Risk Simulator")
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Purchase Amount (USD)", 10, 5000, 500)
            installments = st.slider("Installments", 2, 12, 4)
            income = st.number_input("Monthly Income (USD)", 100, 5000, 1000)
        
        with col2:
            installment = amount / installments
            ratio = (installment / income) * 100
            st.metric("Payment per Installment", f"${installment:.2f}")
            st.metric("Percentage of Income", f"{ratio:.1f}%")
            
            if ratio > 10:
                st.error("⚠️ High Risk: Payment exceeds 10% of income")
                st.markdown("**Recommendation:** Reduce amount or extend installments")
            elif ratio > 5:
                st.warning("⚠️ Medium Risk: Payment is 5-10% of income")
                st.markdown("**Recommendation:** Set payment reminders, avoid using multiple BNPL services")
            else:
                st.success("✅ Low Risk: Payment within 5% of income")
    
    with tab2:
        st.subheader("Compound Interest Calculator")
        col1, col2 = st.columns(2)
        with col1:
            principal = st.number_input("Principal (USD)", 0, 10000, 1000)
            rate = st.slider("Annual Interest Rate (%)", 0.0, 20.0, 5.0) / 100
            years = st.slider("Years", 1, 30, 10)
        
        with col2:
            future = principal * (1 + rate) ** years
            st.metric("Future Value", f"${future:,.2f}")
            st.metric("Total Return", f"${future - principal:,.2f}")
            
            years_range = list(range(1, years + 1))
            values = [principal * (1 + rate) ** y for y in years_range]
            fig = px.area(x=years_range, y=values, title="Compound Growth Curve")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Protection Gap Quick Assessment")
        with st.form("protection_form"):
            col1, col2 = st.columns(2)
            with col1:
                has_insurance = st.radio("Have Insurance?", ["Yes", "No"])
                emergency_fund = st.selectbox("Emergency Fund", ["0 months", "1-3 months", "3-6 months", "6+ months"])
            with col2:
                income_stability = st.selectbox("Income Stability", ["Low", "Medium", "High"])
                literacy = st.slider("Financial Literacy (0-100)", 0, 100, 50)
            
            if st.form_submit_button("Assess"):
                gap = 100
                if has_insurance == "Yes": gap -= 30
                if emergency_fund == "6+ months": gap -= 25
                elif emergency_fund == "3-6 months": gap -= 15
                elif emergency_fund == "1-3 months": gap -= 5
                if income_stability == "High": gap -= 20
                elif income_stability == "Medium": gap -= 10
                gap -= literacy * 0.3
                gap = max(0, min(100, gap))
                
                if gap >= 70:
                    st.markdown(f"<div class='protection-gap'><h2>Protection Gap: {gap:.0f}%</h2><p>High Risk - Take Action Now</p></div>", unsafe_allow_html=True)
                    st.markdown("Recommendations: Get insurance, build emergency fund, learn financial knowledge")
                elif gap >= 40:
                    st.markdown(f"<div class='risk-medium'><h2>Protection Gap: {gap:.0f}%</h2><p>Medium Risk - Needs Improvement</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='risk-low'><h2>Protection Gap: {gap:.0f}%</h2><p>Low Risk - Maintain Good Habits</p></div>", unsafe_allow_html=True)

# ==================== AI Financial Advisor ====================
elif page == "🤖 AI Financial Advisor":
    st.title("🤖 AI Financial Advisor - Intelligent Data Analysis & Advice")
    st.markdown("---")
    
    # Display data status
    if st.session_state.df is not None:
        st.success(f"✅ Loaded {st.session_state.df.shape[0]} user records - ready for intelligent analysis")
    else:
        st.warning("⚠️ No data uploaded. AI will provide general advice. Upload data for personalized analysis!")
    
    # Display API status
    with st.expander("🔧 AI Service Status"):
        if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
            st.success("✅ Gemini API configured - ready to chat!")
        else:
            st.error("❌ Please set your Gemini API key in the code")
            st.markdown("""
            Get a free API key:
            1. Visit https://aistudio.google.com/
            2. Log in with your Google account
            3. Click "Get API Key" to create one
            4. Copy the key and replace YOUR_GEMINI_API_KEY in the code
            """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Your Profile")
        st.markdown("Fill in your information for personalized advice")
        
        with st.form("profile_form"):
            age = st.number_input("Age", min_value=15, max_value=100, value=30)
            country = st.selectbox("Country", list(ASEAN_COUNTRIES.keys()))
            occupation = st.selectbox("Occupation", ["Student", "Employee", "Entrepreneur", "Freelancer", "Retired"])
            monthly_income = st.number_input("Monthly Income (USD)", min_value=0, max_value=50000, value=1000, step=100)
            uses_bnpl = st.checkbox("I use BNPL services")
            has_insurance = st.checkbox("I have insurance")
            goal = st.text_input("My financial goal", "Save for retirement, buy a house, etc.")
            
            if st.form_submit_button("💾 Save Profile"):
                st.session_state.financial_goals = {
                    'age': age, 
                    'country': country, 
                    'occupation': occupation,
                    'monthly_income': monthly_income,
                    'uses_bnpl': uses_bnpl, 
                    'has_insurance': has_insurance,
                    'goal': goal
                }
                st.success("✅ Profile saved successfully!")
        
        # ===== 风险预测部分 =====
        st.markdown("---")
        st.subheader("🔮 Risk Prediction")
        
        # 检查是否有数据可以训练模型
        if st.session_state.df is not None and len(st.session_state.df) > 10:
            # 训练或加载模型
            if 'risk_model' not in st.session_state:
                with st.spinner("Training prediction model..."):
                    model_result = train_risk_prediction_model(st.session_state.df)
                    if model_result:
                        st.session_state.risk_model = model_result
                        st.session_state.risk_model_trained = True
            
            if st.session_state.get('risk_model_trained', False):
                model_result = st.session_state.risk_model
                
                # 显示模型信息
                st.info(f"📊 Model trained on {model_result['n_samples']} users | Accuracy: {model_result['accuracy']:.1%}")
                
                # 预测按钮
                if st.button("🔮 Predict My Risk Level", use_container_width=True):
                    if st.session_state.financial_goals:
                        # 准备用户数据
                        user_data = st.session_state.financial_goals.copy()
                        # 添加默认值
                        user_data['debt_to_income_ratio'] = 0.3  # 假设负债率
                        user_data['credit_score'] = 650  # 假设信用分
                        user_data['savings'] = st.session_state.df['savings_usd'].median() if 'savings_usd' in st.session_state.df.columns else 5000
                        
                        # 预测
                        prediction = predict_user_risk(model_result['model'], user_data)
                        
                        if prediction:
                            # 显示结果
                            st.markdown("### Prediction Results")
                            
                            if prediction['level'] == 0:
                                st.markdown("""
                                <div class='risk-low'>
                                    <h3>✅ Low Risk</h3>
                                    <p>You appear to be in a good financial position</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif prediction['level'] == 1:
                                st.markdown("""
                                <div class='risk-medium'>
                                    <h3>⚠️ Medium Risk</h3>
                                    <p>Some areas need attention</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class='risk-high'>
                                    <h3>🚨 High Risk</h3>
                                    <p>Immediate action recommended</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # 显示概率
                            st.markdown("#### Risk Probabilities")
                            prob_df = pd.DataFrame({
                                'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
                                'Probability': [
                                    prediction['low_risk_prob'],
                                    prediction['medium_risk_prob'],
                                    prediction['high_risk_prob']
                                ]
                            })
                            fig = px.bar(prob_df, x='Risk Level', y='Probability', 
                                       color='Risk Level',
                                       color_discrete_map={
                                           'Low Risk': 'green',
                                           'Medium Risk': 'orange',
                                           'High Risk': 'red'
                                       })
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 显示特征重要性
                            with st.expander("📊 What factors influence this prediction?"):
                                importance = model_result['importance']
                                imp_df = pd.DataFrame({
                                    'Feature': importance.keys(),
                                    'Importance': importance.values()
                                }).sort_values('Importance', ascending=False)
                                
                                fig2 = px.bar(imp_df, x='Importance', y='Feature',
                                            title="Feature Importance",
                                            orientation='h')
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                st.markdown("""
                                **How to improve your score:**
                                - **Reduce debt-to-income ratio** (aim for < 0.4)
                                - **Improve credit score** (pay bills on time)
                                - **Build emergency savings** (3-6 months of expenses)
                                - **Increase income** through career development
                                """)
                    else:
                        st.warning("Please save your profile first")
        else:
            st.info("Need more data (10+ users) to train prediction model")
    
    with col2:
        st.subheader("💬 AI Financial Advisor")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if msg['role'] == 'user':
                    st.markdown(f"<div class='chat-message user-message'>👤 You: {msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message bot-message'>🤖 AI Advisor: {msg['content']}</div>", unsafe_allow_html=True)
        
        # Input box and send button
        question = st.text_input("Enter your question...", key="chat_input")
        
        col_send, col_clear = st.columns([1, 5])
        with col_send:
            if st.button("📤 Send", use_container_width=True):
                if question:
                    # Add user message
                    st.session_state.chat_history.append({'role': 'user', 'content': question})
                    
                    # Call real AI
                    with st.spinner("🤔 AI is thinking..."):
                        response = chat_with_gemini(
                            question,
                            st.session_state.financial_goals,
                            st.session_state.df
                        )
                    
                    # Add AI response
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    # Example questions
    st.markdown("---")
    st.subheader("💡 Try These Questions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    example_questions = [
        ("📊 Data Analysis", "Help me analyze the financial situation of users aged 30-40"),
        ("💰 Financial Advice", "How to manage finances with $1000 monthly income?"),
        ("🌏 Regional Comparison", "Compare user characteristics across different countries"),
        ("⚠️ Risk Assessment", "How to assess my debt risk?"),
        ("📈 Investment Advice", "How to start investing as a beginner?"),
        ("💳 Credit Management", "How to improve my credit score?"),
        ("🛡️ Insurance Planning", "What insurance should young people get?"),
        ("🎯 Financial Goals", "How to achieve financial freedom?")
    ]
    
    for i, (btn_label, btn_question) in enumerate(example_questions):
        cols = [col1, col2, col3, col4]
        with cols[i % 4]:
            if st.button(btn_label, use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': btn_question})
                
                with st.spinner("🤔 AI is thinking..."):
                    response = chat_with_gemini(
                        btn_question,
                        st.session_state.financial_goals,
                        st.session_state.df
                    )
                
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()

# ==================== Protection Gap Assessment ====================
elif page == "📈 Protection Gap Assessment":
    st.title("📈 ASEAN Protection Gap Analysis")
    st.markdown("---")
    
    # 创建两个标签页：群体分析 vs 个性化评估
    tab1, tab2 = st.tabs(["📊 Population Analysis", "👤 Your Personal Assessment"])
    
    with tab1:
        st.subheader("Population Protection Gap Distribution")
        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            
            # 计算保护缺口
            if 'financial_literacy_score' in df.columns:
                if 'protection_gap' not in df.columns:
                    df['protection_gap'] = df.apply(assess_protection_gap, axis=1)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x='protection_gap', nbins=20,
                                      title="Protection Gap Distribution (All Users)",
                                      color_discrete_sequence=['#e53935'])
                    fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="High Risk")
                    fig.add_vline(x=40, line_dash="dash", line_color="green", annotation_text="Low Risk")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(df, x='financial_literacy_score', y='protection_gap',
                                    title="Financial Literacy vs Protection Gap",
                                    labels={'financial_literacy_score': 'Financial Literacy', 'protection_gap': 'Protection Gap'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # 高风险用户统计
                high_risk = df[df['protection_gap'] >= 70]
                if len(high_risk) > 0:
                    st.warning(f"🚨 {len(high_risk)} high-risk users in the dataset ({len(high_risk)/len(df)*100:.1f}% of total)")
                    
                    # 显示高风险用户列表
                    with st.expander("View High-Risk Users"):
                        if 'user_id' in high_risk.columns and 'region' in high_risk.columns:
                            st.dataframe(high_risk[['user_id', 'region', 'protection_gap', 'financial_literacy_score']].head(10))
        else:
            st.info("Please upload data on the Home page first for population analysis")
            # 显示示例数据预览
            st.markdown("""
            **No data loaded yet.** 
            - Go to **Home** page to upload your dataset
            - Or use the sample data provided
            """)
    
    with tab2:
        st.subheader("Your Personal Protection Gap Assessment")
        
        # 检查用户是否已保存档案
        if st.session_state.financial_goals:
            user = st.session_state.financial_goals
            
            # 基于用户档案计算保护缺口
            gap = 100
            
            # 根据年龄调整
            age = user.get('age', 30)
            if age > 50:
                gap -= 15  # 年长可能更有经验
            elif age > 40:
                gap -= 10
            elif age > 30:
                gap -= 5
            elif age < 25:
                gap += 15  # 年轻可能风险更高
            elif age < 30:
                gap += 5
            
            # 根据收入调整
            income = user.get('monthly_income', 1000)
            if income > 5000:
                gap -= 20
            elif income > 3000:
                gap -= 15
            elif income > 2000:
                gap -= 10
            elif income > 1000:
                gap -= 5
            elif income < 500:
                gap += 20
            elif income < 800:
                gap += 10
            
            # 根据保险情况
            if user.get('has_insurance'):
                gap -= 30
            else:
                gap += 20
            
            # 根据BNPL使用
            if user.get('uses_bnpl'):
                gap += 25
            
            # 根据职业
            occupation = user.get('occupation', 'Employee')
            if occupation == 'Retired':
                gap -= 10
            elif occupation == 'Student':
                gap += 15
            elif occupation == 'Freelancer':
                gap += 10
            
            gap = max(0, min(100, int(gap)))
            
            # 显示用户的结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Your Profile")
                st.write(f"**Age:** {user.get('age')}")
                st.write(f"**Country:** {user.get('country')}")
                st.write(f"**Occupation:** {user.get('occupation')}")
                st.write(f"**Monthly Income:** ${user.get('monthly_income'):,.0f}")
                st.write(f"**Has Insurance:** {'✅ Yes' if user.get('has_insurance') else '❌ No'}")
                st.write(f"**Uses BNPL:** {'✅ Yes' if user.get('uses_bnpl') else '❌ No'}")
                st.write(f"**Financial Goal:** {user.get('goal', 'Not specified')}")
            
            with col2:
                st.markdown("#### Your Protection Gap")
                
                # 显示保护缺口仪表盘
                if gap >= 70:
                    st.markdown(f"""
                    <div class='protection-gap' style='text-align: center; padding: 20px;'>
                        <h1 style='font-size: 48px;'>{gap}%</h1>
                        <h3>🚨 HIGH RISK</h3>
                        <p>Immediate action required</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.error("⚠️ Your protection gap is critically high!")
                    st.markdown("""
                    **📋 Action Plan:**
                    1. **Get insurance** - Health, life, and property insurance are essential
                    2. **Build emergency fund** - Aim for 3-6 months of expenses
                    3. **Stop using BNPL** - Avoid high-interest debt
                    4. **Create a budget** - Track all income and expenses
                    5. **Improve financial literacy** - Learn about personal finance
                    """)
                    
                elif gap >= 40:
                    st.markdown(f"""
                    <div class='risk-medium' style='text-align: center; padding: 20px;'>
                        <h1 style='font-size: 48px;'>{gap}%</h1>
                        <h3>⚠️ MEDIUM RISK</h3>
                        <p>Needs improvement</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("⚡ Your protection gap needs attention")
                    st.markdown("""
                    **📋 Improvement Plan:**
                    1. **Review insurance** - Make sure you have adequate coverage
                    2. **Increase savings** - Build towards 3 months of expenses
                    3. **Reduce debt** - Focus on high-interest debt first
                    4. **Use BNPL wisely** - Only for essential purchases
                    5. **Set financial goals** - Create short-term and long-term goals
                    """)
                    
                else:
                    st.markdown(f"""
                    <div class='risk-low' style='text-align: center; padding: 20px;'>
                        <h1 style='font-size: 48px;'>{gap}%</h1>
                        <h3>✅ LOW RISK</h3>
                        <p>Good financial health</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("🎉 You're in good financial shape!")
                    st.markdown("""
                    **📋 Maintenance Plan:**
                    1. **Keep up the good work** - Maintain your healthy habits
                    2. **Consider investments** - Grow your wealth over time
                    3. **Review insurance annually** - Ensure coverage still fits
                    4. **Plan for retirement** - Start early for compound benefits
                    5. **Help others** - Share your knowledge with family
                    """)
            
            # ===== 对比群体（修复的部分）=====
            if st.session_state.df is not None and len(st.session_state.df) > 0:
                st.markdown("---")
                st.subheader("📊 Compare with Population")
                
                df = st.session_state.df.copy()
                
                # 确保数据有 protection_gap 列
                if 'protection_gap' not in df.columns:
                    if 'financial_literacy_score' in df.columns:
                        df['protection_gap'] = df.apply(assess_protection_gap, axis=1)
                    else:
                        # 如果没有金融素养得分，先计算
                        df['financial_literacy_score'] = df.apply(calculate_financial_literacy_score, axis=1)
                        df['protection_gap'] = df.apply(assess_protection_gap, axis=1)
                
                # 计算统计数据
                avg_gap = df['protection_gap'].mean()
                median_gap = df['protection_gap'].median()
                min_gap = df['protection_gap'].min()
                max_gap = df['protection_gap'].max()
                
                # 显示对比指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Your Gap", f"{gap}%", 
                             delta=f"{gap - avg_gap:.1f}%" if gap != avg_gap else None)
                with col2:
                    st.metric("Population Average", f"{avg_gap:.1f}%")
                with col3:
                    st.metric("Population Median", f"{median_gap:.1f}%")
                with col4:
                    st.metric("Population Range", f"{min_gap:.0f}-{max_gap:.0f}%")
                
                # 计算用户的百分位数
                better_than = (df['protection_gap'] < gap).sum()
                worse_than = (df['protection_gap'] > gap).sum()
                same_as = (df['protection_gap'] == gap).sum()
                total = len(df)
                
                st.markdown(f"""
                **Your position in the population:**
                - You have a **lower protection gap than {better_than} users** ({better_than/total*100:.1f}%)
                - You have a **higher protection gap than {worse_than} users** ({worse_than/total*100:.1f}%)
                - You are **same as {same_as} users** ({same_as/total*100:.1f}%)
                """)
                
                # 创建对比图表
                fig = px.histogram(df, x='protection_gap', nbins=20,
                                  title="📊 Your Position in the Protection Gap Distribution",
                                  labels={'protection_gap': 'Protection Gap (%)'},
                                  color_discrete_sequence=['lightgray'])
                
                # 添加你的位置线
                fig.add_vline(x=gap, line_dash="dash", line_color="blue", 
                             annotation_text="You", annotation_position="top",
                             line_width=3)
                
                # 添加风险阈值线
                fig.add_vline(x=70, line_dash="dot", line_color="red", 
                             annotation_text="High Risk", annotation_position="top")
                fig.add_vline(x=40, line_dash="dot", line_color="green", 
                             annotation_text="Low Risk", annotation_position="bottom")
                
                fig.update_layout(
                    xaxis_title="Protection Gap (%)",
                    yaxis_title="Number of Users",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 按人口统计特征对比
                with st.expander("📈 Compare with Similar Demographics"):
                    st.markdown("#### Compare with users of same age group")
                    
                    # 年龄分组对比
                    user_age = user.get('age', 30)
                    age_groups = [
                        (0, 25, "Under 25"),
                        (25, 35, "25-35"),
                        (35, 50, "35-50"),
                        (50, 100, "Over 50")
                    ]
                    
                    for age_min, age_max, age_label in age_groups:
                        if age_min <= user_age < age_max:
                            st.info(f"**Your age group:** {age_label}")
                            
                            # 筛选同年龄组
                            same_age = df[(df['age'] >= age_min) & (df['age'] < age_max)]
                            if len(same_age) > 0:
                                avg_age_gap = same_age['protection_gap'].mean()
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Your Gap", f"{gap}%")
                                with col2:
                                    st.metric(f"Avg for {age_label}", f"{avg_age_gap:.1f}%",
                                             delta=f"{gap - avg_age_gap:.1f}%")
                            
                            break
                    
                    st.markdown("#### Compare with users in same country")
                    user_country = user.get('country', 'Singapore')
                    same_country = df[df['region'] == user_country]
                    if len(same_country) > 0:
                        avg_country_gap = same_country['protection_gap'].mean()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Your Gap", f"{gap}%")
                        with col2:
                            st.metric(f"Avg in {user_country}", f"{avg_country_gap:.1f}%",
                                     delta=f"{gap - avg_country_gap:.1f}%")
        else:
            st.warning("⚠️ Please save your profile in the AI Financial Advisor page first")
            st.info("""
            **How to get your personal assessment:**
            1. Go to **AI Financial Advisor** page
            2. Fill in your profile information
            3. Click **'Save Profile'** button
            4. Return here to see your personalized results
            """)
            
            # 显示预览
            with st.expander("👀 Preview: What you'll see"):
                st.markdown("""
                After saving your profile, you'll see:
                - Your personalized protection gap score
                - Risk level assessment
                - Comparison with population data
                - Age group and country comparisons
                - Actionable recommendations
                """)