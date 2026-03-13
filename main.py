# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI()

# 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit的地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载数据
try:
    df = pd.read_csv('synthetic_personal_finance_dataset.csv')
    print(f"✅ 已加载 {len(df)} 条用户记录")
except Exception as e:
    print(f"❌ 加载数据失败: {e}")
    df = None

class ChatRequest(BaseModel):
    message: str
    user_profile: dict = {}

@app.post("/chat")
async def chat(request: ChatRequest):
    if df is None:
        return {"response": "❌ 数据加载失败"}
    
    message = request.message.lower()
    
    # 简单的数据分析
    if "平均收入" in message:
        avg_income = df['monthly_income_usd'].mean()
        return {"response": f"📊 所有用户的平均月收入: ${avg_income:.2f}"}
    
    elif "高风险" in message or "debt" in message:
        if 'debt_to_income_ratio' in df.columns:
            high_risk = df[df['debt_to_income_ratio'] > 0.6]
            return {"response": f"⚠️ 发现 {len(high_risk)} 位高风险用户"}
        else:
            return {"response": "❌ 数据中没有 debt_to_income_ratio 列"}
    
    elif "总用户" in message:
        return {"response": f"👥 总用户数: {len(df)}"}
    
    elif "信用评分" in message or "credit score" in message:
        if 'credit_score' in df.columns:
            avg_score = df['credit_score'].mean()
            return {"response": f"💳 平均信用评分: {avg_score:.0f}"}
        else:
            return {"response": "❌ 数据中没有 credit_score 列"}
    
    else:
        return {"response": "🤖 我是AI金融顾问，可以帮你分析数据。试试问：平均收入、高风险用户、信用评分等"}

@app.get("/health")
async def health():
    return {"status": "healthy", "data_size": len(df) if df is not None else 0}