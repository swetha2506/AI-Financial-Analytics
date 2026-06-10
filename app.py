# PROFESSIONAL AI FINANCIAL ANALYTICS PLATFORM 
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai

import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")
st.set_page_config(
    page_title="AI Financial Analytics Platform",
    page_icon="💰",
    layout="wide"
)
st.markdown("""
<style>

/* =========================
   MAIN APP
========================= */

.stApp{
    background: #0F172A;
    color: #F8FAFC;
}

/* =========================
   SIDEBAR
========================= */

[data-testid="stSidebar"]{

    background:linear-gradient(
        180deg,
        #E5E7EB,
        #D1D5DB
    ) !important;

}
[data-testid="stSidebar"] *{
    color:#111827 !important;
}

/* =========================
   HEADINGS
========================= */

h1{
    color:#E5E7EB !important;
    font-weight:800 !important;
}

h2{
    color:#D1D5DB !important;
    font-weight:700 !important;
}

h3,h4,h5,h6{
    color:#D1D5DB !important;
}
/* ALL INPUT LABELS */

.stTextInput label,
.stNumberInput label,
.stDateInput label,
.stSelectbox label,
.stTextArea label,
.stRadio label,
.stCheckbox label{

    color:#E2E8F0 !important;

    font-weight:600 !important;

    opacity:1 !important;
}
.stSlider label{
    color:#F8FAFC !important;
    font-weight:600 !important;
}
div[data-testid="stSlider"] label{
    color:#F8FAFC !important;
}
.stInfo{

    color:white !important;
}
/* =========================
   KPI CARDS
========================= */


div[data-testid="stMetric"] {

    background: linear-gradient(
       135deg,
       #334155,
       #3F4D63
    ) !important;

    padding: 20px !important;

    border-radius: 18px !important;

    border: none !important;

    box-shadow: none
     !important;
}

/* KPI LABEL */

div[data-testid="stMetric"] label {

    color: white !important;

    font-weight: 600 !important;

    opacity: 1 !important;
}

/* KPI VALUE */

div[data-testid="stMetric"] div {

    color: white !important;
}
/* =========================
   BUTTONS
========================= */

.stButton button{

    background:
    linear-gradient(
        135deg,
        #334155,
        #3F4D63
    ) !important;

    color:white !important;

    border:none !important;

    border-radius:14px !important;

    font-weight:700 !important;

    padding:10px 24px !important;

    transition:all 0.3s ease;

    box-shadow:
    0 4px 15px rgba(
        59,
        130,
        246,
        0.25
    );

}

.stButton button:hover{

    background:
    linear-gradient(
        135deg,
        #334155,
        #3F4D63
    ) !important;

    transform:none;

    box-shadow:
    0 4px 12px rgba(
        0,
        0,
        0,
        0.25
    );

}
/* =========================
   DOWNLOAD BUTTON
========================= */

.stDownloadButton button{

    background:
    linear-gradient(
        90deg,
        #334155,
        #475569
    ) !important;

    color:white !important;

    font-weight:700 !important;

    border:none !important;

    border-radius:14px !important;

}

.stDownloadButton button:hover{
    background:
    linear-gradient(
        90deg,
        #334155,
        #475569
    ) !important;

    transform:none;

    box-shadow:
    0 10px 25px rgba(
        71,
        85,
        105,
        0.45
    ) !important;

}


/* =========================
   INPUT BOXES
========================= */

.stTextInput input,
.stNumberInput input,
.stDateInput input{

    background:#1E293B !important;

    color:white !important;

    border:1px solid #334155 !important;

    border-radius:10px !important;

}

/* =========================
   SELECT BOXES
========================= */

.stSelectbox div[data-baseweb="select"]{

    background:#1E293B !important;

    color:white !important;

    border-radius:10px !important;

}

/* Dropdown menu text */
div[role="listbox"]{

    background:#1E293B !important;

    color:white !important;

}

/* =========================
   DATAFRAMES
========================= */

[data-testid="stDataFrame"]{

    border-radius:15px;

    overflow:hidden;

}

/* Table Headers */

thead tr th{

    background:#1E293B !important;

    color:white !important;

    font-weight:bold !important;

}

/* =========================
   SUCCESS INFO WARNING
========================= */

.stSuccess{

    border-radius:12px;

}

.stInfo{

    border-radius:12px;

}

.stWarning{

    border-radius:12px;

}

.stError{

    border-radius:12px;

}

/* =========================
   EXPANDERS
========================= */

.streamlit-expanderHeader{

    background:#1E293B !important;

    color:white !important;

    border-radius:10px;

}

/* =========================
   CHAT HISTORY BOX
========================= */

.chat-history{

    background:#1E293B;

    padding:15px;

    border-radius:15px;

    margin-bottom:10px;

}

/* =========================
   SCROLLBAR
========================= */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-track{
    background:#0F172A;
}

::-webkit-scrollbar-thumb{
    background:#3B82F6;
    border-radius:10px;
}
[data-testid="stElementToolbar"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

FINANCE_COLORS = {
    "income": "#10B981",
    "expense": "#EF4444",
    "saving": "#06B6D4",
    "investment": "#8B5CF6",
    "debt": "#F59E0B"
}
if "logged_in" not in st.session_state:

    st.session_state.logged_in = False

if "username" not in st.session_state:

    st.session_state.username = ""

st.markdown(
    """
    <style>

    .main {
        background-color: #0F172A;
        color: white;
    }

    h1, h2, h3, h4 {
        color: #38BDF8;
    }

    .stMetric {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #1E293B;
    }

    .css-1d391kg {
        background-color: #020617;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data

def load_data():
    df = pd.read_csv("synthetic_personal_finance_dataset.csv")
    df['record_date'] = pd.to_datetime(df['record_date'])
    return df


df = load_data()

# -----------------------------------------------------
# DATABASE CONNECTION
# -----------------------------------------------------

conn = sqlite3.connect(
    'financial_analytics.db',
    check_same_thread=False
)

cursor = conn.cursor()

conn.commit()
cursor = conn.cursor()
cursor.execute(
    '''
    CREATE TABLE IF NOT EXISTS users (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT UNIQUE,

        email TEXT UNIQUE,

        password TEXT

    )
    '''
)

conn.commit()

cursor.execute(

    '''

    CREATE TABLE IF NOT EXISTS user_financial_records (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        income REAL,

        savings REAL,

        credit_score INTEGER,

        debt_ratio REAL,

        predicted_expense REAL,

        risk_level TEXT,

        financial_status TEXT

    )

    '''

)
cursor.execute("""
CREATE TABLE IF NOT EXISTS expense_records (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT,

    expense_date TEXT,

    category TEXT,

    amount REAL,

    description TEXT

)
""")

conn.commit()

# -----------------------------------------------------
# USER PROFILE TABLE
# -----------------------------------------------------

cursor.execute(

    '''

    CREATE TABLE IF NOT EXISTS user_profiles (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT UNIQUE,

        income REAL,

        expenses REAL,

        savings REAL,

        debt REAL,

        credit_score INTEGER,

        investments REAL,

        financial_goal TEXT

    )

    '''

)


conn.commit()
cursor.execute(
    '''

    CREATE TABLE IF NOT EXISTS analysis_history (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT,

        income REAL,

        expenses REAL,

        savings REAL,

        debt REAL,

        investments REAL,

        credit_score REAL,

        financial_goal TEXT,

        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP

    )

    '''

)

conn.commit()
# -----------------------------------------------------
# PDF REPORT GENERATOR
# -----------------------------------------------------

def generate_pdf_report(

    income,
    savings,
    predicted_expense,
    risk_level,
    financial_status

):

    doc = SimpleDocTemplate(
        "financial_report.pdf"
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "AI Financial Analytics Report",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    report_content = f'''

    <b>Monthly Income:</b> ${income}<br/><br/>

    <b>Total Savings:</b> ${savings}<br/><br/>

    <b>Predicted Expenses:</b>
    ${round(predicted_expense,2)}<br/><br/>

    <b>Risk Level:</b>
    {risk_level}<br/><br/>

    <b>Financial Status:</b>
    {financial_status}<br/><br/>

    '''

    paragraph = Paragraph(
        report_content,
        styles['BodyText']
    )

    elements.append(paragraph)

    doc.build(elements)

# -----------------------------------------------------
# TRAIN ML MODELS
# -----------------------------------------------------

# EXPENSE PREDICTION MODEL

X_expense = df[[
    'monthly_income_usd',
    'savings_usd',
    'credit_score',
    'debt_to_income_ratio'
]]

Y_expense = df['monthly_expenses_usd']

X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
    X_expense,
    Y_expense,
    test_size=0.2,
    random_state=42
)

expense_model = LinearRegression()
expense_model.fit(X_train_exp, y_train_exp)

# CREDIT RISK MODEL


df['risk_label'] = (
    df['credit_score'] < 600
).astype(int)

X_risk = df[[
    'monthly_income_usd',
    'monthly_expenses_usd',
    'savings_usd',
    'debt_to_income_ratio'
]]

Y_risk = df['risk_label']

X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
    X_risk,
    Y_risk,
    test_size=0.2,
    random_state=42
)

risk_model = RandomForestClassifier()
risk_model.fit(X_train_risk, y_train_risk)
# -----------------------------------------------------
# LOGIN / SIGNUP SYSTEM
# -----------------------------------------------------

st.title("TEST PAGE")
st.write("Render is working")
st.stop()
    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        st.title("AI-Driven Personal Financial Intelligence Platform")
        st.markdown("""
        <p style="
        color:#F8FAFC;
        font-size:20px;
        font-weight:500;
        margin-bottom:20px;
        ">
        Smart Expense Tracking • Risk Analysis • Forecasting •
        AI Insights
        </p>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(
        ["📝 Signup", "🔑 Login"]
    )

    # ---------------- SIGNUP ----------------

        with tab1:

            signup_username = st.text_input(
                "Username",
                key="signup_user"
            )

            signup_password = st.text_input(
                "Password",
                type="password",
                key="signup_pass"
            )
            if st.button(
                "Create Account",
                use_container_width=True
            ):

                st.success("Button clicked successfully")

    

            # ---------------- LOGIN ----------------

        with tab2:

            login_username = st.text_input(
                "Username",
                key="login_user"
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="login_pass"
            )

            if st.button(
                "Login",
                use_container_width=True
            ):
                cursor.execute(

                    '''
                    SELECT *
                    FROM users
                    WHERE username = ?
                    AND password = ?
                    ''',

                    (
                        login_username,
                        login_password
                    )

                )

                user = cursor.fetchone()

                if user:

                    st.session_state.logged_in = True

                    st.session_state.username = (
                        login_username
                    )

                    cursor.execute(

                        '''
                        SELECT income,
                               expenses,
                               savings,
                               debt,
                               credit_score,
                               investments,
                               financial_goal
                        FROM user_profiles
                        WHERE username = ?
                        ''',

                        (
                            login_username,
                        )

                    )

                    profile = cursor.fetchone()

                    if profile:

                        st.session_state.user_profile = {

                            "income": profile[0],
                            "expenses": profile[1],
                            "savings": profile[2],
                            "debt": profile[3],
                            "credit_score": profile[4],
                            "investments": profile[5],
                            "financial_goal": profile[6]

                        }

                    st.success(
                        "Login successful."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid username or password."
                    )
    st.stop()
# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""
st.sidebar.title("💰 Fintech Analytics")

if st.session_state.logged_in:
    st.sidebar.success(
        f"Logged in as: {st.session_state.username}"
    )

pages = [

    "Login",

    "Register"

]

if st.session_state.logged_in:

    pages = [

        "Home",

        "Financial Profile",

        "Analytics Dashboard",

        "Expense Tracker",

        "Spending Analysis",

        "Risk Analyzer",

        "Financial Health",

        "Forecasting",

        "Smart Recommendations",

        "Analysis History",

        "AI Chatbot",

        "Logout"

    ]

if "page" not in st.session_state:
    st.session_state.page = pages[0]

page = st.sidebar.radio(

    "Navigation",

    pages,

    index=pages.index(st.session_state.page)

)

st.session_state.page = page
if page == "Logout":

    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_profile = {}

    st.success("Logged out successfully.")

    st.rerun()
#-----------------------------------------------------
# home
# -----------------------------------------------------
if page == "Home":

    st.markdown(
        """
        <h1 style='text-align:center;'>
        Financial Analytics Platform
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='text-align:center;'>

        <h3 style='color:#4A90E2;'>
         Your Personal AI-Powered Financial Intelligence Platform
        </h3>


        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        div.stButton > button {
            height: 70px;
            font-size: 20px;
            font-weight: bold;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    

    col1, col2, col3 = st.columns(3)

    
# -----------------------------
# COLUMN 1
# -----------------------------
    with col1:

        if st.button(
            " Financial Profile",
            use_container_width=True
        ):
            st.session_state.page = "Financial Profile"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Analytics Dashboard",
            use_container_width=True
        ):
            st.session_state.page = "Analytics Dashboard"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Expense Tracker",
            use_container_width=True
        ):
            st.session_state.page = "Expense Tracker"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Spending Analysis",
            use_container_width=True
        ):
            st.session_state.page = "Spending Analysis"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)


    # -----------------------------
    # COLUMN 2
    # -----------------------------
    with col2:

        if st.button(
            "Expense Predictor",
            use_container_width=True
        ):
            st.session_state.page = "Expense Predictor"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Risk Analyzer",
            use_container_width=True
        ):
            st.session_state.page = "Risk Analyzer"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Financial Health",
            use_container_width=True
        ):
            st.session_state.page = "Financial Health"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Forecasting",
            use_container_width=True
        ):
            st.session_state.page = "Forecasting"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)


    # -----------------------------
    # COLUMN 3
    # -----------------------------
    with col3:

        if st.button(
            "Smart Recommendations",
            use_container_width=True
        ):
            st.session_state.page = "Smart Recommendations"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Analysis History",
            use_container_width=True
        ):
            st.session_state.page = "Analysis History"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "AI Chatbot",
            use_container_width=True
        ):
            st.session_state.page = "AI Chatbot"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
# -----------------------------------------------------
# financial profile
# -----------------------------------------------------

elif page == "Financial Profile":

    st.title("🏦 Create Your Financial Profile")

    st.markdown(
        "Enter your financial details to generate personalized AI insights."
    )

    # LOAD EXISTING PROFILE

    profile = st.session_state.get(
        "user_profile",
        {}
    )

    # USER INPUTS

    income = st.number_input(
        "Monthly Income (₹)",
        min_value=0.0,
        value=profile.get("income", 0.0),
        help="Enter your monthly salary or income"
    )

    expenses = st.number_input(
        "Monthly Expenses (₹)",
        min_value=0.0,
        value=profile.get("expenses", 0.0),
        help="Enter your total monthly expenses"
    )

    savings = st.number_input(
        "Total Savings (₹)",
        min_value=0.0,
        value=profile.get("savings", 0.0),
        help="Enter your current total savings"
    )

    debt = st.number_input(
        "Total Debt (₹)",
        min_value=0.0,
        value=profile.get("debt", 0.0),
        help="Enter your total outstanding debt"
    )

    credit_score = st.slider(
        "Credit Score",
        300,
        850,
        profile.get("credit_score", 650),
        help="Higher credit score indicates better financial health"
    )

    if credit_score < 550:

        st.error(
            "🔴 Poor Credit Score"
        )

    elif credit_score < 700:

        st.warning(
            "🟡 Fair Credit Score"
        )

    else:

        st.success(
            "🟢 Excellent Credit Score"
        )
    investments = st.number_input(
        "Monthly Investments (₹)",
        min_value=0.0,
        value=profile.get("investments", 0.0),
        help="Enter your monthly investment amount"
    )

    goals = [
        "Save More",
        "Reduce Debt",
        "Buy House",
        "Retirement Planning",
        "Emergency Fund"
    ]

    financial_goal = st.selectbox(
        "Financial Goal",
        goals,
        index=goals.index(
            profile.get(
                "financial_goal",
                "Save More"
            )
        ),
        help="Select your primary financial objective"
    )

    # SAVE PROFILE

    if st.button("Save Financial Profile"):

        st.session_state.user_profile = {

            "income": income,
            "expenses": expenses,
            "savings": savings,
            "debt": debt,
            "credit_score": credit_score,
            "investments": investments,
            "financial_goal": financial_goal

        }

        cursor.execute(

            '''

            INSERT OR REPLACE INTO user_profiles (

                username,
                income,
                expenses,
                savings,
                debt,
                credit_score,
                investments,
                financial_goal

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            ''',

            (

                st.session_state.username,
                income,
                expenses,
                savings,
                debt,
                credit_score,
                investments,
                financial_goal

            )

        )

        conn.commit()
        cursor.execute(
            '''
            INSERT INTO analysis_history (

                username,

                income,

                expenses,

                savings,

                debt,

                investments,

                credit_score,

                financial_goal

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

            ''',

            (

                st.session_state.username,

                income,

                expenses,

                savings,

                debt,

                investments,

                credit_score,

                financial_goal

            )

        )
        conn.commit()      # ✅ same indentation as cursor.execute()

        st.success(
            "Financial profile saved successfully."
        )

        # PROFILE SUMMARY

        st.markdown("## 📋 Profile Summary")

        st.info(
            f"""
            💰 Income: ₹{income:,.0f}

            💸 Expenses: ₹{expenses:,.0f}

            🏦 Savings: ₹{savings:,.0f}

            📉 Debt: ₹{debt:,.0f}

            📈 Investments: ₹{investments:,.0f}

            🎯 Goal: {financial_goal}
            """
        )
        if debt < income * 0.5 and savings > expenses:

            st.success(
                "🟢 Healthy Financial Profile"
            )

        elif debt > income:

            st.error(
                "🔴 High Financial Risk"
            )

        else:

            st.warning(
                "🟡 Moderate Financial Health"
            )
            
# -----------------------------------------------------
# ANALYTICS DASHBOARD
# -----------------------------------------------------

elif page == "Analytics Dashboard":

    st.title("Personalized Financial Dashboard")

    # LOAD USER PROFILE

    profile = st.session_state.get(
        "user_profile",
        {}
    )

    if not profile:

        st.warning(
            "Please create your financial profile first."
        )

    else:

        income = profile["income"]
        expenses = profile["expenses"]
        savings = profile["savings"]
        debt = profile["debt"]
        investments = profile["investments"]
        credit_score = profile["credit_score"]

        # FINANCIAL METRICS

        savings_ratio = savings / income if income > 0 else 0

        expense_ratio = expenses / income if income > 0 else 0

        investment_ratio = investments / income if income > 0 else 0

        debt_ratio = debt / income if income > 0 else 0

        predicted_expense = expense_model.predict([[
            income,
            savings,
            credit_score,
            debt_ratio
        ]])[0]
        # KPI CARDS

        st.markdown("## 💳 Financial Overview")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Income",
            f"₹{income:,.0f}"
        )

        col2.metric(
            "Expenses",
            f"₹{expenses:,.0f}"
        )

        col3.metric(
            "Savings",
            f"₹{savings:,.0f}"
        )

        col4.metric(
            "Debt",
            f"₹{debt:,.0f}"
        )

        with col5:
            st.metric(
                "Predicted Expense",
                f"₹ {predicted_expense:,.0f}"
            )
        # PIE CHART
        st.markdown("## 📈 Financial Distribution")

        col1, col2 = st.columns([1.3, 1])

        labels = [
            "Expenses",
            "Savings",
            "Investments"
        ]

        values = [
            expenses,
            savings,
            investments
        ]

        colors=[
            "#EF4444",
            "#06B6D4",
            "#8B5CF6"
        ]

        with col1:

            fig, ax = plt.subplots(
                figsize=(2.8, 2.8)
            )

            ax.pie(
                values,
                labels=None,
                colors=colors,
                startangle=90,
                wedgeprops=dict(
                    width=0.35
                )
            )
            

            ax.axis("equal")
            fig.patch.set_facecolor("#0F172A")
            ax.set_facecolor("#0F172A")

            st.pyplot(
                fig,
                use_container_width=False
            )

        with col2:

            total = sum(values)
        

            st.markdown("<br><br><br>", unsafe_allow_html=True)


            st.markdown("### Distribution")

           
            st.markdown(
                f"<span style='color:#EF4444;font-size:22px'>⬤</span> "
                f"<span style='color:white;font-size:18px'><b>Expenses</b> — {round(expenses/total*100,1)}%</span>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<span style='color:#06B6D4;font-size:22px'>⬤</span> "
                f"<span style='color:white;font-size:18px'><b>Savings</b> — {round(savings/total*100,1)}%</span>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<span style='color:#8B5CF6;font-size:22px'>⬤</span> "
                f"<span style='color:white;font-size:18px'><b>Investments</b> — {round(investments/total*100,1)}%</span>",
                unsafe_allow_html=True
            )
        # RISK ANALYSIS

        st.markdown("## ⚠️ Financial Risk Analysis")

        if debt_ratio > 1:

            st.error(
                "High debt risk detected."
            )

        elif debt_ratio > 0.5:

            st.warning(
                "Moderate debt level detected."
            )

        else:

            st.success(
                "Debt level looks healthy."
            )

        # AI INSIGHTS

        st.markdown("## 🧠 AI Insights")

        insights = []

        if expense_ratio > 0.7:

            insights.append(
                "Your expenses are consuming most of your income."
            )

        if savings_ratio < 2:

            insights.append(
                "Savings growth can be improved."
            )

        if investments < 500:

            insights.append(
                "Consider increasing investments."
            )

        if credit_score < 650:

            insights.append(
                "Credit score improvement recommended."
            )

        if len(insights) == 0:

            insights.append(
                "Financial profile looks strong."
            )

        for insight in insights:

            st.info(insight)

        
        # -----------------------------------------------------
# EXPENSE tracker
# -----------------------------------------------------

elif page == "Expense Tracker":

    st.title("Expense Tracker")

    from datetime import date

    expense_date = st.date_input(
        "Expense Date",
        value=date.today()
    )

    category = st.selectbox(

        "Category",

        [
            "Food",
            "Travel",
            "Shopping",
            "Bills",
            "Entertainment",
            "Healthcare",
            "Education",
            "Others"
        ]

    )

    amount = st.number_input(
        "Amount",
        min_value=0.0
    )

    description = st.text_input(
        "Description"
    )
    if st.button("➕ Add Expense"):

        cursor.execute(

            """
            INSERT INTO expense_records

            (
                username,
                expense_date,
                category,
                amount,
                description
            )

            VALUES (?, ?, ?, ?, ?)
            """,

            (

                st.session_state.username,

                str(expense_date),

                category,

                amount,

                description

            )

        )

        conn.commit()

        st.success(
            "Expense added successfully."
        )
        st.markdown(
        "## 📋 Expense History"
    )

    expenses_df = pd.read_sql_query(

        """
        SELECT *

        FROM expense_records

        WHERE username = ?
        """,

        conn,

        params=(
            st.session_state.username,
        )
    )
        # SEARCH FILTER

    search = st.text_input(
        "🔍 Search Expense by Description"
    )

    if search:

        expenses_df = expenses_df[

            expenses_df["description"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )

        ]


    
    if "username" in expenses_df.columns:
        expenses_df = expenses_df.drop(
            columns=["username"]
        )

    edited_df = st.data_editor(
    expenses_df,
    use_container_width=True,
    num_rows="dynamic"
    )

    if st.button("💾 Save Changes"):

        cursor.execute(
            """
            DELETE FROM expense_records
            WHERE username = ?
            """,
            (st.session_state.username,)
        )

        for _, row in edited_df.iterrows():

            cursor.execute(
                """
                INSERT INTO expense_records
                (
                    id,
                    username,
                    expense_date,
                    category,
                    amount,
                    description
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row["id"]),
                    st.session_state.username,
                    row["expense_date"]),
                    row["category"],
                    float(row["amount"]),
                    row["description"]
                )
            

        conn.commit()

        st.success("Changes saved successfully.")
        expenses_df = expenses_df.drop(
        columns=["username"]
    )
    if st.button("📊 Load Sample Data"):

        demo_expenses = [

            # JANUARY

            ("2026-01-02","Food",450,"Restaurant"),
            ("2026-01-05","Travel",1200,"Bus Ticket"),
            ("2026-01-08","Bills",2500,"Electricity"),
            ("2026-01-10","Shopping",1800,"Clothes"),
            ("2026-01-12","Healthcare",700,"Medicines"),
            ("2026-01-15","Food",350,"Snacks"),
            ("2026-01-18","Travel",800,"Fuel"),
            ("2026-01-20","Entertainment",1000,"Movie"),
            ("2026-01-22","Education",1500,"Course"),
            ("2026-01-25","Food",600,"Dinner"),
            ("2026-01-27","Shopping",1200,"Accessories"),
            ("2026-01-29","Bills",1800,"Internet"),

            # FEBRUARY

            ("2026-02-02","Food",500,"Lunch"),
            ("2026-02-05","Travel",900,"Fuel"),
            ("2026-02-07","Bills",2600,"Electricity"),
            ("2026-02-09","Shopping",2200,"Shoes"),
            ("2026-02-11","Healthcare",600,"Clinic"),
            ("2026-02-14","Entertainment",1500,"Concert"),
            ("2026-02-17","Food",700,"Dinner"),
            ("2026-02-19","Education",1800,"Books"),
            ("2026-02-21","Travel",1100,"Train Ticket"),
            ("2026-02-24","Shopping",1400,"Bag"),
            ("2026-02-26","Bills",2000,"Water Bill"),
            ("2026-02-28","Food",450,"Breakfast"),

            # MARCH

            ("2026-03-03","Food",650,"Restaurant"),
            ("2026-03-05","Travel",1300,"Flight"),
            ("2026-03-08","Bills",2800,"Electricity"),
            ("2026-03-10","Shopping",2500,"Clothes"),
            ("2026-03-12","Healthcare",800,"Medicines"),
            ("2026-03-15","Entertainment",1800,"Gaming"),
            ("2026-03-17","Food",500,"Lunch"),
            ("2026-03-20","Education",2000,"Course"),
            ("2026-03-22","Travel",1200,"Fuel"),
            ("2026-03-24","Shopping",1700,"Accessories"),
            ("2026-03-27","Bills",2200,"Internet"),
            ("2026-03-29","Food",750,"Dinner"),

            # APRIL

            ("2026-04-02","Food",700,"Dinner"),
            ("2026-04-05","Travel",1400,"Fuel"),
            ("2026-04-08","Bills",3000,"Electricity"),
            ("2026-04-10","Shopping",2800,"Fashion"),
            ("2026-04-12","Healthcare",900,"Hospital"),
            ("2026-04-15","Entertainment",1600,"Movie"),
            ("2026-04-17","Food",600,"Restaurant"),
            ("2026-04-20","Education",2200,"Certification"),
            ("2026-04-22","Travel",1300,"Train"),
            ("2026-04-24","Shopping",1900,"Footwear"),
            ("2026-04-27","Bills",2400,"Internet"),
            ("2026-04-29","Food",800,"Family Dinner"),

            # MAY

            ("2026-05-02","Food",750,"Restaurant"),
            ("2026-05-05","Travel",1500,"Fuel"),
            ("2026-05-08","Bills",3200,"Electricity"),
            ("2026-05-10","Shopping",3200,"Shopping Mall"),
            ("2026-05-12","Healthcare",1000,"Medical Checkup"),
            ("2026-05-15","Entertainment",2000,"Concert"),
            ("2026-05-17","Food",650,"Lunch"),
            ("2026-05-20","Education",2500,"Online Course"),
            ("2026-05-22","Travel",1400,"Flight"),
            ("2026-05-24","Shopping",2100,"Electronics"),
            ("2026-05-27","Bills",2600,"Internet"),
            ("2026-05-29","Food",900,"Dinner")

        ]

        for exp in demo_expenses:

            cursor.execute(

                """
                INSERT INTO expense_records
                (
                    username,
                    expense_date,
                    category,
                    amount,
                    description
                )

                VALUES (?, ?, ?, ?, ?)
                """,

                (
                    st.session_state.username,
                    exp[0],
                    exp[1],
                    exp[2],
                    exp[3]
                )

            )

        conn.commit()

        st.success("✅ 60 Demo Expenses Loaded Successfully!")

        st.rerun()# -----------------------------------------------------
# SPENDING ANALYSIS
# -----------------------------------------------------

elif page == "Spending Analysis":

    st.title("Spending Analysis")

    expenses_df = pd.read_sql_query(

        """
        SELECT *
        FROM expense_records
        WHERE username = ?
        """,

        conn,

        params=(
            st.session_state.username,
        )

    )

    if expenses_df.empty:

        st.warning(
            "No expense data available. Please add expenses first."
        )

    else:

        total_expense = expenses_df[
            "amount"
        ].sum()
        category_count = expenses_df["category"].nunique()

        avg_expense = expenses_df["amount"].mean()

        highest_expense = expenses_df["amount"].max()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Expenses",
                f"₹ {total_expense:,.0f}"
            )

        with col2:
            st.metric(
                " Categories",
                category_count
            )

        with col3:
            st.metric(
                "Avg Expense",
                f"₹ {avg_expense:,.0f}"
            )

        with col4:
            st.metric(
                "Highest Expense",
                f"₹ {expenses_df['amount'].max():,.0f}"
            )
        st.markdown("## 🥧 Category Breakdown")

        category_data = expenses_df.groupby(
            "category"
        )["amount"].sum().sort_values(
            ascending=False
        )

        chart_col, legend_col = st.columns([1,1])

        colors = [
            "#EF4444",
            "#10B981",
            "#8B5CF6",
            "#F59E0B",
            "#06B6D4",
            "#EC4899",
            "#84CC16"
        ]

        with chart_col:

            fig1, ax1 = plt.subplots(
                figsize=(2.0,2.0)
            )

            ax1.pie(
                category_data.values,
                labels=None,
                startangle=90,
                colors=colors[:len(category_data)]
            )
            

            ax1.axis("equal")
            fig1.patch.set_facecolor("#0F172A")
            ax1.set_facecolor("#0F172A")

            st.pyplot(fig1)

        with legend_col:

            st.markdown("<br><br><br><br>", unsafe_allow_html=True)

            total = category_data.sum()

            for label, value, color in zip(
                category_data.index,
                category_data.values,
                colors
            ):

                percentage = round(
                    value / total * 100,
                    1
                )

                st.markdown(
                    f"""
            <div style="display:flex;align-items:center;margin-bottom:12px;">

            <div style="
            width:18px;
            height:18px;
            background:{color};
            margin-right:10px;
            border-radius:4px;
            ">
            </div>

            <div>
            <b>{label}</b> - {percentage}%
            </div>

            </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown(
            "## 📈 Monthly Spending Trend"
        )

        expenses_df["expense_date"] = pd.to_datetime(
            expenses_df["expense_date"]
        )

        expenses_df["month"] = expenses_df[
            "expense_date"
        ].dt.to_period("M")

        monthly_data = expenses_df.groupby(
            "month"
        )["amount"].sum()


        monthly_data.index = pd.to_datetime(
            monthly_data.index.astype(str)
        ).strftime("%b")
        left, center, right = st.columns([1,3,1])

        with center:

            fig2, ax2 = plt.subplots(
                figsize=(5,2.5)
            )

            ax2.plot(
                monthly_data.index,
                monthly_data.values,
                marker="o",
                linewidth=3,
                color=FINANCE_COLORS["expense"]
            )

            ax2.set_ylabel("Amount")
            ax2.set_xlabel("Month")
            fig2.patch.set_facecolor("#0F172A")
            ax2.set_facecolor("#0F172A")

            ax2.grid(alpha=0.3,color="#64748B")

            ax2.tick_params(colors="white")

            ax2.xaxis.label.set_color("white")
            ax2.yaxis.label.set_color("white")

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            ax2.spines['left'].set_color("#64748B")
            ax2.spines['bottom'].set_color("#64748B")
            st.pyplot(fig2)
        st.markdown(
                "## 🤖 AI Spending Insights"
            )

        top_category = category_data.idxmax()

        top_amount = category_data.max()

        avg_expense = expenses_df["amount"].mean()

        monthly_total = monthly_data.max()

        prompt = f"""

        You are a financial analyst.


        Provide response exactly like:

        🏆 Top Category: <text>

        📈 Observation: <text>

        💡 Recommendation: <text>

        ⚠️ Risk Level: <one word>

        Rules:
        1. Every point must be on a separate line.
        2. Do not write paragraph.
        3. Do not combine lines.
        4. Maximum 80 words.

        Data:

        Total Expenses: ₹{total_expense}

        Top Category: {top_category}

        Top Category Amount: ₹{top_amount}

        Average Expense: ₹{avg_expense:.2f}

        Highest Monthly Spending: ₹{monthly_total}

        """

        try:

            response = model.generate_content(
                prompt
            )

            text = response.text

            text = text.replace(
                "📈 Observation:",
                "\n\n📈 Observation:"
            )

            text = text.replace(
                "💡 Recommendation:",
                "\n\n💡 Recommendation:"
            )

            text = text.replace(
                "⚠️ Risk Level:",
                "\n\n⚠️ Risk Level:"
            )

            st.info(text)

        except Exception as e:
            st.error(
                f"Actual Error: {e}"
            )

                # --------------------------------------------------
        # SPENDING HIGHLIGHTS
        # --------------------------------------------------

        highest_month = monthly_data.idxmax()

        st.markdown("## 🏆 Spending Highlights")

        highest_expense = expenses_df[
            "amount"
        ].max()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Top Category",
                top_category
            )

        with col2:
            st.metric(
                "Highest Month",
                highest_month
            )

       
        # --------------------------------------------------
        # CATEGORY SUMMARY TABLE
        # --------------------------------------------------

        st.markdown(
            "## 📋 Category Summary"
        )

        summary_df = category_data.reset_index()

        summary_df.columns = [
            "Category",
            "Total Amount"
        ]

        st.dataframe(
            summary_df,
            use_container_width=True
        )
# -----------------------------------------------------
# RISK ANALYZER
# -----------------------------------------------------

elif page == "Risk Analyzer":

    st.title(" AI Financial Risk Analyzer")

    profile = st.session_state.get(
        "user_profile",
        {}
    )

    if not profile:

        st.warning(
            "Please create your financial profile first."
        )

    else:

        income = profile["income"]
        expenses = profile["expenses"]
        savings = profile["savings"]
        debt = profile["debt"]
        investments = profile["investments"]
        credit_score = profile["credit_score"]

        # RATIOS

        debt_ratio = debt / income if income > 0 else 0

        savings_ratio = savings / income if income > 0 else 0

        investment_ratio = investments / income if income > 0 else 0

        expense_ratio = expenses / income if income > 0 else 0

        # RISK SCORE

        risk_score = 0

        if debt_ratio > 1:
            risk_score += 40

        elif debt_ratio > 0.5:
            risk_score += 25

        else:
            risk_score += 10

        if savings_ratio < 1:
            risk_score += 25

        else:
            risk_score += 10

        if investment_ratio < 0.2:
            risk_score += 15

        else:
            risk_score += 5

        if credit_score < 600:
            risk_score += 30

        elif credit_score < 700:
            risk_score += 15

        else:
            risk_score += 5

        risk_score = min(risk_score, 100)

        # RISK LEVEL

        if risk_score >= 75:

            risk_level = "HIGH RISK"

            risk_color = "red"

        elif risk_score >= 45:

            risk_level = "MEDIUM RISK"

            risk_color = "orange"

        else:

            risk_level = "LOW RISK"

            risk_color = "green"

        # HEADER

        st.markdown(
            f"""
            <h2 style='color:{risk_color};'>
            {risk_level}
            </h2>
            """,
            unsafe_allow_html=True
        )

        # METRICS

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

        col2.metric(
            "Debt Ratio",
            round(debt_ratio, 2)
        )

        col3.metric(
            "Savings Ratio",
            round(savings_ratio, 2)
        )

        col4.metric(
            "Investment Ratio",
            round(investment_ratio, 2)
        )

        col5.metric(
            "Credit Score",
            credit_score
        )
        # PROGRESS BAR

        st.markdown("## 📉 Risk Meter")

        st.progress(
            risk_score / 100
        )
        st.markdown("## 📊 Risk Breakdown")

        risk_data = pd.DataFrame({
            "Factor": ["Debt", "Savings", "Investment", "Credit"],
            "Score": [
                25 if debt_ratio > 0.5 else 10,
                25 if savings_ratio < 1 else 10,
                15 if investment_ratio < 0.2 else 5,
                30 if credit_score < 600 else (15 if credit_score < 700 else 5)
            ]
        })

        left, center, right = st.columns([1,3,1])

        with center:

            fig, ax = plt.subplots(figsize=(5,3))

            ax.bar(
                risk_data["Factor"],
                risk_data["Score"],
                width=0.5,
                color="#F59E0B"

            )

            ax.set_ylabel("Risk Points")
            ax.set_xlabel("")

            plt.tight_layout()
            fig.patch.set_facecolor("#0F172A")
            ax.set_facecolor("#0F172A")

            ax.grid(axis='y',alpha=0.3,color="#64748B")

            ax.tick_params(colors="white")

            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            st.pyplot(
                fig,
                use_container_width=False
            )
        # RISK FACTORS

        st.markdown("## 🔍 Risk Factors")

        if debt_ratio > 1:

            st.error(
                "Debt is higher than income."
            )

        if expense_ratio > 0.7:

            st.warning(
                "Expenses are consuming most income."
            )

        if savings_ratio < 1:

            st.warning(
                "Savings are lower than recommended."
            )

        if investments < 500:

            st.info(
                "Investment amount is relatively low."
            )

        if credit_score < 650:

            st.error(
                "Credit score improvement needed."
            )

        # LOAN ELIGIBILITY

        st.markdown("## 🏦 Loan Eligibility")

        if risk_score < 40:

            st.success(
                "Eligible for premium loan offers."
            )

        elif risk_score < 70:

            st.warning(
                "Eligible for standard loan approval."
            )

        else:

            st.error(
                "Loan approval risk is high."
            )

        # AI RECOMMENDATIONS

        st.markdown("## 🤖 AI Recommendations")

        recommendations = []

        if debt_ratio > 0.5:

            recommendations.append(
                "Reduce existing debt obligations."
            )

        if savings_ratio < 1:

            recommendations.append(
                "Increase monthly savings."
            )

        if investments < 500:

            recommendations.append(
                "Improve investment portfolio."
            )

        if credit_score < 700:

            recommendations.append(
                "Improve credit score through timely payments."
            )

        if len(recommendations) == 0:

            recommendations.append(
                "Financial risk profile looks healthy."
            )

        for rec in recommendations:

            st.info(rec)

        
# -----------------------------------------------------
# FINANCIAL HEALTH PAGE
# -----------------------------------------------------

elif page == "Financial Health":

    st.title("AI Financial Health Engine")

    profile = st.session_state.get(
        "user_profile",
        {}
    )

    if not profile:

        st.warning(
            "Please create your financial profile first."
        )

    else:

        income = profile["income"]
        expenses = profile["expenses"]
        savings = profile["savings"]
        debt = profile["debt"]
        investments = profile["investments"]
        credit_score = profile["credit_score"]

        # RATIOS

        savings_ratio = savings / income if income > 0 else 0

        expense_ratio = expenses / income if income > 0 else 0

        debt_ratio = debt / income if income > 0 else 0

        investment_ratio = investments / income if income > 0 else 0

        # HEALTH SCORE

        health_score = (

            (savings_ratio * 35)

            + ((1 - expense_ratio) * 30)

            + ((1 - debt_ratio) * 20)

            + (investment_ratio * 10)

            + ((credit_score / 850) * 5)

        )

        health_score = max(
            0,
            min(100, round(health_score, 2))
        )

        # HEALTH STATUS

        if health_score >= 80:

            health_status = "EXCELLENT"

            color = "green"

        elif health_score >= 60:

            health_status = "GOOD"

            color = "orange"

        elif health_score >= 40:

            health_status = "AVERAGE"

            color = "gold"

        else:

            health_status = "POOR"

            color = "red"

        # HEADER

        st.markdown(
            f"""
            <h1 style='color:{color};'>
            {health_status}
            </h1>
            """,
            unsafe_allow_html=True
        )
        if health_score >= 80:
            health_grade = "A"

        elif health_score >= 60:
            health_grade = "B"

        elif health_score >= 40:
            health_grade = "C"

        else:
            health_grade = "D"

        # SCORE DISPLAY

        st.markdown("## 📊 Financial Health Score")

        st.progress(
            health_score / 100
        )

        st.write(
            f"Overall Score: {health_score}/100"
        )

        # KPI CARDS

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Savings Ratio",
            round(savings_ratio, 2)
        )

        col2.metric(
            "Debt Ratio",
            round(debt_ratio, 2)
        )

        col3.metric(
            "Investment Ratio",
            round(investment_ratio, 2)
        )

        col4.metric(
            "Credit Score",
            credit_score
        )

        col5.metric(
            "Health Grade",
            health_grade
        )
        # HEALTH CHART

        st.markdown("## 📈 Financial Health Breakdown")

        health_df = pd.DataFrame({

            "Category": [
                "Savings",
                "Expenses",
                "Debt",
                "Investments"
            ],

            "Score": [
                savings_ratio * 100,
                (1 - expense_ratio) * 100,
                (1 - debt_ratio) * 100,
                investment_ratio * 100
            ]
        })

        left, center, right = st.columns([1,3,1])

        with center:

            fig, ax = plt.subplots(
                figsize=(5,3)
            )

            ax.bar(
                health_df["Category"],
                health_df["Score"],
                width=0.5,
                color="#34D399"

            )

            ax.set_ylabel("Score")

            plt.tight_layout()
            fig.patch.set_facecolor("#0F172A")
            ax.set_facecolor("#0F172A")

            ax.grid(axis='y',alpha=0.3,color="#64748B")

            ax.tick_params(colors="white")

            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(
                fig,
                use_container_width=False
            )
        # AI INSIGHTS

        st.markdown("## 🧠 AI Financial Insights")

        insight_text = f"""

        Savings Ratio: {round(savings_ratio,2)}

        Debt Ratio: {round(debt_ratio,2)}

        Investment Ratio: {round(investment_ratio,2)}

        Credit Score: {credit_score}

        Overall Financial Health Score: {health_score}/100

        """

        if health_score >= 80:

            insight_text += """
            Financial profile is very strong with excellent long-term stability.
            """

        elif health_score >= 60:

            insight_text += """
            Financial profile is healthy but there is room for optimization.
            """

        elif health_score >= 40:

            insight_text += """
            Financial profile is average. Improving savings and investments can significantly increase financial strength.
            """

        else:

            insight_text += """
            Financial profile shows elevated stress levels. Immediate improvement in savings and debt management is recommended.
            """

        st.info(insight_text)
        # SMART RECOMMENDATIONS

        st.markdown("## 💡 Personalized Recommendations")

        recommendations = []

        if savings_ratio < 1:

            recommendations.append(
                "Increase monthly savings by 15-20%."
            )

        if expense_ratio > 0.7:

            recommendations.append(
                "Reduce unnecessary spending."
            )

        if investments < 500:

            recommendations.append(
                "Start SIP or long-term investments."
            )

        if debt_ratio > 0.5:

            recommendations.append(
                "Focus on debt repayment strategy."
            )

        if credit_score < 700:

            recommendations.append(
                "Improve credit history through timely payments."
            )

        if len(recommendations) == 0:

            recommendations.append(
                "Your financial profile is excellent."
            )

        for rec in recommendations:

            st.success(rec)

        # FUTURE FINANCIAL OUTLOOK

        st.markdown("## 🔮 Future Financial Outlook")

        if health_score >= 80:

            st.success(
                f"""
                Based on your current profile, financial stability is expected to remain strong.
                You are well-positioned for long-term wealth growth and investment expansion.
                """
            )

        elif health_score >= 60:

            st.info(
                f"""
                Your finances are stable. With moderate improvements in savings and investments,
                your health score can potentially reach Grade A within the next 12 months.
                """
            )

        elif health_score >= 40:

            st.warning(
                f"""
                Financial profile requires optimization.
                Increasing savings and reducing unnecessary expenses could improve your
                health score from {health_score} to above 60 over time.
                """
            )

        else:

            st.error(
                f"""
                High financial stress detected.
                Focus on debt reduction and savings growth immediately to improve future stability.
                """
            )
# -----------------------------------------------------
# FORECASTING PAGE
# -----------------------------------------------------

elif page == "Forecasting":

    st.title(
        "Financial Growth Simulator"
    )
    profile = st.session_state.get(
        "user_profile",
        {}
    )

    if not profile:

        st.warning(
            "Please create your financial profile first."
        )

    else:

        income = profile["income"]
        expenses = profile["expenses"]
        savings = profile["savings"]
        investments = profile["investments"]

        st.markdown(
            "## 📅 Future Financial Predictions"
        )

        months = np.arange(1, 13)

        # PREDICTIONS

        future_expenses = []

        future_savings = []

        future_wealth = []

        current_savings = savings

        for month in months:

            predicted_expense = expenses * (
        1 + np.random.uniform(0.005, 0.02) * month
    )
            predicted_saving = (

                income - predicted_expense

            )

            current_savings += (

                predicted_saving + investments

            )

            future_expenses.append(
                predicted_expense
            )

            future_savings.append(
                predicted_saving
            )

            future_wealth.append(
                current_savings
            )

        # FORECAST DATAFRAME

        forecast_df = pd.DataFrame({

            "Month": months,

            "Predicted Expenses": future_expenses,

            "Predicted Savings": future_savings,

            "Predicted Wealth": future_wealth

        })

        #st.dataframe(forecast_df)

        # EXPENSE FORECAST CHART

        st.markdown("## 📉 Expense Forecast")

        left, center, right = st.columns([1,3,1])

        with center:

            fig1, ax1 = plt.subplots(
                figsize=(5,3)
            )

            ax1.plot(
                months,
                future_expenses,
                marker="o",
                linewidth=3,
                color=FINANCE_COLORS["expense"]
            )

            ax1.set_xlabel("Month")
            ax1.set_ylabel("Expenses")

            ax1.spines['top'].set_visible(False)

            ax1.spines['right'].set_visible(False)
            fig1.patch.set_facecolor("#0F172A")
            ax1.set_facecolor("#0F172A")
            ax1.grid(alpha=0.3,color="#64748B")
            ax1.tick_params(colors="white")

            ax1.xaxis.label.set_color("white")

            ax1.yaxis.label.set_color("white")

            ax1.title.set_color("white")

            st.pyplot(fig1)

        # SAVINGS FORECAST CHART

        st.markdown("## 💰 Savings Forecast")

        left, center, right = st.columns([1,3,1])

        with center:

            fig2, ax2 = plt.subplots(
                figsize=(5,3)
            )

            ax2.plot(
                months,
                future_savings,
                marker="o",
                linewidth=3,
                color=FINANCE_COLORS["saving"]
            )

            ax2.set_xlabel("Month")
            ax2.set_ylabel("Savings")

            ax2.spines['top'].set_visible(False)

            ax2.spines['right'].set_visible(False)
            fig2.patch.set_facecolor("#0F172A")
            ax2.set_facecolor("#0F172A")
            ax2.grid(alpha=0.3,color="#64748B")
            ax2.tick_params(colors="white")

            ax2.xaxis.label.set_color("white")

            ax2.yaxis.label.set_color("white")

            ax2.title.set_color("white")

            st.pyplot(fig2)

        # WEALTH GROWTH CHART

        st.markdown("## 📈 Wealth Growth Prediction")

        left, center, right = st.columns([1,3,1])

        with center:

            fig3, ax3 = plt.subplots(
                figsize=(5,3)
            )

            ax3.plot(
                months,
                future_wealth,
                marker="o",
                linewidth=3,
                color=FINANCE_COLORS["investment"]
            )

            ax3.set_xlabel("Month")
            ax3.set_ylabel("Wealth")

            ax3.spines['top'].set_visible(False)

            ax3.spines['right'].set_visible(False)
            fig3.patch.set_facecolor("#0F172A")
            ax3.set_facecolor("#0F172A")
            ax3.grid(alpha=0.3,color="#64748B")
            ax3.tick_params(colors="white")

            ax3.xaxis.label.set_color("white")

            ax3.yaxis.label.set_color("white")

            ax3.title.set_color("white")

            st.pyplot(fig3)
        # AI FORECAST INSIGHTS
        growth_percent = (
            (future_wealth[-1] - savings)
            / savings
        ) * 100

        st.markdown("## 🧠 AI Forecast Insights")

        st.success(
            f"""
        📈 Projected Wealth:  ₹ {future_wealth[-1]:,.0f}

        🚀 Expected Growth: {growth_percent:.1f}%

        💡 Outlook: {'Strong' if growth_percent > 20 else 'Moderate'}
            """
        )
        # FUTURE FINANCIAL STATUS

        st.markdown("## 🚀 Future Financial Status")

        final_wealth = future_wealth[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "12 Month Wealth",
                f" ₹ {future_wealth[-1]:,.0f}"
            )

        with col2:
            st.metric(
                "Future Savings",
                f" ₹ {future_savings[-1]:,.0f}"
            )

        with col3:
            st.metric(
                "Future Expenses",
                f" ₹ {future_expenses[-1]:,.0f}"
            )

        if final_wealth > 100000:

            st.success(
                "Excellent future wealth projection."
            )

        elif final_wealth > 50000:

            st.info(
                "Good financial future predicted."
            )

        else:

            st.warning(
                "Financial growth needs improvement."
            )
        # AI RECOMMENDATIONS

        st.markdown("## 💡 Smart Forecast Recommendations")

        recommendations = []

        if expenses > income * 0.7:

            recommendations.append(
                "Reduce monthly expenses."
            )

        if investments < 500:

            recommendations.append(
                "Increase investment contributions."
            )

        if savings < income:

            recommendations.append(
                "Improve monthly savings discipline."
            )

        if len(recommendations) == 0:

            recommendations.append(
                "Future financial growth looks strong."
            )

        for rec in recommendations:

            st.success(rec)
# -----------------------------------------------------
# SMART RECOMMENDATIONS
# -----------------------------------------------------

elif page == "Smart Recommendations":

    st.title("AI Smart Financial Advisor")

    profile = st.session_state.get(
        "user_profile",
        {}
    )

    if not profile:

        st.warning(
            "Please create your financial profile first."
        )

    else:

        income = profile["income"]
        expenses = profile["expenses"]
        savings = profile["savings"]
        debt = profile["debt"]
        investments = profile["investments"]
        credit_score = profile["credit_score"]
        financial_goal = profile["financial_goal"]

        # RATIOS

        savings_ratio = savings / income if income > 0 else 0

        expense_ratio = expenses / income if income > 0 else 0

        debt_ratio = debt / income if income > 0 else 0

        investment_ratio = investments / income if income > 0 else 0

        # HEADER

        st.markdown(
            "## 💡 Personalized Financial Recommendations"
        )

        recommendations = []

        # SAVINGS RECOMMENDATIONS

        if savings_ratio < 1:

            recommendations.append({

                "title": "Increase Savings",

                "message":
                "Try saving at least 20% of monthly income.",

                "priority": "High"

            })

        # EXPENSE RECOMMENDATIONS

        if expense_ratio > 0.7:

            recommendations.append({

                "title": "Reduce Expenses",

                "message":
                "Your expenses are consuming most of your income.",

                "priority": "High"

            })

        # DEBT RECOMMENDATIONS

        if debt_ratio > 0.5:

            recommendations.append({

                "title": "Debt Reduction Plan",

                "message":
                "Focus on reducing high-interest debt first.",

                "priority": "Critical"

            })

        # INVESTMENT RECOMMENDATIONS

        if investments < 500:

            recommendations.append({

                "title": "Increase Investments",

                "message":
                "Consider SIPs, index funds, or long-term investing.",

                "priority": "Medium"

            })

        # CREDIT SCORE

        if credit_score < 700:

            recommendations.append({

                "title": "Improve Credit Score",

                "message":
                "Pay bills on time and reduce credit utilization.",

                "priority": "Medium"

            })

        # GOAL BASED RECOMMENDATIONS

        if financial_goal == "Buy House":

            recommendations.append({

                "title": "Home Planning",

                "message":
                "Increase down-payment savings and improve credit profile.",

                "priority": "High"

            })

        elif financial_goal == "Retirement Planning":

            recommendations.append({

                "title": "Retirement Investments",

                "message":
                "Increase long-term retirement contributions.",

                "priority": "High"

            })

        elif financial_goal == "Emergency Fund":

            recommendations.append({

                "title": "Emergency Fund",

                "message":
                "Maintain at least 6 months of expenses as emergency savings.",

                "priority": "Critical"

            })

        # DISPLAY RECOMMENDATIONS

        if len(recommendations) == 0:

            st.success(
                "Your financial profile looks excellent."
            )

        else:

            for rec in recommendations:

                if rec["priority"] == "Critical":

                    st.error(
                        f"""
                        {rec['title']}

                        {rec['message']}
                        """
                    )

                elif rec["priority"] == "High":

                    st.warning(
                        f"""
                        {rec['title']}

                        {rec['message']}
                        """
                    )

                else:

                    st.info(
                        f"""
                        {rec['title']}

                        {rec['message']}
                        """
                    )
        st.markdown("## 🎯 Suggested Monthly Targets")

        target_savings = income * 0.2

        target_investments = income * 0.15

        target_expenses = income * 0.5

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Target Savings",
            f" ₹{round(target_savings,2)}"
        )

        col2.metric(
            "Target Investments",
            f" ₹{round(target_investments,2)}"
        )

        col3.metric(
            "Recommended Expenses",
            f" ₹{round(target_expenses,2)}"
        )


        # BUDGET ALLOCATION

        st.markdown("## 💰 Recommended Budget Allocation")

        budget_df = pd.DataFrame({
            "Category": [
                "Needs",
                "Savings",
                "Investments",
                "Entertainment"
            ],
            "Recommended %": [
                50,
                20,
                20,
                10
            ]
        })

        colors = [
            "#3B82F6",
            "#10B981",
            "#8B5CF6",
            "#F59E0B"
        ]

        fig, ax = plt.subplots(figsize=(6,3))

        wedges, texts = ax.pie(
            budget_df["Recommended %"],
            labels=None,
            startangle=90,
            colors=colors
        )

        ax.legend(
            wedges,
            [
                f"{cat} - {pct}%"
                for cat, pct in zip(
                    budget_df["Category"],
                    budget_df["Recommended %"]
                )
            ],
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )

        plt.tight_layout()
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")


        st.pyplot(fig)
        # MONTHLY SAVINGS TARGET

        
        # FUTURE IMPROVEMENT SCORE

        st.markdown("## 🚀 Financial Improvement Potential")

        improvement_score = 100

        if expense_ratio > 0.7:
            improvement_score -= 25

        if debt_ratio > 0.5:
            improvement_score -= 30

        if savings_ratio < 1:
            improvement_score -= 20

        if investments < 500:
            improvement_score -= 15

        improvement_score = max(
            0,
            improvement_score
        )

        st.progress(
            improvement_score / 100
        )

        st.write(
            f"Financial Improvement Potential: {improvement_score}/100"
        )

        # AI SUMMARY

        st.markdown("## 🤖 AI Financial Summary")

        st.success(
            f"""
        📈 Financial Goal: {financial_goal}

        💪 Strongest Area:
        {'Credit Stability' if credit_score > 700 else 'Growth Potential'}

        🎯 Priority Focus:
        {'Expense Reduction' if expense_ratio > 0.7 else 'Investment Growth'}

        🚀 Improvement Potential:
        {improvement_score}/100
        """
        )
    # -----------------------------------------------------
# ANALYSIS HISTORY PAGE
# -----------------------------------------------------

elif page == "Analysis History":

    st.title("Financial Analysis History")
    

    # LOAD USER PROFILE

    profile = st.session_state.get(
        "user_profile",
        {}
    )

    if not profile:

        st.warning(
            "Please create your financial profile first."
        )

    else:
        # LOAD HISTORY

       
        history_df = pd.read_sql_query(

            '''

            SELECT *

            FROM analysis_history

            WHERE username = ?

            ORDER BY timestamp DESC

            ''',

            conn,

            params=(
                st.session_state.username,
            )

        )

        if history_df.empty:

            st.info(
                "No saved analysis history found."
            )

        else:
            latest_record = history_df.iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Records",
                    len(history_df)
                )

            with col2:
                st.metric(
                    "Latest Savings",
                    f"₹ {latest_record['savings']:,.0f}"
                )

            with col3:
                st.metric(
                    "Latest Credit Score",
                    int(latest_record['credit_score'])
                )
            st.markdown(
                "## 📊 Saved Financial Records"
            )

            st.dataframe(

                history_df[
                    [
                        "username",
                        "timestamp",
                        "income",
                        "expenses",
                        "savings",
                        "debt",
                        "investments",
                        "credit_score",
                        "financial_goal"
                    ]
                ],

                use_container_width=True

            )
            csv = history_df.to_csv(
                index=False
            )

            st.download_button(

                label="📥 Download History CSV",

                data=csv,

                file_name="analysis_history.csv",

                mime="text/csv"

            )
            # HISTORY CHARTS

            st.markdown("## 📈 Overall Financial Trend")

            history_df["timestamp"] = pd.to_datetime(
                history_df["timestamp"]
            )

            fig, ax = plt.subplots(figsize=(6,3))
            history_df["date"] = pd.to_datetime(
                history_df["timestamp"]
            ).dt.strftime("%d-%b")

            ax.plot(
                history_df["timestamp"],
                history_df["expenses"],
                marker='o',
                label="Expenses",
                color=FINANCE_COLORS["expense"]
            )

            ax.plot(
                history_df["timestamp"],
                history_df["savings"],
                marker='o',
                label="Savings",
                color=FINANCE_COLORS["saving"]
            )

            ax.plot(
                history_df["timestamp"],
                history_df["investments"],
                marker='o',
                label="Investments",
                color=FINANCE_COLORS["investment"]
            )

            # smaller labels
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            # rotate labels
            plt.xticks(rotation=30)

            # smaller legend
            ax.legend(fontsize=8)

            # smaller axis titles
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Amount", fontsize=9)

            plt.tight_layout()
            fig.patch.set_facecolor("#0F172A")

            ax.set_facecolor("#0F172A")

            ax.grid(alpha=0.3,color="#64748B")

            ax.tick_params(colors="white")
            ax.set_xlabel("Date", color="white")
            ax.set_ylabel("Amount", color="white")
            ax.set_title("Overall Financial Trend",color="white")


            ax.legend(
                facecolor="#0F172A",
                edgecolor="#64748B",
                labelcolor="white"
            )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.spines['left'].set_color("#64748B")
            ax.spines['bottom'].set_color("#64748B")
            st.pyplot(fig)
            # AI INSIGHTS

            st.markdown(
                "## 🧠 AI Trend Insights"
            )

            latest = history_df.iloc[0]

            oldest = history_df.iloc[-1]

            if latest["savings"] > oldest["savings"]:

                st.success(
                    "Savings trend is improving."
                )

            else:

                st.warning(
                    "Savings trend is decreasing."
                )

            if latest["expenses"] > oldest["expenses"]:

                st.warning(
                    "Expenses are increasing over time."
                )

            else:

                st.success(
                    "Expense management is improving."
                )

            if latest["investments"] > oldest["investments"]:

                st.success(
                    "Investment growth is positive."
                )

            else:

                st.info(
                    "Investment growth is stable."
                )
    # -----------------------------------------------------
# AI CHATBOT PAGE
# -----------------------------------------------------

elif page == "AI Chatbot":

    st.title("AI Financial Assistant")

    st.markdown(
        "Ask personalized finance questions based on your profile."
    )

    profile = st.session_state.get(
        "user_profile",
        {}
    )

    if not profile:

        st.warning(
            "Please create your financial profile first."
        )

    else:

        income = profile["income"]

        expenses = profile["expenses"]

        savings = profile["savings"]

        debt = profile["debt"]

        investments = profile["investments"]

        credit_score = profile["credit_score"]

        financial_goal = profile["financial_goal"]
        # LOAD RECENT ANALYSIS HISTORY

        history_df = pd.read_sql_query(

            """
            SELECT *

            FROM analysis_history

            WHERE username = ?

            ORDER BY timestamp DESC

            LIMIT 5
            """,

            conn,

            params=(
                st.session_state.username,
            )

        )

        history_summary = f"""
        Average Income: {history_df['income'].mean():.0f}
        Average Expenses: {history_df['expenses'].mean():.0f}
        Average Savings: {history_df['savings'].mean():.0f}
        Average Investments: {history_df['investments'].mean():.0f}
        """
        # CHAT HISTORY STORAGE

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        # CHAT INPUT

        user_question = st.text_input(
            "Ask your finance question"
        )

        if st.button("Generate AI Response"):

            if user_question:

                financial_context = f"""
                You are an expert financial advisor.

                User Financial Profile:

                Income: {income}
                Expenses: {expenses}
                Savings: {savings}
                Debt: {debt}
                Investments: {investments}
                Credit Score: {credit_score}
                Financial Goal: {financial_goal}

                Recent Financial History:

                {history_summary}

                User Question:

                {user_question}

                Instructions:

                - Analyze the user's profile and financial history before answering.
                - Choose the most suitable response format automatically.
                - If the question asks for a plan, provide step-by-step guidance.
                - If the question asks for advice, provide recommendations.
                - If the question asks for comparison, provide a comparison table or pros and cons.
                - If the question asks for a decision, explain reasoning and recommendation.
                - Keep answers concise and easy to read.
                - Use bullet points when helpful.
                - Use ₹ currency.
                - Prefer personalized answers using the user's financial profile and history.
                - Keep the response under 200 words.
                - Answer ONLY finance related questions.
                - If the question is unrelated to finance, respond:
                    "I am a Financial Assistant and can answer only finance-related questions."
                
                """
                try:

                    response = model.generate_content(
                        financial_context
                    )
                    st.session_state.chat_history.append(
                        {
                            "question": user_question,
                            "answer": response.text
                        }
                    )

                    st.markdown("## 🤖 AI Response")

                    st.markdown(response.text)
                    st.divider()

                except Exception as e:

                    st.error(
                        f"Actual Error: {str(e)}"
                    )

            else:

                st.warning(
                    "Please enter a question."
                )

        # CHAT HISTORY

        if len(st.session_state.chat_history) > 0:

            with st.expander(
                "💬 View Recent Conversations"
            ):

                for chat in reversed(
                    st.session_state.chat_history[-5:]
                ):

                    st.write(
                        f"**Q:** {chat['question']}"
                    )

                    st.write(
                        f"**A:** {chat['answer']}"
                    )

                    st.divider()