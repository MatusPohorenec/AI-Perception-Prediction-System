"""
AI Perception Prediction App - Individual & Company Analysis
Interactive Streamlit application for predicting AI adoption perceptions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import hmac

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="AI Perception Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.title("🔒 AI Perception Prediction System")
    st.text_input(
        "Enter password to access the app:", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# Check password before showing app
if not check_password():
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load saved regression and classification models."""
    models_path = Path("saved_models")
    
    reg_model = None
    class_models = None
    
    if (models_path / "regression_model.pkl").exists():
        with open(models_path / "regression_model.pkl", "rb") as f:
            reg_model = pickle.load(f)
    
    if (models_path / "classification_models.pkl").exists():
        with open(models_path / "classification_models.pkl", "rb") as f:
            class_models = pickle.load(f)
    
    return reg_model, class_models

@st.cache_resource
def load_correlations():
    """Load pre-computed correlation matrix."""
    models_path = Path("saved_models")
    
    if (models_path / "correlation_matrix.pkl").exists():
        with open(models_path / "correlation_matrix.pkl", "rb") as f:
            return pickle.load(f)
    return None

reg_model, class_models = load_models()
correlation_data = load_correlations()

# Feature definitions
INDIVIDUAL_FEATURES = {
    'job_position': {
        'label': '💼 Job Position',
        'options': {
            'Preparation/Budget Specialist': 1,
            'Designer/Architect': 2,
            'Technologist/Technician': 3,
            'Site Manager': 4,
            'Assistant': 5,
            'Manager': 6,
            'Executive/Owner': 7
        },
        'help': 'Select your current job position in the company'
    },
    'work_experience': {
        'label': '📅 Work Experience',
        'options': {
            'Less than 1 year': 1,
            '1-3 years': 2,
            '3-5 years': 3,
            '5-10 years': 4,
            'More than 10 years': 5
        },
        'help': 'How long have you been working in this field?'
    },
    'age': {
        'label': '🎂 Age Group',
        'options': {
            'Under 25': 1,
            '25-35 years': 2,
            '36-45 years': 3,
            '46-50 years': 4,
            'Over 50': 5
        },
        'help': 'Select your age group'
    },
    'digital_competencies': {
        'label': '💻 Digital Competencies',
        'min': 1,
        'max': 5,
        'default': 3,
        'help': 'Rate your digital skills (1=Basic, 5=Expert)'
    },
    'personal_ai_usage': {
        'label': '🤖 Personal AI Usage Level',
        'min': 1,
        'max': 5,
        'default': 2,
        'help': 'How often do you personally use AI tools? (1=Never, 5=Daily)'
    },
    'ai_training': {
        'label': '📚 AI Training Level',
        'options': {
            'No training': 1,
            'Self-taught basics': 2,
            'Online courses': 3,
            'Professional training': 4,
            'Advanced/Certified': 5
        },
        'help': 'What level of AI training have you received?'
    },
    'ict_utilization': {
        'label': '🖥️ ICT Utilization Level',
        'min': 1,
        'max': 5,
        'default': 3,
        'help': 'How much do you use ICT in your daily work? (1=Minimal, 5=Extensive)'
    }
}

COMPANY_FEATURES = {
    'company_size': {
        'label': '🏢 Company Size',
        'options': {
            'Micro (1-9 employees)': 1,
            'Small (10-49 employees)': 2,
            'Medium (50-249 employees)': 3,
            'Large (250+ employees)': 4
        },
        'help': 'Select the size of your company'
    },
    'digitalization_level': {
        'label': '📊 Company Digitalization Level',
        'min': 1,
        'max': 5,
        'default': 3,
        'help': 'How digitalized is your company? (1=Traditional, 5=Fully Digital)'
    },
    'company_ai_usage': {
        'label': '🤖 Company AI Usage Level',
        'min': 1,
        'max': 5,
        'default': 2,
        'help': 'To what extent does your company use AI? (1=Not at all, 5=Extensively)'
    }
}

AI_IMPACT_AREAS = {
    'productivity': {
        'label': '📈 Productivity Impact',
        'help': 'Expected impact on work productivity'
    },
    'job_security': {
        'label': '🔒 Job Security Impact',
        'help': 'Expected impact on job security (1=Very Negative, 5=Very Positive)'
    },
    'skill_requirements': {
        'label': '🎓 Skill Requirements Impact',
        'help': 'Expected change in required skills'
    },
    'work_quality': {
        'label': '✨ Work Quality Impact',
        'help': 'Expected impact on quality of work'
    },
    'cost_efficiency': {
        'label': '💰 Cost Efficiency Impact',
        'help': 'Expected impact on cost efficiency'
    }
}

TARGET_NAMES = [
    "Productivity Enhancement",
    "Job Security Perception", 
    "Skill Development Need",
    "Work Quality Improvement",
    "Cost Efficiency Gain",
    "Innovation Potential",
    "Overall AI Readiness"
]

def create_gauge_chart(value, title, max_val=5):
    """Create a gauge chart for displaying predictions."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [1, max_val], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 2], 'color': '#ff6b6b'},
                {'range': [2, 3], 'color': '#feca57'},
                {'range': [3, 4], 'color': '#54a0ff'},
                {'range': [4, 5], 'color': '#5f27cd'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_radar_chart(categories, values, title):
    """Create a radar chart for profile visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name=title,
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=False,
        title=dict(text=title, x=0.5),
        height=400
    )
    return fig

def create_bar_chart(categories, values, title):
    """Create a horizontal bar chart for predictions."""
    colors = ['#ff6b6b' if v < 2.5 else '#feca57' if v < 3.5 else '#54a0ff' if v < 4.5 else '#5f27cd' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.2f}' for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Score',
        xaxis=dict(range=[0, 5]),
        height=400,
        margin=dict(l=200)
    )
    return fig

def prepare_features(individual_data, company_data):
    """Prepare feature vector for model prediction."""
    # Create composite indices similar to data_prep.py
    ai_experience_index = (
        individual_data.get('personal_ai_usage', 3) * 0.4 +
        individual_data.get('ai_training', 3) * 0.3 +
        individual_data.get('digital_competencies', 3) * 0.3
    )
    
    digitalization_index = (
        company_data.get('digitalization_level', 3) * 0.5 +
        company_data.get('company_ai_usage', 3) * 0.3 +
        individual_data.get('ict_utilization', 3) * 0.2
    )
    
    # Calculate AI impact index from impact assessments
    impact_values = list(company_data.get('ai_impacts', {}).values())
    ai_impact_index = np.mean(impact_values) if impact_values else 3.0
    
    # Feature vector matching the model's expected features
    features = np.array([
        individual_data.get('age', 3),
        company_data.get('company_size', 2),
        individual_data.get('job_position', 4),
        individual_data.get('work_experience', 3),
        individual_data.get('ict_utilization', 3),
        individual_data.get('personal_ai_usage', 3),
        individual_data.get('digital_competencies', 3),
        company_data.get('company_ai_usage', 3),
        company_data.get('digitalization_level', 3),
        individual_data.get('ai_training', 3),
        company_data.get('ai_impacts', {}).get('productivity', 3),
        company_data.get('ai_impacts', {}).get('job_security', 3),
        company_data.get('ai_impacts', {}).get('skill_requirements', 3),
        company_data.get('ai_impacts', {}).get('work_quality', 3),
        company_data.get('ai_impacts', {}).get('cost_efficiency', 3),
    ]).reshape(1, -1)
    
    return features, ai_experience_index, digitalization_index, ai_impact_index

def make_predictions(features):
    """Make predictions using loaded models."""
    predictions = {}
    
    if reg_model is not None:
        try:
            reg_pred = reg_model.predict(features)[0]
            predictions['regression'] = float(reg_pred)
        except Exception as e:
            predictions['regression'] = None
    
    if class_models is not None:
        class_preds = {}
        for target_name, model in class_models.items():
            try:
                pred = model.predict(features)[0]
                class_preds[target_name] = int(pred)
            except Exception as e:
                class_preds[target_name] = None
        predictions['classification'] = class_preds
    
    return predictions

# Main app
st.markdown('<h1 class="main-header">🤖 AI Perception Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.markdown("## 🤖 AI Predictor")
    st.markdown("### About This App")
    st.markdown("""
    This application predicts AI adoption perceptions based on:
    - **Individual Profile**: Your personal characteristics, skills, and AI experience
    - **Company Profile**: Your organization's digital maturity and AI readiness
    
    The predictions are based on machine learning models trained on survey data from construction industry professionals.
    """)
    
    st.markdown("---")
    st.markdown("### Model Performance")
    st.metric("Regression R²", "0.501")
    st.metric("Classification F1", "0.681")
    
    st.markdown("---")
    st.markdown("### Hypothesis Under Test")
    st.info("*'For small companies, it's redundant to invest in AI solutions, and would hurt their finances.'*")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["👤 Individual Profile", "🏢 Company Profile", "📊 Comparison & Analysis", "🔬 Correlation Analysis"])

# Tab 1: Individual Profile
with tab1:
    st.markdown("## Individual Profile Assessment")
    st.markdown("Enter your personal characteristics to predict your AI perception profile.")
    
    col1, col2 = st.columns(2)
    
    individual_data = {}
    
    with col1:
        st.markdown("### 👤 Demographics")
        
        # Job Position
        job_options = list(INDIVIDUAL_FEATURES['job_position']['options'].keys())
        job_position = st.selectbox(
            INDIVIDUAL_FEATURES['job_position']['label'],
            options=job_options,
            index=3,
            help=INDIVIDUAL_FEATURES['job_position']['help']
        )
        individual_data['job_position'] = INDIVIDUAL_FEATURES['job_position']['options'][job_position]
        
        # Work Experience
        exp_options = list(INDIVIDUAL_FEATURES['work_experience']['options'].keys())
        work_experience = st.selectbox(
            INDIVIDUAL_FEATURES['work_experience']['label'],
            options=exp_options,
            index=2,
            help=INDIVIDUAL_FEATURES['work_experience']['help']
        )
        individual_data['work_experience'] = INDIVIDUAL_FEATURES['work_experience']['options'][work_experience]
        
        # Age
        age_options = list(INDIVIDUAL_FEATURES['age']['options'].keys())
        age = st.selectbox(
            INDIVIDUAL_FEATURES['age']['label'],
            options=age_options,
            index=2,
            help=INDIVIDUAL_FEATURES['age']['help']
        )
        individual_data['age'] = INDIVIDUAL_FEATURES['age']['options'][age]
    
    with col2:
        st.markdown("### 💡 Skills & Training")
        
        # Digital Competencies
        individual_data['digital_competencies'] = st.slider(
            INDIVIDUAL_FEATURES['digital_competencies']['label'],
            min_value=INDIVIDUAL_FEATURES['digital_competencies']['min'],
            max_value=INDIVIDUAL_FEATURES['digital_competencies']['max'],
            value=INDIVIDUAL_FEATURES['digital_competencies']['default'],
            help=INDIVIDUAL_FEATURES['digital_competencies']['help']
        )
        
        # Personal AI Usage
        individual_data['personal_ai_usage'] = st.slider(
            INDIVIDUAL_FEATURES['personal_ai_usage']['label'],
            min_value=INDIVIDUAL_FEATURES['personal_ai_usage']['min'],
            max_value=INDIVIDUAL_FEATURES['personal_ai_usage']['max'],
            value=INDIVIDUAL_FEATURES['personal_ai_usage']['default'],
            help=INDIVIDUAL_FEATURES['personal_ai_usage']['help']
        )
        
        # AI Training
        training_options = list(INDIVIDUAL_FEATURES['ai_training']['options'].keys())
        ai_training = st.selectbox(
            INDIVIDUAL_FEATURES['ai_training']['label'],
            options=training_options,
            index=1,
            help=INDIVIDUAL_FEATURES['ai_training']['help']
        )
        individual_data['ai_training'] = INDIVIDUAL_FEATURES['ai_training']['options'][ai_training]
        
        # ICT Utilization
        individual_data['ict_utilization'] = st.slider(
            INDIVIDUAL_FEATURES['ict_utilization']['label'],
            min_value=INDIVIDUAL_FEATURES['ict_utilization']['min'],
            max_value=INDIVIDUAL_FEATURES['ict_utilization']['max'],
            value=INDIVIDUAL_FEATURES['ict_utilization']['default'],
            help=INDIVIDUAL_FEATURES['ict_utilization']['help']
        )
    
    # Store in session state
    st.session_state['individual_data'] = individual_data
    
    # Individual Profile Visualization
    st.markdown("---")
    st.markdown("### 📊 Your Individual Profile")
    
    profile_categories = ['Job Position', 'Experience', 'Age', 'Digital Skills', 'AI Usage', 'AI Training', 'ICT Usage']
    profile_values = [
        individual_data['job_position'] / 7 * 5,  # Normalize to 5-point scale
        individual_data['work_experience'],
        individual_data['age'],
        individual_data['digital_competencies'],
        individual_data['personal_ai_usage'],
        individual_data['ai_training'],
        individual_data['ict_utilization']
    ]
    
    col1, col2 = st.columns([1, 1])
    with col1:
        radar_fig = create_radar_chart(profile_categories, profile_values, "Individual Competency Profile")
        st.plotly_chart(radar_fig, key="individual_radar")
    
    with col2:
        # AI Readiness Score
        ai_experience_index = (
            individual_data['personal_ai_usage'] * 0.4 +
            individual_data['ai_training'] * 0.3 +
            individual_data['digital_competencies'] * 0.3
        )
        
        st.markdown("### 🎯 Individual AI Readiness")
        gauge_fig = create_gauge_chart(ai_experience_index, "AI Experience Index")
        st.plotly_chart(gauge_fig, key="individual_gauge")
        
        # Interpretation
        if ai_experience_index < 2:
            st.error("🔴 **Low AI Readiness**: Consider basic AI training to improve your competitiveness.")
        elif ai_experience_index < 3:
            st.warning("🟡 **Moderate AI Readiness**: You have foundational knowledge but could benefit from more hands-on experience.")
        elif ai_experience_index < 4:
            st.info("🔵 **Good AI Readiness**: You're well-prepared for AI adoption in your workplace.")
        else:
            st.success("🟢 **Excellent AI Readiness**: You're an AI champion who can lead adoption initiatives!")

# Tab 2: Company Profile
with tab2:
    st.markdown("## Company Profile Assessment")
    st.markdown("Enter your company's characteristics and expected AI impacts.")
    
    col1, col2 = st.columns(2)
    
    company_data = {}
    
    with col1:
        st.markdown("### 🏢 Company Characteristics")
        
        # Company Size
        size_options = list(COMPANY_FEATURES['company_size']['options'].keys())
        company_size = st.selectbox(
            COMPANY_FEATURES['company_size']['label'],
            options=size_options,
            index=1,
            help=COMPANY_FEATURES['company_size']['help']
        )
        company_data['company_size'] = COMPANY_FEATURES['company_size']['options'][company_size]
        
        # Digitalization Level
        company_data['digitalization_level'] = st.slider(
            COMPANY_FEATURES['digitalization_level']['label'],
            min_value=COMPANY_FEATURES['digitalization_level']['min'],
            max_value=COMPANY_FEATURES['digitalization_level']['max'],
            value=COMPANY_FEATURES['digitalization_level']['default'],
            help=COMPANY_FEATURES['digitalization_level']['help']
        )
        
        # Company AI Usage
        company_data['company_ai_usage'] = st.slider(
            COMPANY_FEATURES['company_ai_usage']['label'],
            min_value=COMPANY_FEATURES['company_ai_usage']['min'],
            max_value=COMPANY_FEATURES['company_ai_usage']['max'],
            value=COMPANY_FEATURES['company_ai_usage']['default'],
            help=COMPANY_FEATURES['company_ai_usage']['help']
        )
    
    with col2:
        st.markdown("### 📈 Expected AI Impact Assessment")
        st.markdown("Rate the expected impact of AI in these areas (1=Very Negative, 5=Very Positive):")
        
        ai_impacts = {}
        for key, info in AI_IMPACT_AREAS.items():
            ai_impacts[key] = st.slider(
                info['label'],
                min_value=1,
                max_value=5,
                value=3,
                help=info['help'],
                key=f"impact_{key}"
            )
        company_data['ai_impacts'] = ai_impacts
    
    # Store in session state
    st.session_state['company_data'] = company_data
    
    # Company Profile Visualization
    st.markdown("---")
    st.markdown("### 📊 Company AI Impact Profile")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        impact_categories = list(AI_IMPACT_AREAS.keys())
        impact_labels = [AI_IMPACT_AREAS[k]['label'].replace(' Impact', '') for k in impact_categories]
        impact_values = [ai_impacts[k] for k in impact_categories]
        
        radar_fig = create_radar_chart(impact_labels, impact_values, "Expected AI Impact Profile")
        st.plotly_chart(radar_fig, key="company_radar")
    
    with col2:
        # Digitalization Index
        digitalization_index = (
            company_data['digitalization_level'] * 0.5 +
            company_data['company_ai_usage'] * 0.3 +
            st.session_state.get('individual_data', {}).get('ict_utilization', 3) * 0.2
        )
        
        st.markdown("### 🎯 Company Digital Maturity")
        gauge_fig = create_gauge_chart(digitalization_index, "Digitalization Index")
        st.plotly_chart(gauge_fig, key="company_gauge")
        
        # Company Size Analysis
        size_labels = {1: "Micro", 2: "Small", 3: "Medium", 4: "Large"}
        st.markdown(f"**Company Size**: {size_labels[company_data['company_size']]}")
        
        if company_data['company_size'] <= 2:
            st.warning("⚠️ **Small Company Considerations**: AI investments may require careful ROI analysis.")
        else:
            st.info("ℹ️ **Larger Company**: Better positioned for AI infrastructure investments.")

# Tab 3: Comparison & Analysis
with tab3:
    st.markdown("## Prediction & Hypothesis Testing")
    
    # Get data from session state
    ind_data = st.session_state.get('individual_data', {})
    comp_data = st.session_state.get('company_data', {})
    
    if not ind_data or not comp_data:
        st.warning("⚠️ Please complete both Individual and Company profiles first.")
    else:
        # Make predictions
        features, ai_exp_idx, dig_idx, impact_idx = prepare_features(ind_data, comp_data)
        predictions = make_predictions(features)
        
        st.markdown("### 🔮 Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Regression Prediction")
            if predictions.get('regression') is not None:
                reg_val = predictions['regression']
                # Clamp to reasonable range
                reg_val = max(1, min(5, reg_val))
                st.plotly_chart(
                    create_gauge_chart(reg_val, "Overall AI Perception Score"),
                    key="reg_prediction_gauge"
                )
            else:
                st.info("Regression model not available.")
        
        with col2:
            st.markdown("#### Classification Predictions")
            if predictions.get('classification'):
                class_preds = predictions['classification']
                valid_preds = {k: v for k, v in class_preds.items() if v is not None}
                
                if valid_preds:
                    categories = list(valid_preds.keys())
                    values = list(valid_preds.values())
                    
                    bar_fig = create_bar_chart(categories, values, "Predicted Impact Categories")
                    st.plotly_chart(bar_fig, key="class_prediction_bar")
                else:
                    st.info("No classification predictions available.")
            else:
                st.info("Classification models not available.")
        
        st.markdown("---")
        
        # Hypothesis Testing Section
        st.markdown("### 🧪 Hypothesis Analysis: Small Company AI Investment")
        
        st.markdown("""
        **Hypothesis**: *"For small companies, it's redundant to invest in AI solutions, and would hurt their finances."*
        """)
        
        # Simulate small vs large company scenarios
        col1, col2, col3 = st.columns(3)
        
        # Keep individual data same, vary company size
        small_company_data = comp_data.copy()
        small_company_data['company_size'] = 1  # Micro
        small_company_data['digitalization_level'] = 2
        small_company_data['company_ai_usage'] = 1
        
        medium_company_data = comp_data.copy()
        medium_company_data['company_size'] = 3  # Medium
        medium_company_data['digitalization_level'] = 3
        medium_company_data['company_ai_usage'] = 3
        
        large_company_data = comp_data.copy()
        large_company_data['company_size'] = 4  # Large
        large_company_data['digitalization_level'] = 4
        large_company_data['company_ai_usage'] = 4
        
        scenarios = [
            ("Micro/Small Company", small_company_data),
            ("Medium Company", medium_company_data),
            ("Large Company", large_company_data)
        ]
        
        scenario_results = []
        
        for i, (name, comp_scenario) in enumerate(scenarios):
            features_scenario, ai_exp, dig, impact = prepare_features(ind_data, comp_scenario)
            preds = make_predictions(features_scenario)
            
            reg_score = preds.get('regression', 3)
            if reg_score is None:
                reg_score = 3
            reg_score = max(1, min(5, reg_score))
            
            scenario_results.append({
                'name': name,
                'score': reg_score,
                'digitalization': dig,
                'features': features_scenario
            })
            
            cols = [col1, col2, col3]
            with cols[i]:
                st.markdown(f"**{name}**")
                mini_gauge = create_gauge_chart(reg_score, "AI Perception")
                st.plotly_chart(mini_gauge, key=f"scenario_{i}_gauge")
                st.metric("Digitalization Index", f"{dig:.2f}")
        
        # Analysis
        st.markdown("---")
        st.markdown("### 📋 Analysis Results")
        
        scores = [r['score'] for r in scenario_results]
        small_score = scores[0]
        large_score = scores[2]
        score_diff = large_score - small_score
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Comparison chart
            fig = go.Figure(data=[
                go.Bar(
                    x=[r['name'] for r in scenario_results],
                    y=[r['score'] for r in scenario_results],
                    marker_color=['#ff6b6b', '#feca57', '#54a0ff'],
                    text=[f"{s:.2f}" for s in scores],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="AI Perception Score by Company Size",
                yaxis_title="Predicted Score",
                yaxis=dict(range=[0, 5]),
                height=350
            )
            st.plotly_chart(fig, key="comparison_bar")
        
        with col2:
            st.markdown("#### Key Findings")
            
            if score_diff > 0.5:
                st.success(f"""
                ✅ **Finding**: Large companies show **{score_diff:.2f} points higher** AI perception scores.
                
                **Interpretation**: The hypothesis has **partial support** - smaller companies 
                may face more challenges with AI adoption, but this doesn't mean AI investment 
                is redundant. Rather, it suggests:
                
                1. Small companies need more targeted AI solutions
                2. ROI expectations should be adjusted for scale
                3. Cloud-based AI services may be more appropriate than infrastructure investments
                """)
            elif score_diff > 0:
                st.info(f"""
                ℹ️ **Finding**: Modest difference of **{score_diff:.2f} points** between company sizes.
                
                **Interpretation**: The hypothesis is **not strongly supported**. AI perception 
                varies more by individual readiness than company size.
                """)
            else:
                st.warning(f"""
                ⚠️ **Finding**: Unexpectedly, smaller companies show similar or better AI perception.
                
                **Interpretation**: The hypothesis is **not supported** by this analysis. 
                Small companies may be more agile in AI adoption.
                """)
        
        # ROI Considerations
        st.markdown("---")
        st.markdown("### 💰 ROI Considerations by Company Size")
        
        roi_data = pd.DataFrame({
            'Company Size': ['Micro', 'Small', 'Medium', 'Large'],
            'Typical AI Investment': ['$5K-20K', '$20K-100K', '$100K-500K', '$500K+'],
            'Expected ROI Timeline': ['6-12 months', '12-18 months', '18-24 months', '24-36 months'],
            'Recommended AI Solutions': [
                'SaaS tools, ChatGPT, simple automation',
                'Cloud AI services, process automation',
                'Custom ML models, integrated solutions',
                'Enterprise AI platforms, R&D investments'
            ],
            'Risk Level': ['Medium-High', 'Medium', 'Medium-Low', 'Low']
        })
        
        st.dataframe(roi_data, hide_index=True, use_container_width=True)
        
        st.markdown("""
        **Conclusion**: Rather than being "redundant," AI investment for small companies should be 
        **strategic and proportionate**. The key is matching AI solutions to company resources 
        and focusing on high-ROI, low-risk applications first.
        """)

# Tab 4: Correlation Analysis
with tab4:
    st.markdown("## 🔬 Correlation Analysis & Feature Importance")
    st.markdown("Understand which factors have the strongest relationships with AI perception outcomes.")
    
    if correlation_data is not None:
        corr_matrix = correlation_data['correlation_matrix']
        columns = correlation_data['columns']
        
        # Rename columns for better display
        display_names = {
            'Age_Numeric': 'Age',
            'Experience_Numeric': 'Work Experience',
            'Company_Size_Numeric': 'Company Size',
            'Job_Position_Numeric': 'Job Position',
            'ICT_Utilization_Numeric': 'ICT Utilization',
            'AI_Util_Personal_Numeric': 'Personal AI Usage',
            'Digital_Competencies_Numeric': 'Digital Competencies',
            'AI_Util_Company_Numeric': 'Company AI Usage',
            'Digitalization_Level_Numeric': 'Digitalization Level',
            'AI_Training_Numeric': 'AI Training',
            'AI_Impact_Budgeting': 'AI Impact: Budgeting',
            'AI_Impact_Design': 'AI Impact: Design',
            'AI_Impact_ProjectMgmt': 'AI Impact: Project Mgmt',
            'AI_Impact_Marketing': 'AI Impact: Marketing',
            'AI_Impact_Logistics': 'AI Impact: Logistics',
            'Perception_CostReduction_Numeric': 'Perception: Cost Reduction',
            'Perception_Automation_Numeric': 'Perception: Automation',
            'Perception_Materials_Numeric': 'Perception: Materials',
            'Perception_ProjectMonitor_Numeric': 'Perception: Project Monitor',
            'Perception_HR_Numeric': 'Perception: HR',
            'Perception_Admin_Numeric': 'Perception: Admin',
            'Perception_Planning_Numeric': 'Perception: Planning',
            'AI_Experience_Index': 'AI Experience Index',
            'Digitalization_Index': 'Digitalization Index',
            'AI_Impact_Index': 'AI Impact Index',
            'Importance of cost monitoring': 'Importance: Cost Monitoring',
            'Importance of Schedule plan': 'Importance: Schedule Plan'
        }
        
        # Filter to key variables
        key_vars = [
            'Age_Numeric', 'Experience_Numeric', 'Company_Size_Numeric', 'Job_Position_Numeric',
            'ICT_Utilization_Numeric', 'AI_Util_Personal_Numeric', 'Digital_Competencies_Numeric',
            'AI_Util_Company_Numeric', 'Digitalization_Level_Numeric', 'AI_Training_Numeric',
            'AI_Experience_Index', 'Digitalization_Index', 'AI_Impact_Index',
            'Perception_CostReduction_Numeric', 'Perception_Automation_Numeric'
        ]
        
        available_vars = [v for v in key_vars if v in corr_matrix.columns]
        filtered_corr = corr_matrix.loc[available_vars, available_vars]
        
        # Rename for display
        filtered_corr.index = [display_names.get(c, c) for c in filtered_corr.index]
        filtered_corr.columns = [display_names.get(c, c) for c in filtered_corr.columns]
        
        st.markdown("### 📈 Correlation Heatmap")
        st.markdown("This heatmap shows how strongly different factors are correlated. **Red** = positive correlation, **Blue** = negative correlation.")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=filtered_corr.values,
            x=filtered_corr.columns,
            y=filtered_corr.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(filtered_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            height=700,
            xaxis={'tickangle': 45},
            margin=dict(l=150, b=150)
        )
        
        st.plotly_chart(fig, key="correlation_heatmap", use_container_width=True)
        
        # Top correlations table
        st.markdown("---")
        st.markdown("### 🔝 Strongest Correlations")
        
        col1, col2 = st.columns(2)
        
        # Calculate top correlations
        corr_pairs = []
        for i in range(len(filtered_corr.columns)):
            for j in range(i+1, len(filtered_corr.columns)):
                corr_pairs.append({
                    'Factor 1': filtered_corr.columns[i],
                    'Factor 2': filtered_corr.columns[j],
                    'Correlation': filtered_corr.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
        
        with col1:
            st.markdown("#### 🟢 Strongest Positive Correlations")
            positive_corr = corr_df[corr_df['Correlation'] > 0].head(10)[['Factor 1', 'Factor 2', 'Correlation']]
            st.dataframe(positive_corr, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔴 Strongest Negative Correlations")
            negative_corr = corr_df[corr_df['Correlation'] < 0].head(10)[['Factor 1', 'Factor 2', 'Correlation']]
            if len(negative_corr) > 0:
                st.dataframe(negative_corr, hide_index=True, use_container_width=True)
            else:
                st.info("No significant negative correlations found.")
        
        # Key Insights
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        st.success("""
        **Main Findings from Correlation Analysis:**
        
        1. **AI Experience Index** is strongly driven by:
           - Personal AI Usage (r = 0.90)
           - Company AI Usage (r = 0.86)
           - AI Training (r = 0.76)
        
        2. **Digitalization Index** is most influenced by:
           - ICT Utilization (r = 0.84)
           - Digital Competencies (r = 0.68)
        
        3. **AI Impact perception** correlates strongly with:
           - AI Impact on Budgeting (r = 0.88)
           - AI Impact on Project Management (r = 0.87)
           - AI Impact on Design (r = 0.85)
        
        4. **Company Size shows weak correlations** with most perception variables, suggesting that 
           individual readiness matters more than company scale for AI adoption success.
        """)
        
        # Feature Importance from Model
        st.markdown("---")
        st.markdown("### 🎯 Feature Importance (Model-Based)")
        
        if reg_model is not None:
            try:
                # Get feature importance from Lasso coefficients
                if hasattr(reg_model, 'named_steps'):
                    model = reg_model.named_steps.get('lasso') or reg_model.named_steps.get('ridge')
                    if model is not None and hasattr(model, 'coef_'):
                        coefficients = model.coef_
                        feature_names = [
                            'Age', 'Company Size', 'Job Position', 'Work Experience',
                            'ICT Utilization', 'Personal AI Usage', 'Digital Competencies',
                            'Company AI Usage', 'Digitalization Level', 'AI Training',
                            'AI Impact: Productivity', 'AI Impact: Job Security',
                            'AI Impact: Skills', 'AI Impact: Work Quality', 'AI Impact: Cost'
                        ]
                        
                        # Match lengths
                        if len(coefficients) == len(feature_names):
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Coefficient': coefficients,
                                'Abs_Importance': np.abs(coefficients)
                            }).sort_values('Abs_Importance', ascending=True)
                            
                            # Create horizontal bar chart
                            colors = ['#ff6b6b' if c < 0 else '#54a0ff' for c in importance_df['Coefficient']]
                            
                            fig_importance = go.Figure(go.Bar(
                                x=importance_df['Coefficient'],
                                y=importance_df['Feature'],
                                orientation='h',
                                marker_color=colors,
                                text=[f'{c:.3f}' for c in importance_df['Coefficient']],
                                textposition='auto'
                            ))
                            
                            fig_importance.update_layout(
                                title='Feature Coefficients (Lasso Regression)',
                                xaxis_title='Coefficient Value',
                                yaxis_title='Feature',
                                height=500
                            )
                            
                            st.plotly_chart(fig_importance, key="feature_importance", use_container_width=True)
                            
                            st.markdown("""
                            **Interpretation**: 
                            - 🔵 **Blue bars** (positive coefficients) increase the prediction
                            - 🔴 **Red bars** (negative coefficients) decrease the prediction
                            - **Longer bars** indicate stronger influence on the outcome
                            """)
            except Exception as e:
                st.warning(f"Could not extract feature importance: {e}")
    else:
        st.warning("Correlation data not available. Please run `compute_correlations.py` first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>AI Perception Prediction System | Built with Streamlit & scikit-learn</p>
    <p>Model trained on construction industry survey data (n=52)</p>
</div>
""", unsafe_allow_html=True)
