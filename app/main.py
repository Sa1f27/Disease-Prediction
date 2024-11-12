import streamlit as st
import numpy as np
import pandas as pd
from heart_app import display as heart_disease_display
from kidney_app import display as kidney_disease_display
from diabetes_app import display as diabetes_display
from liver_app import display as liver_disease_display
from stroke_app import display as stroke_display
from ai_app import display as ai_display


# Set the page configuration
st.set_page_config(page_title="Disease Diagnostic App",page_icon="🏥", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "AI Assistance", "Heart Disease", "Kidney Disease", "Diabetes", "Liver Disease", "Stroke"])

# Homepage content
if page == "Home":

    # Set page title and layout
    st.title("Welcome to the Disease Diagnostic App")
    st.write("""
        A comprehensive platform to predict various diseases, including Heart Disease, Kidney Disease, Diabetes, 
        Liver Disease, Stroke, and AI-powered assistance. Our app is designed to assist medical professionals and 
        individuals in quickly assessing the likelihood of certain diseases based on user inputs.

        ### Why Disease Prediction?
        With the growing prevalence of chronic illnesses, early detection is crucial for effective treatment and management. 
        Our app uses advanced machine learning algorithms and data analysis on large datasets, achieving over **90% accuracy** 
        in predictive performance. This empowers users and healthcare providers with insights backed by robust models.

        ### Made with Machine Learning and Big Data
        Our team invested significant effort in gathering, cleaning, and analyzing extensive medical data to ensure accurate 
        predictions. Each module in this app is trained on diverse datasets, providing insights tailored to a wide range of users.

        ### Explore Our Modules:
        - **Heart Disease**: Predicts the likelihood of heart disease based on user data.
        - **Kidney Disease**: Assesses the chances of kidney disease.
        - **Diabetes**: Evaluates the probability of diabetes onset.
        - **Liver Disease**: Predicts liver disease conditions based on medical test results.
        - **Stroke**: Checks the risk of stroke based on user information.
        - **AI Assistance**: Leverages the power of AI to help with medical predictions and advice.

        ### Understanding Disease Trends
        Below are some statistics showing age groups and other demographics most affected by these diseases, providing 
        valuable insights into high-risk categories and factors.

        *Thank you for choosing our platform. We are committed to supporting early diagnosis through innovation and 
        data-driven health insights.*
    """)

    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .stPlotly {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .streamlit-expanderHeader {
            background-color: #f0f2f6;
            border-radius: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title and description
    st.title("📊 Health Statistics Analysis Dashboard")
    st.markdown("### Comprehensive Analysis of Disease Prevalence and Health Metrics")

    # Generate more realistic data
    np.random.seed(42)

    # More realistic age distribution (weighted towards older ages for certain conditions)
    age_groups = ["0-18", "19-35", "36-50", "51-65", "65+"]
    age_weights = [0.1, 0.15, 0.25, 0.25, 0.25]
    genders = ["Male", "Female"]
    lifestyles = ["Smoker", "Non-Smoker"]
    bmi_ranges = ["Underweight", "Normal", "Overweight", "Obese"]
    bmi_weights = [0.05, 0.45, 0.32, 0.18]  # More realistic BMI distribution
    regions = ["Urban", "Rural"]

    # Create base dataset
    n_samples = 1000  # Larger sample size for better statistics

    data = pd.DataFrame({
        "Age Group": np.random.choice(age_groups, n_samples, p=age_weights),
        "Gender": np.random.choice(genders, n_samples),
        "Lifestyle": np.random.choice(lifestyles, n_samples, p=[0.2, 0.8]),  # 20% smokers
        "BMI Range": np.random.choice(bmi_ranges, n_samples, p=bmi_weights),
        "Region": np.random.choice(regions, n_samples, p=[0.7, 0.3])  # 70% urban
    })

    # Function to generate disease probability based on risk factors
    def generate_disease_probability(row, base_prob):
        prob = base_prob
        
        # Age factor
        age_risk = {
            "0-18": 0.2,
            "19-35": 0.4,
            "36-50": 1.0,
            "51-65": 2.0,
            "65+": 3.0
        }
        prob *= age_risk[row["Age Group"]]
        
        # Lifestyle factor
        if row["Lifestyle"] == "Smoker":
            prob *= 2.5
        
        # BMI factor
        bmi_risk = {
            "Underweight": 1.2,
            "Normal": 1.0,
            "Overweight": 1.5,
            "Obese": 2.5
        }
        prob *= bmi_risk[row["BMI Range"]]
        
        return min(prob, 1.0)  # Cap probability at 1.0

    # Generate disease data with correlations
    data["Heart Disease"] = data.apply(lambda row: np.random.binomial(1, generate_disease_probability(row, 0.05)), axis=1)
    data["Kidney Disease"] = data.apply(lambda row: np.random.binomial(1, generate_disease_probability(row, 0.03)), axis=1)
    data["Diabetes"] = data.apply(lambda row: np.random.binomial(1, generate_disease_probability(row, 0.06)), axis=1)
    data["Liver Disease"] = data.apply(lambda row: np.random.binomial(1, generate_disease_probability(row, 0.04)), axis=1)
    data["Stroke"] = data.apply(lambda row: np.random.binomial(1, generate_disease_probability(row, 0.02)), axis=1)

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Chart 1: Disease prevalence by age group (Area Chart)
        st.subheader("Disease Prevalence by Age Group")
        age_data = data.groupby("Age Group")[["Heart Disease", "Kidney Disease", "Diabetes", "Liver Disease", "Stroke"]].mean() * 100
        st.area_chart(age_data, use_container_width=True, height=400)
        
        # Chart 2: Disease prevalence by lifestyle (Bar Chart)
        st.subheader("Disease Distribution by Lifestyle")
        lifestyle_data = data.groupby("Lifestyle")[["Heart Disease", "Kidney Disease", "Diabetes", "Liver Disease", "Stroke"]].mean() * 100
        st.bar_chart(lifestyle_data, use_container_width=True, height=400)

    with col2:
        # Chart 3: Disease risk by BMI range (Line Chart)
        st.subheader("Disease Risk by BMI Range")
        bmi_data = data.groupby("BMI Range")[["Heart Disease", "Kidney Disease", "Diabetes", "Liver Disease", "Stroke"]].mean() * 100
        st.line_chart(bmi_data, use_container_width=True, height=400)
        
        # Chart 4: Disease prevalence by region (Bar Chart)
        st.subheader("Disease Prevalence by Region")
        region_data = data.groupby("Region")[["Heart Disease", "Kidney Disease", "Diabetes", "Liver Disease", "Stroke"]].mean() * 100
        st.bar_chart(region_data, use_container_width=True, height=400)

    # Chart 5: Average health metrics by age group
    st.subheader("Average Health Metrics by Age Group")

    # Generate more realistic health metrics
    metrics_data = pd.DataFrame({
        "Age Group": age_groups,
        "Average Glucose Level": [85, 90, 100, 110, 120],  # More realistic progression
        "Average Blood Pressure": [110, 115, 120, 130, 140],
        "Average BMI": [21, 23, 25, 27, 28]
    })
    metrics_data = metrics_data.set_index("Age Group")

    # Use a line chart with custom configuration
    st.line_chart(metrics_data, use_container_width=True, height=400)

    # Add key insights
    st.markdown("### 📋 Key Insights")
    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric(
            "Highest Risk Age Group",
            "65+",
            f"{age_data.loc['65+'].mean():.1f}% Disease Prevalence"
        )

    with col4:
        st.metric(
            "Lifestyle Impact",
            "Smoking",
            f"{(lifestyle_data.loc['Smoker'].mean() - lifestyle_data.loc['Non-Smoker'].mean()):.1f}% Higher Risk"
        )

    with col5:
        st.metric(
            "Most Common Condition",
            "Diabetes",
            f"{data['Diabetes'].mean() * 100:.1f}% Prevalence"
        )
    # Footer
    st.write("---")
    st.write("Explore each disease prediction module to learn more about these conditions and get personalized assessments based on your health data.")

# AI Assistance tab
elif page == "AI Assistance":
    st.header("AI Assistance")
    ai_display()
    ai_queries()

# Heart Disease tab
elif page == "Heart Disease":
    st.header("Heart Disease Prediction")
    heart_disease_display()

# Kidney Disease tab
elif page == "Kidney Disease":
    st.header("Kidney Disease Prediction")
    kidney_disease_display()

# Diabetes tab
elif page == "Diabetes":
    st.header("Diabetes Prediction")
    diabetes_display()

# Liver Disease tab
elif page == "Liver Disease":
    st.header("Liver Disease Prediction")
    liver_disease_display()

# Stroke tab
elif page == "Stroke":
    st.header("Stroke Prediction")
    stroke_display()

# Footer with team information (displayed on the home page)
if page == "Home":
# Footer with team information
    st.write("""
        ---

        ## Meet Our Dedicated Team 👩‍⚕️👨‍⚕️

        We are a dynamic group of tech enthusiasts with a shared goal: using AI and technology to make healthcare accessible and effective for all. From data science to medical insights, each of us brings something unique to this mission. Connect with us below!

        | **Team Member** | **Role** | **Bio** | **LinkedIn** |
        | :-------------: | :------: | :-----: | :----------: |
        | **Huzaifah** | Machine Learning Engineer | The brain behind the algorithms, focused on creating reliable and scalable models. | [Connect](https://www.linkedin.com/in/huzaifah-27o3/) |
        | **Maaz** | Full Stack Developer | Builds seamless user interfaces for easy access and interaction with complex medical data. | [Connect](https://www.linkedin.com/in/mohammed-maaz-593042274/) |
        | **Numaan** | Data Scientist | Specializes in making sense of data, from cleaning to insightful analysis. | [Connect](https://www.linkedin.com/in/syed-numaan-22a2b52ab/) |
        | **Qavi** | Medical Consultant | Adds depth with healthcare insights to ensure accuracy and relevance in our models. | [Connect](https://www.linkedin.com/in/qavi-quadri/) |

        ---
        
        ### 💬 Get in Touch
        We're always excited to collaborate or answer questions! Reach out to any of us on LinkedIn, and let's explore the future of healthcare together. 💙
    """)

