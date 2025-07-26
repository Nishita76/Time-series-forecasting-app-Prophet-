import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from prophet import Prophet


# Set page config
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

st.title(" Time Series Forecasting App (Prophet)")

# ------------------------ 1) Upload Data ------------------------
st.subheader("1️⃣ Upload Your Dataset (CSV or Excel)")

uploaded_file = st.file_uploader("Upload your time series file", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            import openpyxl  # Required for Excel
            df = pd.read_excel(uploaded_file)

        st.success("✅ Data Loaded Successfully!")

        # ------------------------ 2) Preview & Clean ------------------------
        st.subheader("2️⃣ Preview and Clean Data")

        if st.checkbox("Show first 5 rows"):
            st.write(df.head())

        if st.checkbox("Show column names & types"):
            st.write(df.dtypes)

        # if st.checkbox("Drop missing values (NaNs)"):
        #     df = df.dropna()
        #     st.write("✅ Null values dropped. New shape:", df.shape)

        # ------------------------ 3) Select Columns ------------------------
        st.subheader("3️⃣ Select Date & Target Columns")

        columns = df.columns.tolist()
        date_col = st.selectbox("📅 Select Date Column", options=columns)
        value_col = st.selectbox("📊 Select Value Column", options=columns)

        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(by=date_col)
            st.success("✅ Date column converted to datetime.")
        except Exception as e:
            st.error(f"")
        
        st.subheader("Missing Values Check")

        missing_date = df[date_col].isnull().sum()
        missing_target = df[value_col].isnull().sum()

        if missing_date > 0 or missing_target > 0:
            st.warning("Missing values detected in selected columns.")
    
            st.write(f"🔸 Missing in **Date** column: {missing_date}")
            st.write(f"🔸 Missing in **Target** column: {missing_target}")
    
            cleaning_option = st.selectbox(
                "Choose how to handle missing values:",
                ("Do nothing", "Drop rows", "Forward fill", "Backward fill")
  
            )

            if cleaning_option == "Drop rows":
                df = df.dropna(subset=[date_col, value_col])
                st.success("Dropped rows with missing date or target values.")

            elif cleaning_option == "Forward fill":
                df[date_col] = df[date_col].fillna(method='ffill')
                df[value_col] = df[value_col].fillna(method='ffill')
                st.success("Filled missing values using forward fill.")

            elif cleaning_option == "Backward fill":
                df[date_col] = df[date_col].fillna(method='bfill')
                df[value_col] = df[value_col].fillna(method='bfill')
                st.success("Filled missing values using backward fill.")
        else:
            st.success("No missing values in selected columns.")

        # Rename for Prophet compatibility
        df_prophet = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})

        st.write(" Preview Prophet-ready Data:")
        st.write(df_prophet.head())

        # ------------------------ 4) Raw Data Visualization ------------------------
        st.subheader("4️⃣ Visualize Raw Data")
        
        st.markdown("Visualize your raw time series data to check trends, seasonality, and patterns.")

        plot_type = st.selectbox(
            "Choose a plot type:",
           ["Line Plot", "Rolling Mean & Std", "Decomposition Plot", "Boxplot by Month", "Autocorrelation Plot"]
        )

        if plot_type == "Line Plot":
            fig = px.line(df_prophet, x='ds', y='y', title='Line Plot of Time Series')
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Rolling Mean & Std":
            window = st.slider("Select Rolling Window", min_value=3, max_value=30, value=12)
            df_prophet['rolling_mean'] = df_prophet['y'].rolling(window=window).mean()
            df_prophet['rolling_std'] = df_prophet['y'].rolling(window=window).std()
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_prophet['ds'], df_prophet['y'], label='Original')
            ax.plot(df_prophet['ds'], df_prophet['rolling_mean'], label='Rolling Mean', color='red')
            ax.plot(df_prophet['ds'], df_prophet['rolling_std'], label='Rolling Std', color='black')
            ax.legend()
            st.pyplot(fig)

        elif plot_type == "Decomposition Plot":
            period = st.number_input("Enter seasonal period (like 12 for monthly data)", value=12)
            decomposition = sm.tsa.seasonal_decompose(df_prophet['y'], model='additive', period=period)
            fig = decomposition.plot()
            fig.set_size_inches(10, 8)
            st.pyplot(fig)

        elif plot_type == "Boxplot by Month":
            df['month'] = pd.to_datetime(df_prophet['ds']).dt.month
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x='month', y='y', data=df, ax=ax)
            ax.set_title("Boxplot by Month")
            st.pyplot(fig)
            df.drop(columns='month', inplace=True)

        elif plot_type == "Autocorrelation Plot":
            fig, ax = plt.subplots(figsize=(8, 4))
            sm.graphics.tsa.plot_acf(df_prophet['y'], lags=30, ax=ax)
            st.pyplot(fig)

        # ------------------------ 5) Forecast Parameter configuration ------------------------
        # Step 5: Prophet Model Configuration
        st.subheader("4️⃣ Prophet Model Configuration")

        with st.expander("🔧 Customize Prophet Model Parameters"):
            st.markdown("#### Forecast Horizon")
            periods_input = st.number_input('How many future periods would you like to forecast?', 
                                    min_value=1, max_value=365, value=30)

            st.markdown("#### Seasonality Settings")
            col1, col2 = st.columns(2)
            with col1:
                seasonality_mode = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'])
                yearly_seasonality = st.checkbox("Enable Yearly Seasonality", value=True)
            with col2:
                weekly_seasonality = st.checkbox("Enable Weekly Seasonality", value=True)
                daily_seasonality = st.checkbox("Enable Daily Seasonality", value=False)

            st.markdown("#### Trend & Seasonality Flexibility")
            col3, col4 = st.columns(2)
            with col3:
                changepoint_prior_scale = st.slider("Trend Flexibility (Changepoint Prior Scale)", 
                                            min_value=0.001, max_value=0.5, value=0.05)
            with col4:
                seasonality_prior_scale = st.slider("Seasonality Prior Scale", 
                                            min_value=1.0, max_value=20.0, value=10.0)

            st.markdown("#### Holidays")
            include_holidays = st.checkbox("Include Holidays (optional)", value=False)

        
    except Exception as e:
        st.error(f"Error loading file: {e}")

        

