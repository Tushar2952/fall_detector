# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, r2_score, mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from plotly.subplots import make_subplots

# Set global styles
sns.set(style="whitegrid")
st.set_page_config(page_title="Fall Detection Analytics Dashboard", layout="wide")

# --- Sidebar ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Data Exploration", "Model Comparison", "Predictive Capabilities"])

# --- Section: Data Exploration ---
if section == "Data Exploration":
    st.title("📊 Data Exploration")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file:
        excel_data = pd.ExcelFile(uploaded_file)
        st.write("Available Sheets:", excel_data.sheet_names)

        @st.cache_data
        def load_and_clean_data(_excel_file):
            data_in = _excel_file.parse('Data In')
            data = data_in.iloc[3:, :13].copy()
            data.columns = data_in.iloc[2, :13].values
            data = data[~data['TIME'].isin(['TIME', 'Historical Data'])]
            data = data.dropna(how='all')
            data.reset_index(drop=True, inplace=True)

            def extract_value(cell):
                try:
                    return float(str(cell).split(":")[1])
                except:
                    return np.nan

            df = pd.DataFrame()
            df['Timestamp'] = pd.to_datetime(data['TIME'])
            col_map = {
                "CH1": "AccX", "CH2": "AccY", "CH3": "AccZ", "CH4": "GyroX",
                "CH5": "GyroY", "CH6": "GyroZ", "CH7": "AccMag", "CH8": "GyroMag",
                "CH9": "Distance", "CH10": "Pressure", "CH11": "Altitude", "CH12": "Status"
            }
            for ch, name in col_map.items():
                if name != "Status":
                    df[name] = data[ch].apply(extract_value)
                else:
                    df[name] = data[ch].astype(str).str.extract(r'Status:(.*)')

            df['FallDetected'] = df['Status'].str.contains('Fall', case=False, na=False).astype(int)
            return df

        df = load_and_clean_data(_excel_file=excel_data)
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df.head())

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df, x='Timestamp', y='AccMag', label='AccMag')
        sns.lineplot(data=df, x='Timestamp', y='GyroMag', label='GyroMag')
        st.pyplot(fig)

        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.drop(columns=['Timestamp', 'Status']).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

# --- Section: Model Comparison ---
elif section == "Model Comparison":
    st.title("⚙️ Model Comparison")
    st.markdown("Compare Random Forest and XGBoost on synthetic fall data.")

    @st.cache_data
    def generate_synthetic_data():
        n_samples = 1000
        features = {
            'AccMag': np.concatenate([np.random.normal(1.0, 0.2, n_samples//2),
                                      np.random.normal(3.5, 1.0, n_samples//2)]),
            'JerkMag': np.concatenate([np.random.normal(0.5, 0.1, n_samples//2),
                                       np.random.normal(2.5, 0.8, n_samples//2)]),
            'GyroMag': np.concatenate([np.random.normal(0.8, 0.2, n_samples//2),
                                       np.random.normal(3.0, 1.0, n_samples//2)]),
            'Pressure': np.random.normal(101325, 100, n_samples),
            'Altitude': np.random.normal(50, 20, n_samples)
        }
        df = pd.DataFrame(features)
        df['Label'] = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
        return df

    df = generate_synthetic_data()
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    from imblearn.pipeline import Pipeline as ImbPipeline
    pipeline_rf = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('select', SelectKBest(score_func=f_classif, k='all')),
        ('clf', RandomForestClassifier(random_state=42))
    ])
            pipeline_xgb = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('select', SelectKBest(score_func=f_classif, k='all')),
        ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    pipeline_rf.fit(X_train, y_train)
    pipeline_xgb.fit(X_train, y_train)
    st.success("Both models trained.")

    def plot_roc(model, X_test, y_test, label):
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        fig = px.area(x=fpr, y=tpr, title=f"ROC Curve - {label} (AUC={auc_score:.2f})",
                      labels=dict(x='False Positive Rate', y='True Positive Rate'))
        st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Random Forest")
        st.text(classification_report(y_test, pipeline_rf.predict(X_test)))
        plot_roc(pipeline_rf, X_test, y_test, "Random Forest")
    with col2:
        st.write("### XGBoost")
        st.text(classification_report(y_test, pipeline_xgb.predict(X_test)))
        plot_roc(pipeline_xgb, X_test, y_test, "XGBoost")

# --- Section: Predictive Capabilities ---
elif section == "Predictive Capabilities":
    st.title("🤖 Predictive Capabilities")

    def generate_advanced_data():
        n = 2000
        df = pd.DataFrame({
            'AccX': np.random.normal(0, 1, n),
            'AccY': np.random.normal(0, 1, n),
            'AccZ': np.random.normal(9.8, 0.5, n),
            'GyroX': np.random.normal(0, 0.5, n),
            'GyroY': np.random.normal(0, 0.5, n),
            'GyroZ': np.random.normal(0, 0.5, n),
            'Pressure': np.random.normal(101325, 100, n)
        })
        df['AccMag'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
        df['GyroMag'] = np.sqrt(df['GyroX']**2 + df['GyroY']**2 + df['GyroZ']**2)
        df['Jerk'] = df['AccMag'].diff().abs().fillna(0)
        df['Fall_Detected'] = np.random.randint(0, 2, n)
        df['Fall_Severity'] = np.random.choice([0, 1, 2, 3], n)
        df['Fall_Direction'] = np.random.choice(['None', 'Forward', 'Backward'], n)
        df['Impact_Force'] = df['AccMag'] * 0.7 + np.random.normal(0, 0.3, n)
        df['Activity_State'] = np.random.choice(['Walking', 'Standing', 'Sitting', 'Falling'], n)
        return df

    df = generate_advanced_data()
    st.dataframe(df.head())

    # Encode and prepare inputs
    features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AccMag', 'GyroMag', 'Jerk', 'Pressure']
    X = df[features]
    le_dir = LabelEncoder()
    df['Dir_Enc'] = le_dir.fit_transform(df['Fall_Direction'])
    le_act = LabelEncoder()
    df['Act_Enc'] = le_act.fit_transform(df['Activity_State'])
    
    y_det = df['Fall_Detected']
    y_sev = df['Fall_Severity']
    y_dir = df['Dir_Enc']
    y_impact = df['Impact_Force']
    y_act = df['Act_Enc']

    X_train, X_test, y_det_train, y_det_test = train_test_split(X, y_det, test_size=0.3, random_state=42)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)

    detection_model = XGBClassifier().fit(X_train_scaled, y_det_train)
    st.subheader("📌 Detection Model Results")
    y_pred = detection_model.predict(X_test_scaled)
    st.text(classification_report(y_det_test, y_pred))

    fig, ax = plt.subplots()
    cm = confusion_matrix(y_det_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
