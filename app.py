import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import dice_ml
from dice_ml import Dice
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config for wide layout
st.set_page_config(layout="wide")

# Set Seaborn style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14})

# Load dataset
df = pd.read_csv("adult.csv")

# Drop rows with missing values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns
encoders = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Split data
X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(pd.DataFrame(X_train, columns=X.columns), y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred), 4)

# Setup DiCE
d = dice_ml.Data(dataframe=pd.concat([X_train, y_train], axis=1), 
                 continuous_features=X.select_dtypes(include=[np.number]).columns.tolist(), 
                 outcome_name='income')
m = dice_ml.Model(model=model, backend="sklearn")
dice_exp = Dice(d, m)

# Streamlit app
st.title("Counterfactual Explanation Web App")

# Tabs
tab1, tab2 = st.tabs(["Counterfactuals", "EDA & Visualizations"])

# Tab 1: Counterfactuals
with tab1:
    st.write("### 🎯 Model Accuracy:", accuracy)

    # Pick a test sample
    index = st.number_input("🔎 Select index from test set", min_value=0, max_value=len(X_test)-1, step=1)
    sample = X_test.iloc[[index]]

    # Decode sample for display
    decoded_sample = sample.copy()
    for col in categorical_cols:
        if col in decoded_sample.columns:
            le = encoders[col]
            decoded_sample[col] = le.inverse_transform(decoded_sample[col])

    # Show original prediction
    original_prediction = model.predict(sample)[0]
    st.write(f"### 🧠 Original prediction (0=<=50K, 1=>50K): {original_prediction}")
    st.write("### 📋 Selected row data:")
    st.dataframe(decoded_sample)

    # Generate Counterfactuals
    cf = dice_exp.generate_counterfactuals(
        sample, total_CFs=5, desired_class="opposite", features_to_vary="all"
    )

    # Display Counterfactuals
    cf_df = cf.cf_examples_list[0].final_cfs_df
    decoded_cf_df = cf_df.copy()
    for col in categorical_cols:
        if col in decoded_cf_df.columns:
            le = encoders[col]
            decoded_cf_df[col] = le.inverse_transform(decoded_cf_df[col].astype(int))

    st.write("### 🔄 Counterfactual Explanations")
    st.dataframe(decoded_cf_df)

    # Analyze feature changes
    delta_changes = []
    individual_feature_counts = Counter()
    for i, row in cf_df.iterrows():
        changes = []
        for col in sample.columns:
            if sample[col].values[0] != row[col]:
                changes.append(col)
                individual_feature_counts[col] += 1
        delta_changes.append(tuple(sorted(changes)))

    st.write("### 🔁 Grouped Feature Importance")
    for feature, count in individual_feature_counts.most_common():
        st.write(f"- **{feature}** — used in {count} counterfactual{'s' if count > 1 else ''}")

    best_change = individual_feature_counts.most_common(1)[0][0]
    st.write(f"\n✅ **Best minimal change to flip prediction:** `{best_change}`")


# Tab 2: EDA & Visualizations
with tab2:
    st.write("### 📊 Exploratory Data Analysis & Visualizations")

    # Load raw data for EDA
    raw_df = pd.read_csv("adult.csv")
    raw_df.replace('?', np.nan, inplace=True)
    raw_df.dropna(inplace=True)

    # 1. Correlation Heatmap
    st.write("#### Correlation Heatmap of Features")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) 

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, 'label': 'Correlation Coefficient'},
        ax=ax1
    )
    ax1.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig1)

    # 2 & 3: Income Distribution & Hours per Week by Income
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Income Distribution")
        fig2, ax2 = plt.subplots(figsize=(3, 2.5))
        sns.countplot(x="income", data=raw_df, ax=ax2, palette="Set2")
        ax2.set_xticklabels(["<=50K", ">50K"])
        ax2.set_title("Distribution of Income Categories")
        st.pyplot(fig2)

    with col2:
        st.write("#### Hours per Week by Income")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x="income", y="hours-per-week", data=raw_df, 
                    ax=ax3, palette="Pastel1")
        ax3.set_xticklabels(["<=50K", ">50K"])
        ax3.set_title("Hours per Week by Income")
        st.pyplot(fig3)

    # 4: Education vs Income (full width)
    st.write("#### Education Level vs Income")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.countplot(x="education", hue="income", data=raw_df, ax=ax4, palette="muted")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")
    ax4.set_title("Income by Education Level")
    ax4.legend(title="Income", labels=["<=50K", ">50K"])
    plt.tight_layout()
    st.pyplot(fig4)

    # 5. Feature Importance from Counterfactuals
    st.write("#### Feature Importance from Counterfactuals")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    features, counts = zip(*individual_feature_counts.most_common())
    sns.barplot(x=list(counts), y=list(features), ax=ax5, palette="viridis")
    ax5.set_title("Features Most Impacting Counterfactuals", pad=10)
    ax5.set_xlabel("Number of Counterfactuals")
    ax5.set_ylabel("Feature")
    st.pyplot(fig5)

    # Export to Excel
    st.write("### 📥 Download Data as Excel")
    excel_buffer = pd.ExcelWriter("eda_results.xlsx", engine="xlsxwriter")
    raw_df.to_excel(excel_buffer, sheet_name="Raw Data", index=False)
    decoded_cf_df.to_excel(excel_buffer, sheet_name="Counterfactuals", index=False)
    corr.to_excel(excel_buffer, sheet_name="Correlation Matrix")
    excel_buffer.close()
    with open("eda_results.xlsx", "rb") as f:
        st.download_button("Download EDA Results", f, file_name="eda_results.xlsx")



