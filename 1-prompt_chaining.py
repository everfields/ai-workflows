import os
import openai
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

###############################################################################
# 1. SETUP
###############################################################################
# Load environment variables (for OPENAI_API_KEY)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your CSV file
file_path = "dataset_2.csv"  # Replace with your actual dataset file
data = pd.read_csv(file_path)

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Function to save a graph
def save_graph(fig, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)
    return filepath

###############################################################################
# 2. DATA CLEANING
###############################################################################
class CleaningInstructions(BaseModel):
    steps: list[str] = Field(description="List of recommended data cleaning steps.")

def call_llm_for_cleaning_instructions(df: pd.DataFrame) -> CleaningInstructions:
    """
    Calls the LLM to get data cleaning instructions in a structured format.
    """
    prompt = (
        "You are a data analyst. We have the following columns in the dataset:\n"
        f"{list(df.columns)}\n\n"
        "Please recommend appropriate data cleaning steps. Focus on:\n"
        "- Handling missing values\n"
        "- Removing outliers\n"
        "- Converting data types if needed\n\n"
        "Return your answer as a JSON with the key 'steps' containing a list of steps.\n"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful data cleaning assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_content = response["choices"][0]["message"]["content"]
    
    # Attempt extracting JSON
    try:
        json_start = raw_content.find("{")
        json_end = raw_content.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_block = raw_content[json_start:json_end]
            return CleaningInstructions.parse_raw(json_block)
        else:
            raise ValueError("No JSON block found.")
    except Exception:
        # Fallback to a default if something goes wrong
        return CleaningInstructions(steps=["No specific steps identified."])

# Call the LLM to get cleaning instructions
cleaning_instructions = call_llm_for_cleaning_instructions(data)
print("Recommended Cleaning Steps:")
for step in cleaning_instructions.steps:
    print("-", step)

# Simple placeholder data cleaning function
def simple_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Drop duplicates
    df = df.drop_duplicates()
    # 2) Drop rows where all columns are NaN
    df = df.dropna(how="all")
    # 3) Fill numeric NaN with column mean (example)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

data = simple_data_cleaning(data)

###############################################################################
# 3. EDA - SINGLE NUMERICAL COLUMN BASED ON USER QUERY
###############################################################################
class SingleNumericalColumn(BaseModel):
    relevant_column: str = Field(
        description="The single best numerical column to focus on for the given user query."
    )

def call_llm_for_relevant_column(df: pd.DataFrame, query: str) -> SingleNumericalColumn:
    """
    Calls the LLM to select the single most relevant numerical column
    based on the user query.
    """
    prompt = (
        f"User query: \"{query}\"\n\n"
        f"Available columns in the dataset: {list(df.columns)}\n"
        "Which single numeric column is most relevant to address the user query? "
        "Return your answer as JSON with the key 'relevant_column'."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for selecting columns."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )

    raw_content = response["choices"][0]["message"]["content"]
    
    # Attempt extracting JSON
    try:
        json_start = raw_content.find("{")
        json_end = raw_content.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_block = raw_content[json_start:json_end]
            return SingleNumericalColumn.parse_raw(json_block)
        else:
            raise ValueError("No JSON block found in the response.")
    except Exception as e:
        print("Error parsing LLM response:", e)
        # Fallback to a default or empty selection
        return SingleNumericalColumn(relevant_column="")

# Example user query
user_query = "I want to improve the profitability of the company."

# Call the LLM to get the single relevant numeric column
selected_column_obj = call_llm_for_relevant_column(data, user_query)
selected_column = selected_column_obj.relevant_column
print(f"\nLLM-selected numeric column for EDA: {selected_column}")

# Generate histogram & box plot for the selected column
hist_path, box_path = None, None
if selected_column and selected_column in data.columns:
    if pd.api.types.is_numeric_dtype(data[selected_column]):
        column_data = data[selected_column].dropna()

        # Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(column_data, bins=30, color="blue", alpha=0.7)
        plt.title(f"Histogram of {selected_column}")
        plt.xlabel(selected_column)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        # Box Plot
        plt.figure(figsize=(4, 6))
        plt.boxplot(column_data, vert=True, patch_artist=True, labels=[selected_column])
        plt.title(f"Box Plot of {selected_column}")
        plt.grid(True)
        plt.show()
    else:
        print(f"Column '{selected_column}' is not numeric. No plots generated.")
else:
    print("No valid numeric column selected by the LLM. No plots generated.")

###############################################################################
# 4. BASIC STATISTICAL ANALYSIS
###############################################################################
###############################################################################
# Pydantic models for structured output
###############################################################################
class CorrelationPair(BaseModel):
    col1: str
    col2: str
    correlation: float

class CorrelationResponse(BaseModel):
    top_pairs: list[CorrelationPair] = Field(
        description="Top correlation pairs relevant to the user query."
    )

###############################################################################
# Function to get top correlations from the LLM
###############################################################################
def call_llm_for_top_correlations(df: pd.DataFrame, user_query: str) -> CorrelationResponse:
    """
    1) Computes a correlation matrix and flattens it to a list of (col1, col2, correlation).
    2) Sorts by absolute correlation.
    3) Passes the list plus the user_query to the LLM for final selection of the top 10 pairs.
    """

    # 1. Compute correlation matrix (numeric_only=True for Pandas 1.5+)
    corr_matrix = df.corr(numeric_only=True)
    if corr_matrix.empty:
        # No numeric columns, return an empty result
        return CorrelationResponse(top_pairs=[])

    # 2. Flatten into a list of dictionaries
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append({
                "col1": cols[i],
                "col2": cols[j],
                "correlation": corr_matrix.iloc[i, j]
            })

    # 3. Sort by absolute correlation (descending)
    pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    # Convert to JSON so the LLM can parse it easily
    flattened_json = json.dumps(pairs_sorted, indent=2)

    # 4. Prompt for the LLM
    prompt = f"""
        User query: "{user_query}"

        We have these correlation pairs (sorted by absolute correlation from highest to lowest):
        {flattened_json}

        Which top 10 correlation pairs are most relevant to the user query?
        Return them exactly in JSON, with key "top_pairs", for example:

        {{
        "top_pairs": [
            {{
            "col1": "some_column",
            "col2": "another_column",
            "correlation": 0.93
            }},
            ...
        ]
        }}
        """
    # 5. LLM Call
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # or "gpt-4o-mini
        messages=[
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )

    # 6. Parse the LLM response
    raw_content = response["choices"][0]["message"]["content"]
    try:
        json_start = raw_content.find("{")
        json_end = raw_content.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_block = raw_content[json_start:json_end]
            return CorrelationResponse.parse_raw(json_block)
        else:
            # If it fails, return empty
            return CorrelationResponse(top_pairs=[])
    except Exception:
        return CorrelationResponse(top_pairs=[])

class StatisticalInstructions(BaseModel):
    analysis_type: str = Field(
        description="Type of statistical analysis to perform (e.g., correlation, ttest, etc.)."
    )

def call_llm_for_statistical_instructions(df: pd.DataFrame) -> StatisticalInstructions:
    """
    Calls the LLM to decide what statistical test or approach to use.
    """
    prompt = (
        "You are a data analyst. We have the following columns:\n"
        f"{list(df.columns)}\n\n"
        "Suggest a simple statistical analysis method to apply here (e.g., correlation, t-test, ANOVA) "
        "and provide it in a JSON object as {\"analysis_type\": \"...\"}.\n"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful statistical analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_content = response["choices"][0]["message"]["content"]

    # Attempt extracting JSON
    try:
        json_start = raw_content.find("{")
        json_end = raw_content.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_block = raw_content[json_start:json_end]
            return StatisticalInstructions.parse_raw(json_block)
        else:
            raise ValueError("No JSON block found.")
    except Exception:
        return StatisticalInstructions(analysis_type="correlation")

stat_instructions = call_llm_for_statistical_instructions(data)
analysis_type = stat_instructions.analysis_type.lower()
print("\nStatistical Analysis Requested:", analysis_type)

import seaborn as sns

if analysis_type == "correlation":
    # 3. Call the LLM to pick the top 10 relevant correlations
    user_query = "I want to improve the profitability of the company."
    top_corrs = call_llm_for_top_correlations(data, user_query)

    # 4. Print the top 10 correlation pairs selected by the LLM
    print("\nTOP 10 CORRELATION PAIRS SELECTED BY THE LLM:")
    columns_in_top_corrs = set()
    for pair in top_corrs.top_pairs:
        print(f"{pair.col1} <-> {pair.col2} : {pair.correlation}")
        # Collect all unique columns from these pairs
        columns_in_top_corrs.add(pair.col1)
        columns_in_top_corrs.add(pair.col2)

    # 5. Subset the original data to just those columns
    # Step 2: Save Heatmap
    heatmap_path = None
    if len(top_corrs.top_pairs) > 0:
        columns_in_top_corrs = set(pair.col1 for pair in top_corrs.top_pairs) | set(pair.col2 for pair in top_corrs.top_pairs)
        subset_cols = list(columns_in_top_corrs)

    if len(subset_cols) > 1:
        # 6. Create a correlation matrix for those columns only
        sub_corr_matrix = data[subset_cols].corr(numeric_only=True)

        # 7. Plot the heatmap for just the top-10 correlations subset
        plt.figure(figsize=(8, 6))
        sns.heatmap(sub_corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Heatmap of Columns in Top 10 Correlations")
        plt.show()
    else:
        print("\nNot enough columns to create a subset correlation heatmap.")

elif analysis_type == "t-test":
    # This is just a placeholder. In real scenarios, define group columns & numeric columns.
    print("Performing a placeholder t-test. (Implement your logic for grouping here.)")
    # ...
else:
    print(f"No implementation for '{analysis_type}' yet, skipping.")

###############################################################################
# 5. PREDICTIVE ANALYSIS
###############################################################################

###############################################################################
# 5.1. LLM: CHOOSE TARGET & CATEGORICAL COLUMNS
###############################################################################
class PredictiveInstructions(BaseModel):
    target_column: str = Field(
        description="Name of the numeric target column for prediction."
    )
    categorical_column: str = Field(
        description="Name of the categorical segment column (e.g., product category)."
    )

def call_llm_for_predictive_instructions(df: pd.DataFrame, query: str) -> PredictiveInstructions:
    """
    Calls the LLM to decide which numeric column should be used as the 'target' for predictions
    and which categorical column should be used as the 'segment' or grouping column.
    """
    prompt = (
        f"User query: \"{query}\"\n\n"
        f"Columns in the dataset: {list(df.columns)}\n\n"
        "1) Identify the single numeric column that best addresses this user query as 'target_column'.\n"
        "2) Identify the single categorical column that best segments or groups this target, "
        "   as 'categorical_column'. Avoid selecting an ID column; prefer a column with meaningful names.\n"
        "Return them in JSON like:\n\n"
        "{\n"
        "  \"target_column\": \"...\",\n"
        "  \"categorical_column\": \"...\"\n"
        "}\n"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful predictive modeling assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_content = response["choices"][0]["message"]["content"]

    # Attempt to extract JSON from the response
    try:
        json_start = raw_content.find("{")
        json_end = raw_content.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_block = raw_content[json_start:json_end]
            return PredictiveInstructions.parse_raw(json_block)
        else:
            raise ValueError("No JSON block found.")
    except Exception:
        # Fallback if something goes wrong
        return PredictiveInstructions(target_column="", categorical_column="")

predictive_instructions = call_llm_for_predictive_instructions(data, user_query)
target_column = predictive_instructions.target_column
cat_column = predictive_instructions.categorical_column

print("\nLLM-Selected Target Column:", target_column)
print("LLM-Selected Categorical Column:", cat_column)

###############################################################################
# 5.2. CHECK VALIDITY & PREP DATA
###############################################################################
if not target_column or target_column not in data.columns:
    print("No valid target column recognized. Exiting.")
    exit()

if not cat_column or cat_column not in data.columns:
    print("No valid categorical column recognized. Exiting.")
    exit()

if not pd.api.types.is_numeric_dtype(data[target_column]):
    print(f"Target column '{target_column}' is not numeric. Exiting.")
    exit()

# Identify a date/time column for time series
date_cols = [col for col in data.columns if "date" in col.lower() or "time" in col.lower()]
if not date_cols:
    print("No date/time column found in dataset. Exiting.")
    exit()

# We'll pick the first date/time column
date_col = date_cols[0]
data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
data = data.dropna(subset=[date_col])  # Remove rows with invalid dates
data = data.sort_values(by=date_col)

###############################################################################
# 5.3. GROUP & AGGREGATE FOR TIME-SERIES FORECASTS PER CATEGORY
###############################################################################
# We'll assume monthly data
data["year_month"] = data[date_col].dt.to_period("M")

# Select top 10 categories (by total sum of target_column)
top_categories = (
    data.groupby(cat_column)[target_column]
    .sum()
    .nlargest(10)
    .index
)

# Filter data to only those top categories
filtered_data = data[data[cat_column].isin(top_categories)]

# Aggregate numeric target by category & month
grouped_data = (
    filtered_data
    .groupby([cat_column, "year_month"])[target_column]
    .sum()
    .reset_index()
)
grouped_data["year_month"] = grouped_data["year_month"].dt.to_timestamp()

###############################################################################
# 5.4. BUILD FORECASTS FOR EACH CATEGORY
###############################################################################
forecast_horizon = 12  # how many future periods (months) to predict
predictions = {}

prediction_graphs = []
plt.figure(figsize=(12, 8))
for category in top_categories:
    # Extract time series for this category
    category_df = grouped_data[grouped_data[cat_column] == category].copy()
    category_df.set_index("year_month", inplace=True)
    category_series = category_df[target_column]

    # Skip if too few data points
    if len(category_series) < 2:
        print(f"Skipping '{category}' - not enough data points.")
        continue

    # Fit Holt-Winters for each category
    model = ExponentialSmoothing(
        category_series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated"
    )
    fitted_model = model.fit()

    forecast = fitted_model.forecast(forecast_horizon)

    # Plot historical and forecast
    plt.plot(category_series.index, category_series, label=f"{category} (Historical)", marker='o')
    plt.plot(forecast.index, forecast, "--", label=f"{category} (Forecast)", marker='x')
    plt.title(f"Predictions for {category}")
    plt.xlabel("Date")
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)

###############################################################################
# 5.6. USE LLM TO SUMMARIZE PREDICTIONS
###############################################################################
predictions_text = ""
for category, forecast_series in predictions.items():
    predictions_text += f"\nCategory: {category}\n{forecast_series.to_string()}\n"

if predictions_text:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a business analyst. Provide insights and recommendations."},
            {
                "role": "user",
                "content": (
                    f"User query: {user_query}\n\n"
                    f"Target column: {target_column}\n"
                    f"Category column: {cat_column}\n\n"
                    "Here are the monthly predictions for each of the top 10 categories:\n"
                    f"{predictions_text}\n\n"
                    "Please summarize key insights, trends, and recommend next actions."
                )
            }
        ],
        temperature=0.6
    )

    print("\nLLM Summary and Recommendations:\n")
    print(response["choices"][0]["message"]["content"])
else:
    print("No predictions to summarize.")

