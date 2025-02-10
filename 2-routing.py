import openai
import pandas as pd
import os

# Step 1: Load the dataset
file_path = "dataset_2.csv"  # Replace with your dataset file
dataset = pd.read_csv(file_path, encoding="latin1")

# Authenticate with OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 2: Function to interact with ChatGPT API for data cleaning
def chatgpt_data_cleaning(prompt, model="gpt-4o-mini", max_tokens=500):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are a data cleaning assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0
        )
        return response.choices[0].message['content']
    except openai.error.OpenAIError as e:
        print(f"API Request Failed: {e}")
        return None

# Step 3: Automating Tasks
# 1. Filling Missing Values
def fill_missing_values(data):
    for column in data.columns:
        missing_percentage = data[column].isnull().mean() * 100
        if missing_percentage > 0:
            prompt = f"""
            Column '{column}' in a dataset has {missing_percentage:.2f}% missing values.
            Provide a method to fill these values based on the column type 
            (numerical, categorical, or datetime) and suggest replacement values.
            """
            suggestion = chatgpt_data_cleaning(prompt)
            print(f"ChatGPT Suggestion for '{column}': {suggestion}")

# 2. Renaming Columns
def rename_columns(data):
    original_columns = data.columns.tolist()
    prompt = f"""
    The dataset has the following columns: {original_columns}.
    Provide consistent and standardized column names suitable for analysis.
    """
    new_names = chatgpt_data_cleaning(prompt)
    print(f"Suggested Column Names: {new_names}")

# 3. Fixing Data Inconsistencies
def fix_inconsistencies(data, column_name):
    prompt = f"""
    The column '{column_name}' in a dataset has inconsistencies. For example:
    - Invalid negative values for profitability in 'Order Profit Per Order'.
    - Missing or unrealistic values for shipping days in 'Days for shipping (real)'.
    Provide steps or rules to correct these inconsistencies.
    """
    suggestion = chatgpt_data_cleaning(prompt)
    print(f"ChatGPT Suggestion for '{column_name}': {suggestion}")

# Step 4: Apply the automation
# Filling missing values
fill_missing_values(dataset)

# Renaming columns
rename_columns(dataset)

# Fixing inconsistencies
fix_inconsistencies(dataset, "Order Profit Per Order")
fix_inconsistencies(dataset, "Days for shipping (real)")

# Step 5: Save the updated dataset
output_file_path = "dataset_automated_cleaned.csv"
dataset.to_csv(output_file_path, index=False)

print(f"Automated data cleaning completed. Cleaned dataset saved to {output_file_path}.")
