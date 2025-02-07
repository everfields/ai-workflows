# LLM Data Analytics Workflow

A Python-based workflow that leverages large language models (LLMs) to automate data analytics processes. The code integrates with the OpenAI API to assist with data cleaning, exploratory data analysis (EDA), statistical analysis, and predictive forecasting.

## Features

- **Data Cleaning:** Uses LLM to generate cleaning instructions for handling missing values, outliers, and data type conversions
- **Exploratory Data Analysis:** Automatically selects relevant numerical columns and generates visualizations
- **Statistical Analysis:** Computes correlations and uses LLM to identify meaningful relationships
- **Predictive Analysis:** Performs time-series forecasting with Holt-Winters exponential smoothing

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-data-analytics.git
cd llm-data-analytics
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Place your dataset CSV file in the project directory
2. Update the `file_path` variable in `prompt_chaining.py` to match your CSV filename
3. Run the script:
```bash
python prompt_chaining.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.