# DataPilot AI

DataPilot AI is an AI-powered natural language data analysis and visualization platform.

Users can upload CSV datasets and ask questions in natural language. Gemini understands the question, while Python and Pandas perform the actual data analysis. Plotly is used for interactive visualizations.

## Features

- Upload CSV datasets
- Ask questions using natural language
- AI-powered question understanding
- Data analysis using Pandas
- Interactive visualizations using Plotly
- AI-powered business insights
- Local fallback when Gemini is unavailable
- Streamlit-based interactive interface

## Tech Stack

- Python
- Pandas
- Google Gemini API
- Streamlit
- Plotly
- Python-dotenv

## Example Questions

- Which product sold the most?
- Which department has the highest average salary?
- How many employees are in each department?
- What is the total sales?
- Show sales by region.
- Show the sales trend over time.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py