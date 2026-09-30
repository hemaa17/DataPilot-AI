import streamlit as st
import pandas as pd
import os
import json
import re
import plotly.express as px

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. LOAD GEMINI API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# ============================================================
# 2. COLUMN / DATASET HELPERS
# ============================================================

def column_is_mentioned(column, question):
    """Check whether a complete column name appears in the question."""
    pattern = r"\b" + re.escape(column.lower()) + r"\b"
    return re.search(pattern, question.lower()) is not None


def inspect_columns(data):
    """Identify numeric, date/time, and category columns."""
    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    date_columns = []

    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            date_columns.append(column)
            continue

        column_name = column.lower()

        looks_like_date = (
            "date" in column_name
            or "time" in column_name
            or "year" in column_name
        )

        if looks_like_date:
            converted = pd.to_datetime(
                data[column],
                errors="coerce",
                format="mixed"
            )

            if len(data) > 0 and converted.notna().sum() >= len(data) * 0.7:
                date_columns.append(column)

    category_columns = [
        column
        for column in data.columns
        if column not in numeric_columns
        and column not in date_columns
    ]

    return numeric_columns, date_columns, category_columns


# ============================================================
# 3. LOCAL FALLBACK
# ============================================================

def local_fallback(question, data):
    """Understand common questions locally when Gemini is unavailable."""

    question_lower = question.lower()

    (
        numeric_columns,
        date_columns,
        category_columns
    ) = inspect_columns(data)

    mentioned_numeric = next(
        (
            column
            for column in numeric_columns
            if column_is_mentioned(column, question_lower)
        ),
        None
    )

    mentioned_category = next(
        (
            column
            for column in category_columns
            if column_is_mentioned(column, question_lower)
        ),
        None
    )

    mentioned_date = next(
        (
            column
            for column in date_columns
            if column_is_mentioned(column, question_lower)
        ),
        None
    )

    unknown = {
        "intent": "UNKNOWN",
        "category_column": None,
        "value_column": None
    }

    # COUNT BY CATEGORY
    if (
        "how many" in question_lower
        or "count" in question_lower
        or "number of" in question_lower
    ):
        if not mentioned_category:
            return unknown

        return {
            "intent": "COUNT_BY_CATEGORY",
            "category_column": mentioned_category,
            "value_column": None
        }

    # HIGHEST AVERAGE BY CATEGORY
    if (
        any(word in question_lower for word in
            ["highest", "most", "maximum", "largest"])
        and any(word in question_lower for word in
                ["average", "avg", "mean"])
    ):
        category_column = mentioned_category
        value_column = mentioned_numeric or (
            numeric_columns[0] if numeric_columns else None
        )

        if not category_column or not value_column:
            return unknown

        return {
            "intent": "HIGHEST_AVERAGE_BY_CATEGORY",
            "category_column": category_column,
            "value_column": value_column
        }

    # HIGHEST VALUE
    if any(word in question_lower for word in
           ["highest", "most", "maximum", "max", "largest", "best", "strongest", "top", "top-performing"]):
        value_column = mentioned_numeric or (
            numeric_columns[0] if numeric_columns else None
        )
        category_column = mentioned_category or (
            category_columns[0] if category_columns else None
        )

        if not value_column or not category_column:
            return unknown

        return {
            "intent": "HIGHEST_VALUE",
            "category_column": category_column,
            "value_column": value_column
        }

    # LOWEST VALUE
    if any(word in question_lower for word in
           ["lowest", "least", "minimum", "min", "smallest", "weakest"]):
        value_column = mentioned_numeric or (
            numeric_columns[0] if numeric_columns else None
        )
        category_column = mentioned_category or (
            category_columns[0] if category_columns else None
        )

        if not value_column or not category_column:
            return unknown

        return {
            "intent": "LOWEST_VALUE",
            "category_column": category_column,
            "value_column": value_column
        }

    # TOTAL VALUE
    if any(word in question_lower for word in
           ["total", "sum", "overall"]):
        value_column = mentioned_numeric or (
            numeric_columns[0] if numeric_columns else None
        )

        if not value_column:
            return unknown

        return {
            "intent": "TOTAL_VALUE",
            "category_column": None,
            "value_column": value_column
        }

    # TREND OVER TIME
    if (
        "trend" in question_lower
        or "over time" in question_lower
        or "monthly" in question_lower
        or "month" in question_lower
        or "yearly" in question_lower
        or "year" in question_lower
        or "daily" in question_lower
        or "day" in question_lower
    ):
        date_column = mentioned_date or (
            date_columns[0] if date_columns else None
        )
        value_column = mentioned_numeric or (
            numeric_columns[0] if numeric_columns else None
        )

        if not date_column or not value_column:
            return unknown

        return {
            "intent": "TREND_OVER_TIME",
            "category_column": date_column,
            "value_column": value_column
        }

    # AVERAGE
    if any(word in question_lower for word in
           ["average", "avg", "mean"]):
        value_column = mentioned_numeric or (
            numeric_columns[0] if numeric_columns else None
        )

        category_column = mentioned_category or (
            category_columns[0] if category_columns else None
        )

        if not value_column:
            return unknown

        if (
            "by" in question_lower
            or "across" in question_lower
            or "department" in question_lower
            or "category" in question_lower
        ):
            if not category_column:
                return unknown

            return {
                "intent": "AVERAGE_BY_CATEGORY",
                "category_column": category_column,
                "value_column": value_column
            }

        return {
            "intent": "AVERAGE_VALUE",
            "category_column": None,
            "value_column": value_column
        }

    # GROUP BY CATEGORY
    if any(word in question_lower for word in
           ["compare", "comparison", "across", "by"]):
        category_column = mentioned_category or (
            category_columns[0] if category_columns else None
        )
        value_column = mentioned_numeric or (
            numeric_columns[0] if numeric_columns else None
        )

        if not category_column or not value_column:
            return unknown

        return {
            "intent": "GROUP_BY_CATEGORY",
            "category_column": category_column,
            "value_column": value_column
        }

    return unknown


# ============================================================
# 4. GEMINI UNDERSTANDS THE QUESTION
# ============================================================

def understand_question(question, data):
    column_info = [
        f"{column} ({data[column].dtype})"
        for column in data.columns
    ]

    columns_description = ", ".join(column_info)

    prompt = f"""
You are a data analysis assistant.

Dataset columns:
{columns_description}

User question:
{question}

Return ONLY valid JSON.

Use exactly this structure:
{{
    "intent": "HIGHEST_VALUE",
    "category_column": "Product",
    "value_column": "Sales"
}}

Allowed intents:
HIGHEST_VALUE
LOWEST_VALUE
TOTAL_VALUE
AVERAGE_VALUE
AVERAGE_BY_CATEGORY
GROUP_BY_CATEGORY
COUNT_BY_CATEGORY
TREND_OVER_TIME
HIGHEST_AVERAGE_BY_CATEGORY
UNKNOWN

Rules:
- HIGHEST_VALUE = find the category with the highest total numeric value.
- LOWEST_VALUE = find the category with the lowest total numeric value.
- TOTAL_VALUE = calculate the total of a numeric column.
- AVERAGE_VALUE = calculate the average of a numeric column.
- AVERAGE_BY_CATEGORY = calculate the average numeric value for each category.
- GROUP_BY_CATEGORY = compare a numeric value across categories.
- COUNT_BY_CATEGORY = count records in each category.
- TREND_OVER_TIME = analyze a numeric value over a date/time column.
- HIGHEST_AVERAGE_BY_CATEGORY = find the category with the highest average numeric value.
- UNKNOWN = if the question cannot be understood.

Important:
1. Use only actual column names from the dataset.
2. Never invent a column name.
3. category_column must be the column used for grouping.
4. value_column must be the numeric column being analyzed.
5. For TOTAL_VALUE and AVERAGE_VALUE, category_column must be null.
6. For COUNT_BY_CATEGORY, value_column must be null.
7. For TREND_OVER_TIME, category_column must contain the date/time column.
8. Return ONLY JSON.
9. Do not add explanations.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        result = response.text.strip()
        result = result.replace("```json", "").replace("```", "").strip()
        result = json.loads(result)

        valid_intents = [
            "HIGHEST_VALUE",
            "LOWEST_VALUE",
            "TOTAL_VALUE",
            "AVERAGE_VALUE",
            "AVERAGE_BY_CATEGORY",
            "GROUP_BY_CATEGORY",
            "COUNT_BY_CATEGORY",
            "TREND_OVER_TIME",
            "HIGHEST_AVERAGE_BY_CATEGORY",
            "UNKNOWN"
        ]

        if result.get("intent") not in valid_intents:
            raise ValueError("Invalid intent from Gemini.")

        category_column = result.get("category_column")
        value_column = result.get("value_column")

        if (
            category_column is not None
            and category_column not in data.columns
        ):
            raise ValueError("Invalid category column from Gemini.")

        if (
            value_column is not None
            and value_column not in data.columns
        ):
            raise ValueError("Invalid value column from Gemini.")

        return result

    except Exception as e:
        print("Gemini unavailable. Using local fallback.")
        print(e)

        return local_fallback(question, data)


# ============================================================
# 5. NUMBER FORMATTING
# ============================================================

def format_number(value):
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    return f"{value:,}"


# ============================================================
# 6. REUSABLE ANALYSIS FUNCTIONS / TOOLS
# ============================================================

def find_highest(data, category_column, value_column):
    """Find the category with the highest total value."""
    grouped_data = data.groupby(category_column)[value_column].sum()

    highest_category = grouped_data.idxmax()
    highest_value = grouped_data.max()

    return highest_category, highest_value


def find_lowest(data, category_column, value_column):
    """Find the category with the lowest total value."""
    grouped_data = data.groupby(category_column)[value_column].sum()

    lowest_category = grouped_data.idxmin()
    lowest_value = grouped_data.min()

    return lowest_category, lowest_value


def calculate_total(data, value_column):
    """Calculate the total of a numeric column."""
    return data[value_column].sum()


def calculate_average(data, value_column):
    """Calculate the overall average of a numeric column."""
    return data[value_column].mean()


def calculate_average_by_category(data, category_column, value_column):
    """Calculate the average value for each category."""
    return data.groupby(category_column)[value_column].mean().sort_values(
        ascending=False
    )


def count_by_category(data, category_column):
    """Count records in each category."""
    return data.groupby(category_column).size().sort_values(
        ascending=False
    )


def group_by_category(data, category_column, value_column):
    """Calculate total value for each category."""
    return data.groupby(category_column)[value_column].sum().sort_values(
        ascending=False
    )


def find_highest_average_by_category(data, category_column, value_column):
    """Find the category with the highest average value."""
    grouped_data = data.groupby(category_column)[value_column].mean()

    highest_category = grouped_data.idxmax()
    highest_average = grouped_data.max()

    return highest_category, highest_average


def calculate_trend(data, date_column, value_column):
    """Calculate monthly totals from a date/time column."""
    temporary_data = data.copy()

    temporary_data[date_column] = pd.to_datetime(
        temporary_data[date_column],
        errors="coerce",
        format="mixed"
    )

    temporary_data = temporary_data.dropna(
        subset=[date_column]
    )

    monthly_data = temporary_data.groupby(
        temporary_data[date_column].dt.to_period("M")
    )[value_column].sum()

    monthly_data.index = monthly_data.index.astype(str)

    return monthly_data


# ============================================================
# 7. ANALYZE DATA USING THE REUSABLE FUNCTIONS
# ============================================================

def answer_question(result, data):
    intent = result["intent"]
    category_column = result["category_column"]
    value_column = result["value_column"]

    if intent == "HIGHEST_AVERAGE_BY_CATEGORY":

        highest_category, highest_average = (
            find_highest_average_by_category(
                data,
                category_column,
                value_column
            )
        )

        return (
            f"{highest_category} has the highest "
            f"average {value_column.lower()} with "
            f"{format_number(highest_average)}"
        )

    if intent == "HIGHEST_VALUE":

        highest_category, highest_value = find_highest(
            data,
            category_column,
            value_column
        )

        return (
            f"{highest_category} has the highest "
            f"{value_column.lower()} with "
            f"{format_number(highest_value)}"
        )

    elif intent == "LOWEST_VALUE":

        lowest_category, lowest_value = find_lowest(
            data,
            category_column,
            value_column
        )

        return (
            f"{lowest_category} has the lowest "
            f"{value_column.lower()} with "
            f"{format_number(lowest_value)}"
        )

    elif intent == "TOTAL_VALUE":

        total_value = calculate_total(
            data,
            value_column
        )

        return (
            f"The total {value_column.lower()} is "
            f"{format_number(total_value)}"
        )

    elif intent == "AVERAGE_BY_CATEGORY":

        return calculate_average_by_category(
            data,
            category_column,
            value_column
        )

    elif intent == "AVERAGE_VALUE":

        average_value = calculate_average(
            data,
            value_column
        )

        return (
            f"The average {value_column.lower()} is "
            f"{format_number(average_value)}"
        )

    elif intent == "COUNT_BY_CATEGORY":

        return count_by_category(
            data,
            category_column
        )

    elif intent == "GROUP_BY_CATEGORY":

        return group_by_category(
            data,
            category_column,
            value_column
        )

    elif intent == "TREND_OVER_TIME":

        return calculate_trend(
            data,
            category_column,
            value_column
        )

    else:

        return "I don't know how to answer that question yet."


# ============================================================
# 8. AI BUSINESS INSIGHTS
# ============================================================

def prepare_insight_context(result, answer, data):
    """Create a compact summary for the insight model."""
    intent = result["intent"]
    category_column = result["category_column"]
    value_column = result["value_column"]

    if isinstance(answer, pd.Series):
        top_items = answer.head(5)
        bottom_items = answer.tail(3)

        top_text = ", ".join(
            f"{index}: {format_number(value)}"
            for index, value in top_items.items()
        )
        bottom_text = ", ".join(
            f"{index}: {format_number(value)}"
            for index, value in bottom_items.items()
        )

        return (
            f"Analysis type: {intent}\n"
            f"Category: {category_column}\n"
            f"Metric: {value_column or 'Count'}\n"
            f"Top results: {top_text}\n"
            f"Lower results: {bottom_text}"
        )

    return (
        f"Analysis type: {intent}\n"
        f"Category: {category_column or 'None'}\n"
        f"Metric: {value_column or 'None'}\n"
        f"Result: {answer}"
    )


def local_business_insight(result, answer, data):
    """Provide a useful local insight when Gemini is unavailable."""
    intent = result["intent"]
    value_column = result["value_column"]

    if isinstance(answer, pd.Series):
        if len(answer) == 0:
            return "There is not enough data to generate an insight."

        highest_category = answer.index[0]
        highest_value = answer.iloc[0]
        lowest_category = answer.index[-1]
        lowest_value = answer.iloc[-1]

        if intent == "COUNT_BY_CATEGORY":
            return (
                f"{highest_category} has the most records with {format_number(highest_value)}, "
                f"while {lowest_category} has the fewest with {format_number(lowest_value)}."
            )

        return (
            f"{highest_category} has the highest {value_column.lower()} at "
            f"{format_number(highest_value)}, while {lowest_category} is the lowest "
            f"at {format_number(lowest_value)}."
        )

    if intent == "HIGHEST_VALUE":
        return f"{answer}. This identifies the strongest-performing category in the dataset."
    if intent == "LOWEST_VALUE":
        return f"{answer}. This identifies the category with the lowest total in the dataset."
    if intent == "TOTAL_VALUE":
        return f"{answer}. This represents the overall total across the uploaded dataset."
    if intent == "AVERAGE_VALUE":
        return f"{answer}. This gives the typical value across the records in the dataset."
    if intent == "HIGHEST_AVERAGE_BY_CATEGORY":
        return f"{answer}. This category has the highest average value among the groups."
    if intent == "TREND_OVER_TIME":
        if isinstance(answer, pd.Series) and len(answer) > 0:
            first_period = answer.index[0]
            last_period = answer.index[-1]
            first_value = answer.iloc[0]
            last_value = answer.iloc[-1]
            if last_value > first_value:
                direction = "increased"
            elif last_value < first_value:
                direction = "decreased"
            else:
                direction = "remained unchanged"
            return (
                f"The trend {direction} from {first_period} to {last_period}, "
                f"moving from {format_number(first_value)} to {format_number(last_value)}."
            )

    return "The analysis provides a useful summary of the uploaded data."


def generate_business_insight(question, result, answer, data):
    """Ask Gemini to explain the analytical result in simple business language."""
    context = prepare_insight_context(result, answer, data)

    prompt = f"""
You are a business data analyst.

The user asked:
{question}

The analysis produced:
{context}

Write ONE short business insight for a normal user.

Rules:
- Do not mention Python, Pandas, functions, code, intents, columns, algorithms, APIs, Gemini, or backend processing.
- Do not explain how the software works.
- Focus only on what the result means for the data.
- Use simple business language.
- Mention an important comparison or trend when possible.
- Do not invent facts that are not supported by the analysis.
- Keep it to 1 or 2 sentences.
- Return only the insight text.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )
        insight = response.text.strip()
        if insight:
            return insight
    except Exception as e:
        print("Gemini insight generation unavailable. Using local insight.")
        print(e)

    return local_business_insight(result, answer, data)


# ============================================================
# 9. STREAMLIT APP
# ============================================================

st.title("DataPilot AI")

st.write(
    "AI-Powered Natural Language Data Analysis Platform"
)


# ============================================================
# 10. UPLOAD CSV
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ============================================================
# 11. PROCESS DATA
# ============================================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")

    st.dataframe(
        data,
        width="stretch"
    )

    st.write("### Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric(
        "Missing Values",
        data.isnull().sum().sum()
    )

    st.write("### Column Information")

    column_info = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str),
        "Missing Values": data.isnull().sum().values
    })

    st.dataframe(
        column_info,
        width="stretch"
    )

    question = st.text_input(
        "Ask a question about your data:",
        placeholder="Example: Which product sold the most?"
    )

    st.caption(
        "Try: Which product sold the most? • "
        "How much did we sell in total? • "
        "Compare sales across regions • "
        "What is the average salary by department? • "
        "How many employees are in each department? • "
        "Show the sales trend by month"
    )

    if question:

        result = understand_question(
            question,
            data
        )

        answer = answer_question(
            result,
            data
        )

        if isinstance(answer, str):

            st.write("### ✨ Analysis")

            st.success(answer)

        else:

            st.write("### ✨ Analysis")

            st.dataframe(
                answer,
                width="stretch"
            )

            if result["intent"] in [
                "GROUP_BY_CATEGORY",
                "AVERAGE_BY_CATEGORY",
                "COUNT_BY_CATEGORY"
            ]:

                if result["intent"] == "COUNT_BY_CATEGORY":
                    y_label = "Count"
                else:
                    y_label = result["value_column"]

                fig = px.bar(
                    x=answer.index,
                    y=answer.values,
                    labels={
                        "x": result["category_column"],
                        "y": y_label
                    },
                    title=(
                        f"{y_label} by "
                        f"{result['category_column']}"
                    )
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

            elif result["intent"] == "TREND_OVER_TIME":

                fig = px.line(
                    x=answer.index,
                    y=answer.values,
                    markers=True,
                    labels={
                        "x": "Time",
                        "y": result["value_column"]
                    },
                    title=(
                        f"{result['value_column']} "
                        f"Trend Over Time"
                    )
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

        insight = generate_business_insight(
            question,
            result,
            answer,
            data
        )

        st.write("### 💡 AI Insight")
        st.info(insight)
