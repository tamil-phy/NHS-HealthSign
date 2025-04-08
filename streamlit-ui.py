import streamlit as st
import json
import pandas as pd
import altair as alt

# Load JSON files
@st.cache_data

def load_json(uploaded_file):
    if uploaded_file is not None:
        return json.load(uploaded_file)
    return {}

st.title("Underwriting Clinical Summary")

# Sidebar uploaders
st.sidebar.header("Upload Processed JSON Files")
code_file = st.sidebar.file_uploader("Upload code_matches.json", type="json")
vital_file = st.sidebar.file_uploader("Upload vital_matches.json", type="json")
timeline_file = st.sidebar.file_uploader("Upload timeline.json", type="json")

# Load data
code_data = load_json(code_file)
vital_data = load_json(vital_file)
timeline_data = load_json(timeline_file)

if not (code_data and vital_data and timeline_data):
    st.warning("Please upload all three required JSON files.")
    st.stop()

# Tabs
tabs = st.tabs(["🏥 Major Diseases", "💉 Vitals", "📊 Vitals Graph", "🕒 Timeline"])

# Major Diseases Tab
with tabs[0]:
    st.subheader("Matched Read Codes (Major Diseases)")
    if not code_data:
        st.info("No major disease data found.")
    else:
        categories = {}
        for entry in code_data:
            category = entry.get("matched_category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(entry)

        for category, entries in categories.items():
            st.markdown(f"### {category}")
            for entry in entries:
                with st.container():
                    st.markdown(f"**{entry['code_description']}**")
                    st.markdown(f"- Code: `{entry['matched_code']}`")
                    st.markdown(f"- Group: {entry['matched_group']}")
                    st.markdown(f"- Date: {entry['date']}")
                    st.markdown(f"- Context: {entry['line']}")

# Vitals Tab
with tabs[1]:
    st.subheader("Extracted Vitals")
    if "vitals" not in vital_data:
        st.info("No vital data found.")
    else:
        for vital in vital_data["vitals"]:
            st.markdown(f"### {vital['vital_name']}")
            for m in vital['measurements']:
                with st.container():
                    st.markdown(f"- **Date:** {m['date']}")
                    st.markdown(f"- **Value:** {m['value']} {m['unit']}")
                    st.markdown(f"- **Context:** {m['context']}")

# Vitals Graph Tab
with tabs[2]:
    st.subheader("Graphical View of Key Vitals")
    try:
        rows = []
        for vital_item in vital_data.get("vitals", []):
            vital_name = vital_item.get("vital_name", "")
            for measurement in vital_item.get("measurements", []):
                date_str = measurement.get("date", "")
                value_str = measurement.get("value", "")
                unit = measurement.get("unit", "")
                context = measurement.get("context", "")
                try:
                    numeric_value = float(value_str)
                except ValueError:
                    numeric_value = None
                rows.append({
                    "date": date_str,
                    "vital_name": vital_name,
                    "value": numeric_value,
                    "unit": unit,
                    "context": context
                })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date", "value"])
        df = df.sort_values(by="date")

        allowed_vitals = ["BMI", "HDL Cholesterol", "Systolic Blood Pressure", "Diastolic Blood Pressure"]
        df = df[df["vital_name"].isin(allowed_vitals)]

        selected_vitals = st.multiselect(
            "Select vital(s) to plot",
            options=allowed_vitals,
            default=allowed_vitals
        )

        plot_df = df[df["vital_name"].isin(selected_vitals)]

        if plot_df.empty:
            st.warning("No data available for the selected vitals.")
        else:
            chart = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("value:Q", title="Measurement Value"),
                    color="vital_name:N",
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("vital_name:N", title="Vital"),
                        alt.Tooltip("value:Q", title="Value"),
                        alt.Tooltip("unit:N", title="Unit"),
                        alt.Tooltip("context:N", title="Context")
                    ]
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating graph: {e}")

# Timeline Tab
with tabs[3]:
    st.subheader("Chronological Summary of Clinical Events")
    if "events" not in timeline_data:
        st.info("No timeline data found.")
    else:
        for event in timeline_data["events"]:
            with st.container():
                st.markdown(f"### {event['date']} - {event['title']}")
                st.markdown(f"- **Significance:** {event['significance']}/10")
                st.markdown(f"- {event['description']}")
