import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📊 Posture Monitoring Dashboard")
st.markdown("View real-time posture events and refresh as needed.")

# Load data
@st.cache_data(ttl=10)
def load_data():
    try:
        df = pd.read_csv("posture_logs.csv")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Minute'] = df['Timestamp'].dt.floor('T')
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data available.")
else:
    # Event frequency over time
    event_counts = df.groupby(['Minute', 'Event']).size().unstack(fill_value=0)

    st.subheader("📈 Event Frequency Over Time")
    st.line_chart(event_counts)

    # Event distribution
    st.subheader("📊 Overall Event Distribution")
    event_dist = df['Event'].value_counts()
    st.bar_chart(event_dist)

    # Raw table
    with st.expander("📋 View Raw Logs"):
        st.dataframe(df.tail(100).sort_values(by='Timestamp', ascending=False))

st.button("🔁 Refresh")
