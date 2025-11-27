import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Fix plotly theme for streamlit
pio.templates.default = 'plotly'

# Page configuration
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

def load_data(uploaded_file):
    """Load data with proper header handling - skip first row, use second row as header"""
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "csv":
        # Skip first row (category headers), use second row as column names
        df = pd.read_csv(uploaded_file, skiprows=[0])
    else:
        # For Excel files, skip first row
        df = pd.read_excel(uploaded_file, skiprows=[0])
    
    return df

def clean_dataframe(df):
    """Clean the dataframe and handle data types"""
    # Remove any completely empty rows
    df = df.dropna(how='all').reset_index(drop=True)
    
    # Convert currency and numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove currency symbols and commas, then convert to numeric
            cleaned_series = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(cleaned_series, errors='ignore')
    
    # Convert LED to datetime if it exists
    if 'LED' in df.columns:
        df['LED'] = pd.to_datetime(df['LED'], errors='coerce')
    
    return df

def create_visualizations(df):
    """Create automatic visualizations based on the data"""
    st.header("📈 Visual Analysis")
    
    # Add utilization column
    if 'Attendees' in df.columns and 'Capacity' in df.columns:
        df['Utilization %'] = (df['Attendees'] / df['Capacity'] * 100).round(2)
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    # Chart 1: Top 10 Buildings by Total Cost
    with col1:
        if 'Building Name' in df.columns and 'Rent+OpEx' in df.columns:
            top_buildings = df.nlargest(10, 'Rent+OpEx')[['Building Name', 'Rent+OpEx']].copy()
            fig1 = px.bar(
                top_buildings,
                x='Building Name',
                y='Rent+OpEx',
                title='Top 10 Buildings by Total Cost (Rent + OpEx)',
                labels={'Rent+OpEx': 'Total Cost ($)', 'Building Name': 'Building'},
                color='Rent+OpEx',
                color_continuous_scale='Reds'
            )
            fig1.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True, key="chart1")
    
    # Chart 2: Utilization Rate
    with col2:
        if 'Building Name' in df.columns and 'Utilization %' in df.columns:
            util_data = df.nlargest(10, 'Utilization %')[['Building Name', 'Utilization %']].copy()
            fig2 = px.bar(
                util_data,
                x='Building Name',
                y='Utilization %',
                title='Top 10 Buildings by Utilization Rate',
                labels={'Utilization %': 'Utilization (%)', 'Building Name': 'Building'},
                color='Utilization %',
                color_continuous_scale='Greens'
            )
            fig2.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                height=400
            )
            fig2.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Full Capacity")
            st.plotly_chart(fig2, use_container_width=True, key="chart2")
    
    # Chart 3 & 4
    col3, col4 = st.columns(2)
    
    # Chart 3: Rent vs OpEx
    with col3:
        if 'Rent' in df.columns and 'OpEx' in df.columns:
            totals = pd.DataFrame({
                'Category': ['Total Rent', 'Total OpEx'],
                'Amount': [df['Rent'].sum(), df['OpEx'].sum()]
            })
            fig3 = px.pie(
                totals,
                values='Amount',
                names='Category',
                title='Total Rent vs Operating Expenses',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                hole=0.3
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True, key="chart3")
    
    # Chart 4: Building Size vs Cost
    with col4:
        if 'SqFt' in df.columns and 'Building Name' in df.columns:
            fig4 = px.scatter(
                df,
                x='SqFt',
                y='Rent+OpEx' if 'Rent+OpEx' in df.columns else 'Rent',
                size='Capacity' if 'Capacity' in df.columns else None,
                hover_data=['Building Name'],
                title='Building Size vs Total Cost',
                labels={'SqFt': 'Square Feet', 'Rent+OpEx': 'Total Cost ($)'},
                color='Utilization %' if 'Utilization %' in df.columns else None,
                color_continuous_scale='viridis'
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True, key="chart4")
    
    # Chart 5: Cost per Square Foot
    st.subheader("💰 Cost Efficiency Analysis")
    col5, col6 = st.columns(2)
    
    with col5:
        if 'SqFt' in df.columns and 'Rent+OpEx' in df.columns:
            df['Cost per SqFt'] = (df['Rent+OpEx'] / df['SqFt']).round(2)
            cost_data = df.nlargest(10, 'Cost per SqFt')[['Building Name', 'Cost per SqFt']].copy()
            fig5 = px.bar(
                cost_data,
                x='Building Name',
                y='Cost per SqFt',
                title='Top 10 Most Expensive Buildings (Cost per SqFt)',
                labels={'Cost per SqFt': 'Cost per Sq Ft ($)', 'Building Name': 'Building'},
                color='Cost per SqFt',
                color_continuous_scale='Oranges'
            )
            fig5.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True, key="chart5")
    
    with col6:
        if 'Attendees' in df.columns and 'Capacity' in df.columns:
            # Buildings over/under capacity
            df['Capacity Status'] = df['Utilization %'].apply(
                lambda x: 'Over 100%' if x > 100 else ('80-100%' if x >= 80 else 'Under 80%')
            )
            status_counts = df['Capacity Status'].value_counts()
            fig6 = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Building Capacity Distribution',
                color_discrete_sequence=['#FF6B6B', '#FFD93D', '#6BCB77']
            )
            st.plotly_chart(fig6, use_container_width=True, key="chart6")

def generate_ai_insights(agent):
    """Generate AI insights about the data"""
    insights_prompt = """
    Analyze this building/office data and provide a comprehensive report:
    
    1. **Overview**: Total number of buildings, total annual costs, total capacity vs actual attendance
    
    2. **Cost Analysis**: 
       - Most and least expensive buildings
       - Average cost per square foot
       - Rent vs OpEx ratio
    
    3. **Utilization Analysis**:
       - Buildings that are over-capacity (>100% utilization)
       - Buildings that are under-utilized (<50% utilization)
       - Average utilization rate across all buildings
    
    4. **Efficiency Metrics**:
       - Cost per attendee
       - Space efficiency (SqFt per attendee)
    
    5. **Key Recommendations**: 
       - 3-5 specific, actionable recommendations based on the data
       - Identify potential cost savings opportunities
       - Suggest buildings that could be consolidated or need expansion
    
    Format your response with clear markdown headers (##) and bullet points. Include specific numbers, percentages, and building names.
    """
    
    try:
        response = agent.invoke(insights_prompt)
        return response["output"] if isinstance(response, dict) else str(response)
    except Exception as e:
        return f"⚠️ Error generating insights: {str(e)}"

def main():
    st.title("📊 Building Portfolio Analysis Dashboard")
    st.markdown("**AI-Powered Building Data Analytics & Visualization**")

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your building data file",
            type=["csv", "xlsx", "xls"],
            help="File should have headers in second row"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")

    if uploaded_file is not None:
        try:
            # Load and clean data
            with st.spinner("📂 Loading data..."):
                df = load_data(uploaded_file)
                df = clean_dataframe(df)
            
            st.sidebar.info(f"📊 **{len(df)}** buildings loaded")
            
            # Data Preview
            with st.expander("🔍 View Raw Data", expanded=False):
                st.dataframe(df, use_container_width=True, height=300)
            
            # Summary Statistics at the top
            st.header("📋 Key Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Buildings", len(df))
            
            with col2:
                if 'Rent+OpEx' in df.columns:
                    total_cost = df['Rent+OpEx'].sum()
                    st.metric("Total Annual Cost", f"${total_cost:,.0f}")
            
            with col3:
                if 'Attendees' in df.columns:
                    total_attendees = df['Attendees'].sum()
                    st.metric("Total Attendees", f"{total_attendees:,.0f}")
            
            with col4:
                if 'Capacity' in df.columns:
                    total_capacity = df['Capacity'].sum()
                    st.metric("Total Capacity", f"{total_capacity:,.0f}")
            
            with col5:
                if 'Attendees' in df.columns and 'Capacity' in df.columns:
                    avg_util = (df['Attendees'].sum() / df['Capacity'].sum() * 100)
                    st.metric("Avg Utilization", f"{avg_util:.1f}%")
            
            st.divider()
            
            # Create visualizations
            create_visualizations(df)
            
            st.divider()
            
            # AI-Generated Insights
            st.header("🤖 AI-Powered Insights & Recommendations")
            
            with st.spinner("🔍 AI is analyzing your building portfolio..."):
                # Initialize Agent
                llm = ChatOpenAI(temperature=0, model="gpt-4o")
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=False,
                    agent_type="openai-functions",
                    allow_dangerous_code=True
                )
                
                # Generate and display insights
                insights = generate_ai_insights(agent)
                st.markdown(insights)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Make sure your file has:\n- First row: Category headers (can be skipped)\n- Second row: Column names\n- Data rows: Building information")
            
    else:
        # Welcome screen
        st.info("👆 Upload your building data file to begin comprehensive analysis")
        
        st.markdown("""
        ### 🎯 Features:
        - **📊 6 Interactive Visualizations**: Cost analysis, utilization tracking, efficiency metrics
        - **🤖 AI-Powered Insights**: Intelligent analysis with actionable recommendations
        - **📈 Real-time Metrics**: Track costs, occupancy, and performance
        - **💡 Smart Recommendations**: Identify optimization opportunities automatically
        
        ### 📄 Expected Data Format:
        Your file should have these columns (case-insensitive):
        - **Building Name**: Name/ID of the building
        - **SqFt**: Square footage
        - **LED**: Lease end date
        - **Rent**: Monthly/annual rent
        - **OpEx**: Operating expenses
        - **Rent+OpEx**: Total cost
        - **Assigned**: Number of assigned seats
        - **Attendees**: Actual attendance
        - **Capacity**: Maximum capacity
        
        *Note: The first row (category headers) will be automatically skipped.*
        """)

if __name__ == "__main__":
    main()
