import streamlit as st
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")   

os.environ["OPENAI_API_KEY"] = api_key

# Page configuration
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Data Analysis Agent")
    st.markdown("Upload your CSV or Excel file and ask questions about your data!")

    # Sidebar for configuration
    with st.sidebar:
        # File uploader
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"])

    # Main funct
    if uploaded_file is not None:
        try:
            # Load data
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.sidebar.success(f"File uploaded successfully! ({len(df)} rows)")
            
            # Show preview
            with st.expander("Data Preview"):
                st.dataframe(df.head(10))

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Initialize Agent
            llm = ChatOpenAI(temperature=0, model="gpt-4o")
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type="openai-functions",
                allow_dangerous_code=True # Required for pandas agent to execute code
            )

            # # Chat input
            # if prompt := st.chat_input("Ask a question about your data..."):
            #     # Add user message to history
            #     st.session_state.messages.append({"role": "user", "content": prompt})
            #     with st.chat_message("user"):
            #         st.markdown(prompt)

            #     # Generate response
            #     with st.chat_message("assistant"):
            #         with st.spinner("Analyzing..."):
            #             try:
            #                 response = agent.invoke(prompt)
            #                 output = response["output"]
            #                 st.markdown(output)
            #                 st.session_state.messages.append({"role": "assistant", "content": output})
            #             except Exception as e:
            #                 st.error(f"An error occurred: {str(e)}")

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Please upload a CSV or Excel file to start analyzing.")

if __name__ == "__main__":
    main()