# import packages
import streamlit as st
import pandas as pd
import re
import os
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
from groq import Groq
import os

# --- New Groq client initialization ---
# Get the API key from the environment variable
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable not found. Please set it.")
        st.stop()
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# Define the model to use from Groq
GROQ_MODEL = "llama-3.3-70b-versatile"

# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir,"customer_reviews.csv")
    return csv_path

# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Helper function for RAG search
def rag_search(query, df, vectorizer, review_vectors):
    # Clean the query
    cleaned_query = clean_text(query)
    # Vectorize the query
    query_vector = vectorizer.transform([cleaned_query])
    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, review_vectors).flatten()
    # Get the top 5 most similar reviews
    top_indices = similarities.argsort()[-5:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            "score": similarities[i],
            "review": df.iloc[i]["SUMMARY"],
            "product": df.iloc[i]["PRODUCT"]
        })
    return results

# Helper function to generate a response using the Groq API
def generate_response(query, context):
    prompt = f"Answer the following question based on the provided customer reviews. If the answer is not in the reviews, say so.\n\nQuestion: {query}\n\nReviews:\n{context}"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=GROQ_MODEL,
            temperature=0.5,
            max_tokens=250,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating a response: {e}"

# Streamlit app layout
st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing and analysis app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            # Check if 'SUMMARY' column exists before trying to access it
            if "SUMMARY" in st.session_state["df"].columns:
                st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
                
                # --- New RAG components ---
                # Initialize TF-IDF Vectorizer
                vectorizer = TfidfVectorizer(stop_words='english')
                # Fit and transform the cleaned reviews
                review_vectors = vectorizer.fit_transform(st.session_state["df"]["CLEANED_SUMMARY"])
                # Store vectorizer and vectors in session state
                st.session_state["vectorizer"] = vectorizer
                st.session_state["review_vectors"] = review_vectors
                # --- End RAG components ---
                
                st.success("Reviews parsed, cleaned, and vectorized for RAG!")
            else:
                st.error("The 'SUMMARY' column was not found in the dataset.")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)
    
    st.subheader(f"Sentiment Score Distribution for {product}")
    # Create Plotly histogram
    fig = px.histogram(
        filtered_df, 
        x="SENTIMENT_SCORE", 
        nbins=10,
        title="Distribution of Sentiment Scores",
        labels={"SENTIMENT_SCORE": "Sentiment Score", "count": "Frequency"}
    )
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- RAG-based search and chatbot functionality ---
    st.subheader("🤖 GenAI Chatbot (RAG + Analytics)")
    if "vectorizer" in st.session_state and "review_vectors" in st.session_state:
        query = st.text_input("Ask a question about the reviews:")
        if query:
            # Perform RAG search to get relevant reviews
            results = rag_search(
                query,
                st.session_state["df"],
                st.session_state["vectorizer"],
                st.session_state["review_vectors"]
            )
            
            # Combine the reviews into a single context string
            context_text = "\n\n".join([f"Review: {r['review']}\nProduct: {r['product']}" for r in results])
            
            with st.spinner("Generating response..."):
                # Generate a response using the LLM and the retrieved context
                llm_response = generate_response(query, context_text)
                
                st.write("**Here's what the reviews say:**")
                st.info(textwrap.fill(llm_response, width=80))

                st.subheader("Top Retrieved Reviews (Context for the Answer):")
                for result in results:
                    with st.expander(f"Review for {result['product']} (Score: {result['score']:.2f})"):
                        st.write(result["review"])
    else:
        st.warning("Please click 'Parse Reviews' to enable the RAG chatbot.")