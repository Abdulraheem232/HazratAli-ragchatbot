import streamlit as st
from connect_memory_with_llm import chain as qa_chain

st.title("Hazrat Ali chatbot")
st.write("A ai Hazrat Ali chatbot made using rag technoloy and groq it will answer your questions related to Hazrat Ali (R.A) and his life.")
text_query = st.text_area(height=250,placeholder="Enter your query related to Hazrat Ali here..",label="Hazrat Ali related query")
submit_button = st.button("Enter")


if submit_button or text_query:
   response = qa_chain.invoke({"query":text_query})
   st.chat_message("ai").markdown(response["result"])
   st.write("Source")
   sources = response["source_documents"][0]
   st.code(f"Page {sources.metadata["page"]+1} out of {sources.metadata["total_pages"]} of {sources.metadata["source"]}")