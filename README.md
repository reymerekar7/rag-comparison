# rag-comparison

This is a repository with a streamlit application to compare RAG performance and outputs between OpenAI models and Cohere models.

## Start-up
- Clone the repository
- Create new virtual environment and install the dependencies from requirements.txt
- Command to run the streamlit app: ```streamlit run streamlit_comparison_app_v3.py```

## App Usage
- Choose your model to get started (Cohere Command R+ or OpenAI GPT-4)
- Enter your respective API key
- Upload a pdf, it will be parsed and embeddings will be generated
- Start chatting with your docs!
- In addition to an answer, there is another tab where you can see the top 3 pieces of retrieved context for the RAG pipeline along with the similarity scores



![Screenshot 2024-08-24 at 8 59 07 AM](https://github.com/user-attachments/assets/b571bd75-51b0-499c-a231-87c39451e034)
![Screenshot 2024-08-24 at 9 00 17 AM](https://github.com/user-attachments/assets/891fac74-8c4e-432a-928e-360e64f3ef40)
