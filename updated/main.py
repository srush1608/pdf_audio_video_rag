import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, AssemblyAIAudioTranscriptLoader
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  

def validate_env_variables():
    if not DB_CONNECTION_URL_2:
        raise ValueError("Database connection URL is missing!")
    print("Database connection URL loaded successfully.")

    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is missing!")
    print("Google API key loaded successfully.")

    if not ASSEMBLYAI_API_KEY:
        print("AssemblyAI API key is missing! Audio processing will be skipped.")

    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing!")
    print("Groq API key loaded successfully.")

validate_env_variables()

def load_text_samples(file_paths):
    texts = []
    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                print(f"\n\n\nProcessing PDF file: {file_path}\n\n\n")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                print(documents)
                for doc in documents:
                    texts.append(doc.page_content)
                print(f"\n\nProcessed PDF file: {file_path}\n\n")

            elif file_path.startswith("https://www.youtube.com/watch?v=1bUy-1hGZpI"):
                print(f"\n\n\nProcessing YouTube video from {file_path}\n\n\n")
                loader = YoutubeLoader.from_youtube_url(file_path)
                documents = loader.load()
                print(documents)
                for doc in documents:
                    texts.append(doc.page_content)
                print(f"Processed YouTube video: {file_path}")

            elif file_path.endswith((".mp3", ".wav")):
                print(f"Processing audio file: {file_path}")
                if ASSEMBLYAI_API_KEY:
                    try:
                        loader = AssemblyAIAudioTranscriptLoader(file_path, api_key=ASSEMBLYAI_API_KEY)
                        documents = loader.load()

                        if documents:
                            print(f"Successfully processed audio file: {file_path}")
                            for doc in documents:
                                texts.append(doc.page_content)
                        else:
                            print(f"No documents returned from audio processing for: {file_path}")

                    except Exception as e:
                        print(f"Error processing audio file {file_path}: {e}")

                else:
                    print("AssemblyAI API key is missing; skipping audio processing.")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return texts

def chunk_texts(documents, chunk_size=500, overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))
    print(f"Generated {len(chunks)} chunks.")
    return chunks

def store_embeddings_pgvector(texts, table_name):
    try:
        embeddings_generator = GoogleGenerativeAIEmbeddings(
            api_key=GOOGLE_API_KEY,
            model="models/embedding-001"  
        )

        vectorstore = PGVector(
            connection_string=DB_CONNECTION_URL_2,
            embedding_function=embeddings_generator,
        )

        vectorstore.add_texts(texts, table_name=table_name)
        print(f"Data successfully stored in the '{table_name}' table.")
    except Exception as e:
        print(f"Error storing embeddings in PGVector: {e}")

def similarity_search_with_response(query, table_name="langchain_pg_embedding"):
    try:
        chatgroq = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-8b-8192",
            temperature=0.0,
            max_retries=2
        )

        prompt_template = PromptTemplate(
        input_variables=["query"], 
        template="Query: {query}\nPlease generate a detailed response based on the provided documents."
        )
        
        llm_chain = prompt_template | chatgroq

        print(f"\n\nGenerating response for query: {query}\n\n")
        response = llm_chain.invoke({"query": query})

        print("\n\nGenerated response:\n\n", response)
        return response

    except Exception as e:
        print(f"Error in similarity search or LLM response generation: {e}")
        return "An error occurred while processing your query."

if __name__ == "__main__":
    try:
        # Define file paths
        file_paths = [
            "./documents/attention.pdf",
            "https://www.youtube.com/watch?v=1bUy-1hGZpI",
            "./documents/langchain_agents_webinar.wav"
        ]



        store_embeddings_pgvector(chunks, table_name="langchain_pg_embedding")
        while True:
            user_query = input("Enter your query (type 'exit' to stop): ")
            if user_query.lower() == "exit":
                print("Exiting the program...")
                break
            print("User Query:", user_query)

            response = similarity_search_with_response(user_query)
            print("\nFinal Response:\n", response)

    except Exception as e:
        print(f"Error in processing: {e}")


