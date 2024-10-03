import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import traceback

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")

# Assistant class with API key integration
class Assistant:
    def __init__(self, file_path, context):
        self.context = context
        self.docs = self.load_text(file_path)
        self.vectorStore = self.create_db(self.docs)
        self.chain = self.create_chain()
        self.chat_history = []
        self.is_new_user = False

    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def create_db(self, docs):
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return Chroma.from_documents(docs, embedding=embedding)

    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=openai_api_key
        )

        # Creating the prompt template with context explicitly defined
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant chatbot specifically focused on giving a tutorial on how to navigate the Atlas map, based on {context}."
                       "Your primary goal is to help users with {context} only. Use Australian English spelling in your response."),
            ("system", "Context: {context}"),
            ("system", "Instructions for {context}:"
                       "\n1. If given a one-word or vague query, ask for clarification before proceeding."
                       "\n2. For all users, provide the following general steps for finding data on a specific theme or indicator:"
                       "\n   - Direct users to open the Atlas maps"
                       "\n   - Instruct users to use the theme or indicator search box in Atlas maps"
                       "\n   - Explain that if data is available on the topic, it will appear as a dropdown"
                       "\n   - Do not interpret specific data or findings"
                       "\n3. Always relate your responses back to the user's original query, regardless of the theme or indicator."
                       "\n4. Never interpret the data, even when asked by the user. Instead, advise that you can only help with map navigation queries."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", "Remember to be concise, clear, and helpful in your responses - give a maximum of 3 sentences. "
                       "Focus exclusively on {context} and do not discuss other topics unless explicitly asked."
                       "After giving guidance, suggest one relevant follow-up query that you think the user may ask next based on the {context}."
                       "Always use Australian English spelling.")
        ])

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt,
            document_variable_name="context"  # This ensures context is passed correctly as an input
        )

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Given the above conversation about {context}, generate a search query to look up relevant information")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )

        return create_retrieval_chain(
            history_aware_retriever,
            chain
        )

    def process_chat(self, question):
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
            "context": self.context  # Pass context explicitly here
        })
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"]

    def reset_chat_history(self):
        self.chat_history = []

class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('Raw data - maps.txt', 'map navigation')

# Flask API route
@app.route('/')
def index():
    return "Welcome to the Chatbot API! Access the /chat endpoint to communicate with the chatbot."

# Chat route to handle incoming chat requests
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    chat_history = data.get("chat_history", [])
    context = "map navigation"

    if user_message:
        try:
            # Create an instance of MapAssistant and load documents and set up vector store
            assistant = MapAssistant()

            # Process the user message through LangChain
            bot_reply = assistant.process_chat(user_message)

            # Append the messages to chat history
            chat_history.append(HumanMessage(content=user_message))
            chat_history.append(AIMessage(content=bot_reply))

            # Serialize chat history before returning
            serialized_history = [{"type": "human", "content": msg.content} if isinstance(msg, HumanMessage)
                                  else {"type": "ai", "content": msg.content} for msg in chat_history]

            return jsonify({"reply": bot_reply, "chat_history": serialized_history})
        except Exception as e:
            # Log the full traceback for debugging
            print("Error: ", str(e))
            traceback.print_exc()
            return jsonify({"reply": "Sorry, there was an error processing your request.", "error": str(e)}), 500

    return jsonify({"reply": "No message provided."}), 400

# Interactive mode for CLI
def run_interactive_mode():
    assistant = MapAssistant()

    print("Hello! Welcome to the Atlas Map Navigation Assistant! Are you new to our interactive map platform? (Yes/No)")

    user_response = input("You: ").lower()
    if user_response in ['yes', 'y']:
        assistant.is_new_user = True
        print("Great! Let's start by familiarising you with the map platform.")
        print("You can start by reading the help screens. Please follow these steps:")
        print("1. Click on Atlas maps")
        print("2. Navigate to the right-hand side pane")
        print("3. Click the 'i' icon in the top right-hand corner")
        print("This will open the help screens. There are three screens covering different aspects of the platform: the National scale, Atlas menu items, and map interactions.")
        print("Are you ready to continue? (Yes/No)")
        continue_response = input("You: ").lower()
        if continue_response in ['yes', 'y']:
            print("Great! What specific question can I assist you with first?")
        else:
            print("Alright. Feel free to ask any questions when you're ready to explore further.")
    else:
        print("Welcome back! I'm here to assist you with any questions about our map platform. What can I help you with today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break

        try:
            response = assistant.process_chat(user_input)
            print("Assistant:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try that again. Could you rephrase your question?")

# Main entry point for CLI or Flask API
if __name__ == "__main__":
    # Check if running in CLI or Flask mode based on environment variable or argument
    if os.getenv("MODE") == "interactive":
        run_interactive_mode()
    else:
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
