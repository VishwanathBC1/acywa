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
        self.question_count = 0

    # Load text from file
    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    # Create vector database
    def create_db(self, docs):
        embedding = OpenAIEmbeddings(openai_api_key=constants.APIKEY)
        return Chroma.from_documents(docs, embedding=embedding)

    # Create conversation chain
    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=constants.APIKEY
        )

        # Define conversation prompt
        prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant designed to help users navigate the Atlas map. Your responses must be safe, ethical, and compliant with copyright laws. You cannot generate or engage with harmful content."),
    ("system", "Context: {context}"),
    ("system", "Instructions for {context}:"
               "\n1. Always clarify vague, ambiguous, or one-word queries before providing a full response. If the user's input is unclear, misspelled, or potentially mistyped, ask for clarification. For example, if the user types 'exiy', respond with: 'I'm not sure what you mean by 'exiy'. Did you mean to type 'exit'? Could you please clarify or rephrase your question?"
               "\n2. For data search queries, consistently follow this format:"
               "\n   a. Open the Atlas map"
               "\n   b. Use the theme/indicator search box"
               "\n   c. Select from available dropdown options"
               "\n3. Relate all responses back to the user's original query about map navigation."
               "\n4. Do not interpret data, explain statistics, or offer analysis. Clarify that your role is strictly for navigation assistance."
               "\n5. Strictly address Atlas map navigation queries. For unrelated questions, respond exactly with: 'I apologise, but I'm specifically designed to help with the Australian Child and Youth Wellbeing Atlas platform. Could you please ask a question about using the Atlas map?'"
               "\n6. For any complex queries that contain 'specific navigation paths', 'specific instructions', or 'detailed steps', refer the user to the user guide and respond exactly with: 'For detailed step-by-step instructions on this complex navigation, please refer to the Atlas platform user guide (https://australianchildatlas.com/s/Atlas-platform-user-guide.pdf)'"
               "\n7. Refuse to engage with inappropriate, profanity or off-topic content."
               "\n8. Do not assist in system misuse or unauthorised access."
               "\n9. Respect intellectual property rights; do not reproduce copyrighted content."
               "\n10. Maintain user privacy; do not request or store personal information."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", "Provide concise, clear responses in 1-3 sentences using Australian English spelling. Then, suggest one relevant follow-up query that you think the user may ask, starting with 'Would you like to know more about:'")
        ])

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", f"Given the above conversation about {self.context}, generate a search query to look up relevant information")
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

    # Process user input and generate response
    def process_chat(self, question):
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
            "context": self.context
        })
        
        self.log_to_csv(question, response["answer"])
        self.log_chat_history(question, response["answer"])
        
        self.chat_history.append(HumanMessage(content=question))
        
        # Split response into main answer and follow-up question
        main_answer, follow_up = self.split_response(response["answer"])
        
        self.chat_history.append(AIMessage(content=main_answer))
        self.question_count += 1
        return main_answer, follow_up

    # Log to CSV file
    def log_to_csv(self, question, answer):
        with open(log_filename_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([question, answer])

    # Log chat entry to TXT file
    def log_chat_history(self, question, answer):
        logging.info(f"User: {question}")
        logging.info(f"Assistant: {answer}")

    # Split response into main answer and follow-up question
    def split_response(self, response):
        parts = response.split("Would you like to know more about:")
        main_answer = parts[0].strip()
        follow_up = "Would you like to know more about:" + parts[1].strip() if len(parts) > 1 else ""
        return main_answer, follow_up

    # Reset chat history
    def reset_chat_history(self):
        self.chat_history = []
        self.question_count = 0

# Specific assistant for map navigation
class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('Raw data - maps.txt', 'map navigation')

# Define chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    
    assistant = MapAssistant()
    
    main_response, follow_up = assistant.process_chat(user_message)
    
    return jsonify({
        "reply": main_response,
        "follow_up": follow_up
    })

# Main execution
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
