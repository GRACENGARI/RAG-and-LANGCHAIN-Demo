import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from dotenv import load_dotenv
import os

# Load environment variables (e.g., API keys)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is set
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

# Streamlit page configuration
st.set_page_config(
    page_title="CyberShield SME Assistant", 
    page_icon="🛡️", 
    layout="wide"
)

# Custom CSS for cybersecurity theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .security-badge {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ CyberShield SME Assistant</h1>
    <p style="color: #e8f4fd; text-align: center; margin: 0;">
        AI-Powered Cybersecurity Guidance for Small & Medium Enterprises
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with platform info
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/1e3c72/ffffff?text=CyberShield", caption="Protecting SMEs in the Digital Age")
    st.markdown("### 🎯 Platform Features")
    st.markdown("""
    <div class="security-badge">Enterprise-Level Security</div>
    <div class="security-badge">No IT Department Required</div>
    <div class="security-badge">Real-Time Threat Detection</div>
    <div class="security-badge">Simplified Management</div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Risk Assessment")
    if st.button("🔍 Analyze Current Threats"):
        st.success("System Status: Secure ✅")
        st.info("Last Scan: 2 minutes ago")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load the cybersecurity document
@st.cache_resource
def load_and_process_document():
    try:
        # Load the cybersecurity research document
        loader = PyPDFLoader("C:/Users/grace/Desktop/254/RAG-and-LANGCHAIN-Demo/CyberShield SME Assistant.pdf")
        data = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = text_splitter.split_documents(data)
        
        # Create embeddings using OpenAI
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Store the document embeddings in FAISS
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Set up a retriever for similarity search
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        
        return retriever
    except FileNotFoundError:
        st.error("📄 Cybersecurity knowledge base not found. Please upload the PDF document.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading document: {str(e)}")
        return None

# Load document and create retriever
retriever = load_and_process_document()

if retriever:
    # Initialize the OpenAI model for the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.3, 
        max_tokens=500
    )
    
    # Add Memory to retain conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    # Define cybersecurity-focused system prompt
    system_prompt = (
        "You are CyberShield Assistant, an expert cybersecurity advisor specifically designed for Small and Medium Enterprises (SMEs). "
        "Your mission is to provide enterprise-level security guidance to businesses that don't have dedicated IT departments. "
        
        "CONTEXT: You help SMEs understand and implement cybersecurity best practices, threat prevention, "
        "incident response, and digital security strategies. Focus on practical, actionable advice that's accessible to non-technical business owners. "
        
        "GUIDELINES:\n"
        "- Prioritize SME-specific cybersecurity challenges and solutions\n"
        "- Explain technical concepts in business-friendly language\n"
        "- Provide actionable recommendations with clear next steps\n"
        "- Focus on cost-effective security measures suitable for smaller budgets\n"
        "- Emphasize prevention over complex remediation\n"
        "- Consider resource constraints typical of SMEs\n"
        
        "Use the following retrieved context to provide accurate, research-backed answers. "
        "If you don't know something specific, recommend consulting with cybersecurity professionals. "
        "Keep responses concise but comprehensive.\n\n"
        "Retrieved Context:\n{context}"
    )
    
    # Create the chat prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Define a tool for document retrieval
    def retrieve_documents(query):
        docs = retriever.get_relevant_documents(query)
        return docs
    
    retrieval_tool = Tool(
        name="Cybersecurity Knowledge Base",
        func=retrieve_documents,
        description="Retrieves relevant cybersecurity information and research specifically focused on SME security challenges"
    )
    
    # Create main chat interface
    st.markdown("### 💬 Ask Your Cybersecurity Questions")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.expander(f"💼 Question {i+1}: {question[:50]}..." if len(question) > 50 else f"💼 Question {i+1}: {question}"):
            st.markdown(f"**You:** {question}")
            st.markdown(f"**CyberShield:** {answer}")
    
    # Input query from user with examples
    st.markdown("#### 🔍 Common Questions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔐 Basic Security Setup"):
            st.session_state.current_query = "What are the essential cybersecurity measures every SME should implement immediately?"
    
    with col2:
        if st.button("📧 Email Security"):
            st.session_state.current_query = "How can SMEs protect against phishing attacks and email-based threats?"
    
    with col3:
        if st.button("💰 Budget-Friendly Solutions"):
            st.session_state.current_query = "What are cost-effective cybersecurity solutions for small businesses with limited budgets?"
    
    # Text input for custom queries
    query = st.text_input(
        "Ask me anything about cybersecurity for your business:", 
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., How do I protect my customer data? What should I do if I suspect a breach?"
    )
    
    # Clear the session state query after displaying
    if 'current_query' in st.session_state:
        del st.session_state.current_query
    
    # Process query
    if query and st.button("🚀 Get Security Guidance", type="primary"):
        with st.spinner("🔍 Analyzing cybersecurity best practices..."):
            try:
                # Create the RAG chain
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Invoke the RAG chain and get the response
                response = rag_chain.invoke({"input": query})
                
                # Store conversation history
                memory.save_context({"input": query}, {"output": response["answer"]})
                st.session_state.chat_history.append((query, response["answer"]))
                
                # Display the answer with styling
                st.markdown("### 🛡️ CyberShield Recommendation:")
                st.success(response["answer"])
                
                # Show relevant sources
                with st.expander("📚 Sources & Additional Context"):
                    if "context" in response:
                        st.markdown("**Research-backed information from:**")
                        for i, doc in enumerate(response["context"][:3]):
                            st.markdown(f"**Source {i+1}:** {doc.page_content[:200]}...")
                
                # Security tip
                st.info("💡 **Pro Tip:** Implement security measures gradually, starting with the most critical ones for your business type.")
                
            except Exception as e:
                st.error(f"❌ Error processing your query: {str(e)}")
                st.markdown("Please try rephrasing your question or contact our support team.")
    
    # Additional resources section
    with st.expander("📋 Quick Security Checklist for SMEs"):
        st.markdown("""
        **Immediate Actions:**
        - [ ] Enable multi-factor authentication on all business accounts
        - [ ] Regular software updates and patches
        - [ ] Employee cybersecurity training
        - [ ] Secure backup strategy (3-2-1 rule)
        - [ ] Basic firewall and antivirus protection
        
        **Monthly Reviews:**
        - [ ] Access permissions audit
        - [ ] Security incident review
        - [ ] Backup testing
        - [ ] Password policy compliance
        """)

else:
    st.error("⚠️ Unable to load the cybersecurity knowledge base. Please ensure the PDF document is available.")