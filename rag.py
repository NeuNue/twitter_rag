# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from typing import List
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone clients
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize OpenAI embeddings with text-embedding-3-large model
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="text-embedding-3-large"  # 使用3072维度的模型
)

# Create text splitter for document processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

def clean_column_name(col):
    """清理列名"""
    if pd.isna(col) or col == '' or str(col).startswith('Unnamed:'):
        return None
    return str(col).strip()

def process_excel_to_documents(file_path: str) -> List[Document]:
    """
    读取Excel文件并转换为Document列表
    """
    print(f"Reading Excel file: {file_path}")
    # 跳过前两行（标题和描述），使用第三行作为列名
    df = pd.read_excel(file_path, skiprows=2)
    
    # 清理列名
    df.columns = [clean_column_name(col) for col in df.columns]
    # 删除所有列名为None的列
    df = df.loc[:, [col for col in df.columns if col is not None]]
    
    # 删除全为空的行
    df = df.dropna(how='all')
    
    print(f"Cleaned DataFrame shape: {df.shape}")
    
    # 将所有列转换为字符串并合并
    documents = []
    for idx, row in df.iterrows():
        # 只处理非空值
        content_parts = []
        for col, val in row.items():
            if pd.notna(val) and str(val).strip() != '':
                content_parts.append(f"{col}: {str(val).strip()}")
        
        if content_parts:  # 只有当有内容时才创建文档
            content = "\n".join(content_parts)
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "row_index": idx
                }
            )
            documents.append(doc)
    
    print(f"Created {len(documents)} documents from Excel file")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    分割文档为更小的块
    """
    print("Splitting documents into chunks...")
    return text_splitter.split_documents(documents)

def get_or_create_vectorstore(documents: List[Document] = None, index_name: str = "twitter-rag"):
    """
    获取或创建向量存储
    """
    # 检查索引是否存在
    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=3072,  # text-embedding-3-large dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
        if documents is not None:
            print(f"Storing {len(documents)} documents in Pinecone...")
            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=index_name
            )
        else:
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
    else:
        print(f"Using existing Pinecone index: {index_name}")
        # 获取索引统计信息
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Index contains {stats['total_vector_count']} vectors")
        
        # 如果提供了新文档且索引为空，则添加文档
        if documents is not None and stats['total_vector_count'] == 0:
            print(f"Index is empty. Storing {len(documents)} documents...")
            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=index_name
            )
        else:
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
    
    return vectorstore

def create_qa_chain(vectorstore):
    """
    创建问答链
    """
    # Create RAG chain
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer: """

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain

def main(load_data: bool = False):
    """
    主函数
    load_data: 是否需要加载新数据
    """
    if load_data:
        # 处理Excel文件
        excel_path = "merged_result.xlsx"
        documents = process_excel_to_documents(excel_path)
        
        # 分割文档
        split_docs = split_documents(documents)
        
        # 获取或创建向量存储
        vectorstore = get_or_create_vectorstore(split_docs)
    else:
        # 直接使用现有的向量存储
        vectorstore = get_or_create_vectorstore()
    
    # 创建问答链
    qa_chain = create_qa_chain(vectorstore)
    
    print("Process completed successfully!")
    return qa_chain

if __name__ == "__main__":
    main() 