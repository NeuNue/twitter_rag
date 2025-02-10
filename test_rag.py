from rag import main
import argparse

def test_rag(load_data: bool = False):
    """
    测试 RAG 系统
    load_data: 是否需要加载新数据
    """
    # 获取问答链
    qa_chain = main(load_data=load_data)
    
    print("\nRAG system is ready! Type 'exit' to quit.")
    
    while True:
        # 获取用户输入
        question = input("\nEnter your question: ")
        
        # 检查是否退出
        if question.lower() == 'exit':
            break
        
        # 使用问答链获取答案
        result = qa_chain({"query": question})
        
        # 打印答案
        print("\nAnswer:", result["result"])
        
        # 打印来源
        print("\nSources:")
        for doc in result["source_documents"]:
            print("-" * 50)
            print(doc.page_content)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Test RAG system')
    parser.add_argument('--load-data', action='store_true', 
                      help='Load new data into Pinecone')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行测试
    test_rag(load_data=args.load_data) 