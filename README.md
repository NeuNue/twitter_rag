# Twitter RAG System

这是一个基于 LangChain、OpenAI 和 Pinecone 构建的检索增强生成（RAG）系统，用于处理和查询 Twitter 用户数据。

## 功能特点

- 使用 OpenAI 的 text-embedding-3-large 模型进行文本嵌入
- 使用 Pinecone 作为向量数据库进行高效的相似度搜索
- 支持 Excel 文件的数据导入和预处理
- 智能文本分块和重叠处理
- 基于上下文的问答系统
- 支持源文档追踪

## 环境要求

- Python 3.8+
- OpenAI API 密钥
- Pinecone API 密钥和环境
- 足够的 API 调用额度

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd <repository-name>
```

2. 安装依赖：
```bash
pip install langchain langchain-openai langchain-pinecone pinecone-client python-dotenv pandas openpyxl
```

3. 配置环境变量：
   - 复制 `.env.template` 文件为 `.env`
   - 填写以下配置：
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_openai_base_url  # 可选，用于自定义 API 端点
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

## 使用方法

### 1. 启动问答系统：
```bash
python test_rag.py
```

### 2. 交互方式
- 输入问题并按回车
- 系统会返回答案和相关的源文档
- 输入 'exit' 退出系统

## 示例问题

1. 查询用户信息：
   - "Who are the most followed crypto influencers?"
   - "Tell me about government officials in the dataset"

2. 统计信息：
   - "What's the average follower count of verified users?"
   - "When was the earliest account created?"

3. 主题分析：
   - "What are the main topics discussed by tech influencers?"
   - "Who are the most active users in the blockchain space?"

## 项目结构

```
.
├── README.md           # 项目文档
├── .env               # 环境变量配置
├── rag.py            # RAG 系统核心实现
├── test_rag.py       # 交互式测试脚本
└── merged_result.xlsx # 数据文件
```

## 核心组件

1. 数据处理 (`rag.py`)
   - Excel 文件读取和清理
   - 文本分块和预处理
   - 文档向量化

2. 向量存储 (Pinecone)
   - 高效的相似度搜索
   - 持久化存储
   - 增量更新支持

3. 问答系统
   - 基于上下文的答案生成
   - 源文档追踪
   - 自然语言交互
