from transformers import AutoTokenizer, AutoModel
import torch  

from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import LongContextReorder
import faiss
import os
import joblib
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2

class HuggingFaceEmbeddings:  
    def __init__(self, model_name='multilingual-e5-large'):  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.model = AutoModel.from_pretrained(model_name)  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model.to(self.device)  
  
    def embed_documents(self, documents):  
        # 将文档转换为模型输入  
        inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors="pt").to(self.device)  
        # 获取隐藏状态  
        with torch.no_grad():  
            outputs = self.model(**inputs)  
        # 取[CLS]标记的嵌入作为文档嵌入（或平均池化所有token的嵌入）  
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  
        return embeddings  
  
    def embed_query(self, query):  
        # 将查询转换为模型输入  
        inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(self.device)  
        # 获取隐藏状态  
        with torch.no_grad():  
            outputs = self.model(**inputs)  
        # 取[CLS]标记的嵌入作为查询嵌入  
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]  
        return embedding  

# Prompt library
EnglishTemplate_0 = """
You are a very good English native speaker. You are great at answering questions in English. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together in English to answer the broader question.

Here is a question:
{query}
"""


ChineseTemplate_0 = """
你是一个使用中文，熟悉中华文化的人，你非常擅长回答中文的问题。\
你非常擅长拆解困难的问题并把他们拆解成问题的组成部分，\
按照每个组成部分作答，并使用中文把他们合并成整个问题的完整回答。

下面是一个问题：
{query}
"""


def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

long_reorder = RunnableLambda(LongContextReorder().transform_documents)




# example of getting embedding from huggingface model
embeddings = HuggingFaceEmbeddings(model_name='/home/simon/disk1/Simon/Code/COLING/models/intfloat/multilingual-e5-large')  
prompt_templates = [EnglishTemplate_0, ChineseTemplate_0]  
prompt_embeddings = embeddings.embed_documents(prompt_templates)  
# print(prompt_embeddings)



# Route question to prompt 
def prompt_router(input):
    #   使用该链之前需要外部指定好embedder，待查询的embedding_prompt,以及待选prompt template

    
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    
    # Chosen prompt 
    return PromptTemplate.from_template(most_similar)




from typing import Any, List, Mapping, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.llms.base import LLM
class llm_wrapper(LLM):
    
    model_path_or_name:str
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "huggingface model for langchain"
    
    def _make_api_call(self, prompt: str) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.model_path_or_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path_or_name,
            load_in_4bit=True,
            #attn_implementation="flash_attention_2", # if you have an ampere GPU
        )
        llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, top_k=50, temperature=0.1))
 
        output = llm(prompt)
        return output
    def _call(self, prompt: str, **kwargs: Any) -> str:
        """Call to Together endpoint."""
        try:
            print("Making API call to Together endpoint.")
            return self._make_api_call(prompt)
        except Exception as e:
            print(f"Error in TogetherLLM _call: {e}", exc_info=True)
            raise

    
    




# example openai chain
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# chain_multilingual_template = (
#     {"query": RunnablePassthrough()}
#     | RunnableLambda(prompt_router)
#     | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
#     | StrOutputParser()
# )

# print(chain_multilingual_template.invoke("What's a black hole"))

# example
# /home/simon/disk1/Simon/Code/LLM/models/meta-llama/Llama-2-7b-hf
# llama_llm = llm_wrapper("/home/simon/disk1/Simon/Code/LLM/models/meta-llama/Meta-Llama-3-8B")

# chain_llama_tempalte= (
#     {"query": RunnablePassthrough()}
#     | RunnableLambda(prompt_router)
#     | llama_llm
#     | StrOutputParser()
# )

# example2
# default_llm example
# embeddings = HuggingFaceEmbeddings(model_name='/home/simon/disk1/Simon/Code/COLING/models/intfloat/multilingual-e5-large')  
# prompt_templates = [EnglishTemplate_0, ChineseTemplate_0]  


# llm_llama = llm_wrapper(model_path_or_name = "/home/simon/disk1/Simon/Code/LLM/models/meta-llama/Llama-2-7b-hf") 
# chain_default_tempalte= (
#     {"query": RunnablePassthrough()}
#     | RunnableLambda(prompt_router)
#     | llm_llama
#     | StrOutputParser()
# )



''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# example
# result = chain_llama_tempalte.invoke("what is llama?")
# result = chain_default_tempalte.invoke("what is gpt?")
# 本地未用instrcut/chat版本的llama 所以会出现回答复述/重复生成的情况
# print(result)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Load data
class FaissIdx_v2:

    def __init__(self, model, dim=1024, index_file=None):
        self.index = faiss.IndexFlatIP(dim)
        # Maintaining the document data
        self.doc_map = dict()
        self.model = model
        self.ctr = 0
        self.dim = dim
        if index_file is not None:  
            self.index = faiss.read_index(index_file)  
        else:  
            self.index = faiss.IndexFlatIP(dim)  
        
        if index_file and os.path.exists(index_file.replace('.index', '.pkl')):  
            self.doc_map = joblib.load(index_file.replace('.index', '.pkl'))      
            

    def add_doc(self, document_text):
        if isinstance(document_text, list):
            for i in range(len(document_text)):
                self.index.add(self.model.encode(document_text[i]))
                self.doc_map[self.ctr] = document_text # store the original document text
                self.ctr += 1
        elif isinstance(document_text, str):
            self.index.add(self.model.encode(document_text))
            self.doc_map[self.ctr] = document_text # store the original document text
            self.ctr += 1
    def search_doc(self, query, k=3):
        D, I = self.index.search(self.model.encode(query), k)
        return [{self.doc_map[idx]: score} for idx, score in zip(I[0], D[0]) if idx in self.doc_map]
    
    def save_doc(self,save_file_dir,db_name):
        import joblib
        embedded_text_name = db_name+".pkl"
        joblib.dump(self.doc_map,os.path.join(save_file_dir,embedded_text_name))
        index_name = db_name+".index"
        faiss.write_index(self.index, os.path.join(save_file_dir,index_name))
    
 
 
# From faiss to FAISS

 

# RAG 
# implementation for faiss
def retrieve_text_from_faiss(query_embedding, index, embedded_text,k=1):  
    # k = 1  # 假设我们只想要最相似的结果  
    
    # 将查询嵌入转换为FAISS期望的格式  
    query_embedding = query_embedding.reshape(1, -1).astype('float32')  
    
    distances, indices = index.search(query_embedding, k)  
      
    # 检索最相似嵌入的索引  
    best_index = indices[0][0]  
      
    # 使用索引从embeddings_dict中检索文本  
    retrieved_text = embedded_text[best_index]  # best_index直接用作字典的键  
      
    return retrieved_text    





def default_FAISS(embedder):
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embedder.encode("test").shape[1]),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores,embedder):
    ## 初始化一个空的 FAISS 索引并将其他索引合并到其中
    agg_vstore = default_FAISS(embedder)
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore


# langchain implementation

EnglishTemplate = """
You are a very good English native speaker. You are great at answering questions in English. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together in English to answer the broader question.
Here is the provided context:
{context}
Here is a question:
{query}
"""


    
    