import os
import chromadb
# conda install pyyaml importlib_metadata     
import numpy as np
from langchain import hub
# from langchain.llms import ollama
from langchain_community.llms import ollama
from langchain.chains import RetrievalQA
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores.chroma import Chroma
# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import re
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
from tqdm import tqdm
import logging
import torch

from tqdm import tqdm
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union

from transformers import AutoModel, AutoTokenizer


# useing load_documents you can directly let the data store in FAISS
def load_documents(document_dir):
    """Loads PDF documents from the specified directory, handling errors and splitting PDFs."""

    loader_cls = PyPDFLoader  # Only use PyPDFLoader for this function
    files = os.listdir(document_dir)
    for index, filename in enumerate(tqdm(files, desc="Processing files", total=len(files))):
        if os.path.splitext(filename)[1].lower() == ".pdf":  # Check for lowercase ".pdf" extension
            try:
                loader = loader_cls(os.path.join(document_dir, filename))             
                doc = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=30)
                chunks = text_splitter.split_documents(doc)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")  # Log any errors

    return chunks

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# chunks = load_documents(r"D:\COLING\Multilingual AI LAW")
# print(chunks)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



class TextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))') 
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list



def load_file(filepath):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = TextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)        
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = TextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# docs = load_file("/home/simon/disk1/Simon/Code/COLING/data_pre.html")
# docs = load_file("/home/simon/disk1/Simon/Code/COLING/Multilingual AI LAW/Croatian.pdf")
# docs = load_file("/home/simon/disk1/Simon/Code/COLING/multilingual_rag/README.md")
# print(docs)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class EmbeddingModel:
    def __init__(
            self,
            model_name_or_path: str='maidalun1020/bce-embedding-base_v1',
            pooler: str='cls',
            use_fp16: bool=False,
            device: str=None,
            **kwargs
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model.to(self.device)  

        assert pooler in ['cls', 'mean'], f"`pooler` should be in ['cls', 'mean']. 'cls' is recommended!"
        self.pooler = pooler
        
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        # logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16};\t embedding pooling type: {self.pooler};\t trust remote code: {kwargs.get('trust_remote_code', False)}")

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int=32,    # 256 for 14G
            max_length: int=512,
            normalize_to_unit: bool=True,
            return_numpy: bool=True,
            enable_tqdm: bool=True,
            query_instruction: str="",
            **kwargs
        ):
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        with torch.no_grad():
            embeddings_collection = []
            for sentence_id in tqdm(range(0, len(sentences), batch_size), desc='Extract embeddings', disable=not enable_tqdm):
                if isinstance(query_instruction, str) and len(query_instruction) > 0:
                    sentence_batch = [query_instruction+sent for sent in sentences[sentence_id:sentence_id+batch_size]] 
                else:
                    sentence_batch = sentences[sentence_id:sentence_id+batch_size]
                inputs = self.tokenizer(
                        sentence_batch, 
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs_on_device, return_dict=True)

                if self.pooler == "cls":
                    embeddings = outputs.last_hidden_state[:, 0]
                elif self.pooler == "mean":
                    attention_mask = inputs_on_device['attention_mask']
                    last_hidden = outputs.last_hidden_state
                    embeddings = (last_hidden * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                else:
                    raise NotImplementedError
                
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embeddings_collection.append(embeddings.cpu())
            
            embeddings = torch.cat(embeddings_collection, dim=0)
        
        if return_numpy and not isinstance(embeddings, ndarray):
            embeddings = embeddings.numpy()
        
        return embeddings
    
    
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
    
    # def __call__(self, *args, **kwargs):  
    def __call__(self, *args):  
        # 这里可以添加任何你想在“调用”实例时执行的代码  
        # 例如，我们可以简单地返回实例的某个属性，或者基于输入参数进行一些计算  
        # print(f"CallableInstance called with args: {args} and kwargs: {kwargs}")  
        return self.embed_query(*args)  # 或者其他基于args和kwargs的计算结果  

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
# from langchain_community.document_loaders import UnstructuredHTMLLoader
# loader = UnstructuredHTMLLoader("/home/simon/disk1/Simon/Code/COLING/data_pre.html")
# data = loader.load()    
# print(data)



# sentences = ['sentence_0', 'sentence_1']

# # init embedding model
# model = EmbeddingModel(model_name_or_path="/home/simon/disk1/Simon/Code/COLING/models/BAAI/bge-reranker-v2-m3")

# # extract embeddings
# embeddings = model.encode(sentences)
# print(embeddings)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def collect_docs(document_dir,is_only_pdf=False):
    all_docs = []
    if is_only_pdf:
        all_docs = load_documents(document_dir)
    else:
        files = os.listdir(document_dir)
        for index, filename in enumerate(tqdm(files, desc="Processing files", total=len(files))):
            try:
                docs = load_file(os.path.join(document_dir,filename))
            except:
                continue
            all_docs += docs

    return all_docs



class FaissIdx:

    def __init__(self, model, dim=1024):
        self.index = faiss.IndexFlatIP(dim)
        # Maintaining the document data
        self.doc_map = dict()
        self.model = model
        self.ctr = 0

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
    

# Langchain FAISS 接口实现

import torch  
  
def torch_gc():  
    """  
    释放PyTorch GPU缓存。  
    """  
    if torch.cuda.is_available():  
        torch.cuda.empty_cache()  
        print("GPU memory cache cleared.")  
    else:  
        print("CUDA is not available. Skipping GPU memory cache clearing.")  
def batch_save_FAISS(docs,embedder,vs_path="embed/",batch_size = 1 ,from_type="documents"):
    # from_type is from ["documents","texts"]
    
    # 在Langchain API下 批量保存FAISS：
    if from_type=="documents":
        batch_idx = 0
        for i in tqdm(range(0, len(docs), batch_size)):
            batch_docs = docs[i:i + batch_size]
            print(f"Building vector db from {batch_idx} batch docs")
            if i == 0:
                vector_store = FAISS.from_documents(batch_docs,embedder)  # docs 为Document列表
                
                torch_gc()
            else:
                vector_store_append = FAISS.from_documents(batch_docs,embedder)  # # docs 为Document列表
                print(f"Merging vector db, batch {batch_idx}")
                vector_store.merge_from(vector_store_append)  # 合并向量库
                torch_gc()
            batch_idx += 1
        print(f"Saving vector db to {vs_path}")
        vector_store.save_local(vs_path)        

    elif from_type=="texts":
        batch_idx = 0
        for i in tqdm(range(0, len(docs), batch_size)):
            batch_docs = docs[i:i + batch_size]
            print(f"Building vector db from {batch_idx} batch docs")
            if i == 0:
                # print(batch_docs)
                vector_store = FAISS.from_texts(batch_docs,embedder)  # docs 为Document列表
                
                torch_gc()
            else:
                vector_store_append = FAISS.from_texts(batch_docs,embedder)  # # docs 为Document列表
                print(f"Merging vector db, batch {batch_idx}")
                vector_store.merge_from(vector_store_append)  # 合并向量库
                torch_gc()
        batch_idx += 1
        print(f"Saving vector db to {vs_path}")
        vector_store.save_local(vs_path)    


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
# 优点：集成了数据添加保存，缺点使用的时候会占用显存  
# example
# docs = collect_docs("/home/simon/disk1/Simon/Code/COLING/Multilingual AI LAW",is_only_pdf=True)
# docs = collect_docs("/home/simon/disk1/Simon/Code/COLING/Multilingual AI LAW",is_only_pdf=False)
# print(docs)    
# model = EmbeddingModel(model_name_or_path="/home/simon/disk1/Simon/Code/COLING/models/BAAI/bge-reranker-v2-m3")

# method 1
# faiss_index = FaissIdx(model=model)
# sentences = ['The nature of AI, which often relies on large and varied datasets and which may be embedded in any product or service circulating freely within the internal market, entails that the objectives of this proposal cannot be effectively achieved by Member States alone. ',
#              'sentence_1']
# for doc in tqdm(docs):
#     faiss_index.add_doc(doc.page_content)
# results = faiss_index.search_doc(sentences)
# print(results)  
# faiss_index.save_doc("/home/simon/disk1/Simon/Code/COLING/RAG_data/faiss_idx_db","faiss_idx_db")

# method2
# FAISS_DB = "/home/simon/disk1/Simon/Code/COLING/RAG_data/FAISS_DB"
# batch_save_FAISS(docs,model,FAISS_DB)




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 
# faiss 实现
def create_vectorstore(docs,model_name_or_path,save_path=None,save_name="embed",is_save=True):
    # save_path dir
    # 提取所有文档的文本内容  
    texts = [doc.page_content for doc in docs]  
    model = EmbeddingModel(model_name_or_path)
    # 生成嵌入向量  
    embeddings =  model.encode(texts)
    
    # 初始化 FAISS 索引  
    d = embeddings.shape[1]  # 向量的维度  
    index = faiss.IndexFlatL2(d)  # 使用 L2 距离的精确搜索  
    
    # 添加向量到索引  
    index.add(embeddings)  
    
    if save_path!= None:
        # （可选）保存 FAISS 索引到磁盘  
        faiss.write_index(index, save_path+"/"+save_name+".index")  
        metadata_list = [doc.metadata for doc in docs]  
        import json  
        with open(save_path+"/"+save_name+".json", 'w') as f:  
            json.dump(metadata_list, f)  
        import joblib
        joblib.dump(texts,save_path+"/"+save_name+".pkl")




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# example
# docs = collect_docs("/home/simon/disk1/Simon/Code/COLING/Multilingual AI LAW",is_only_pdf=True)
# docs = collect_docs("/home/simon/disk1/Simon/Code/COLING/Multilingual AI LAW",is_only_pdf=False)
# print(docs)
# create_vectorstore(docs,"/home/simon/disk1/Simon/Code/COLING/models/intfloat/multilingual-e5-large","/home/simon/disk1/Simon/Code/COLING/RAG_data")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def load_vectorstore(save_path,save_name):
    index = faiss.read_index(save_path + "/" + save_name + ".index")
    return index
    
def load_embedded_text(save_path,save_name):
    import joblib
    text = joblib.load(save_path + "/" + save_name + ".pkl")
    return text

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
## example
# index = load_vectorstore("/home/simon/disk1/Simon/Code/COLING/RAG_data","embed") 
# embedded_text = load_embedded_text("/home/simon/disk1/Simon/Code/COLING/RAG_data","embed")
# sentences = ['The nature of AI, which often relies on large and varied datasets and which may be embedded in any product or service circulating freely within the internal market, entails that the objectives of this proposal cannot be effectively achieved by Member States alone. ',
#              'sentence_1']

# # init embedding model
# model = EmbeddingModel(model_name_or_path="/home/simon/disk1/Simon/Code/COLING/models/BAAI/bge-reranker-v2-m3")

# # extract embeddings
# embeddings = model.encode(sentences)
# # print(embeddings)
# search_embedding = np.array(embeddings[0]) # 确保它是numpy数组，并且没有多余的维度  
  
# # 使用FAISS进行搜索  
# k = 4  # 你想找到的最近邻的数量  
# distances, indices = index.search(search_embedding.reshape(1, -1).astype('float32'), k)  
  
# # 打印结果  
# for i in range(k):  
#     print(f"Distance: {distances[0][i]:.4f}, Index: {indices[0][i]}")   
#     print(f"Text_retrived:{embedded_text[indices[0][i]]}")
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



