# http://127.0.0.1:5000/v1/ 

from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from retriever_module.retriever import llm_wrapper
app = Flask(__name__)

# # 加载模型和分词器
# llm = llm_wrapper(model_path_or_name="/home/simon/disk1/Simon/Code/LLM/models/meta-llama/Llama-2-7b-hf")

# llm("hi")

from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from transformers import BitsAndBytesConfig



import torch 
class MultiConversation: 
    def __init__(self, model_name, init_prompt, device='cuda' if torch.cuda.is_available() else 'cpu'): 
        nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.device = device 
        self.tokenizer =  AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config = nf4_config)
        self.messages = [] 
        self.messages.append(self.tokenizer(init_prompt, return_tensors="pt"))
    def qa(self, question): 
        # 将新问题添加到消息列表中 
        question_tensors = self.tokenizer(question, return_tensors="pt")
        # 将新问题和历史消息合并（注意：这里需要稍微处理一下，因为模型可能需要特定的输入格式） 
        # # 假设模型可以直接接受这样的序列（这取决于模型的实际要求） 
        # # 注意：这里只是简单地将它们连接在一起，实际情况可能需要更复杂的处理 
        # print(self.messages)
        # print([m['input_ids'] for m in self.messages])
        print([m['input_ids'] for m in self.messages] + [question_tensors['input_ids']])
        combined_input = torch.cat([m['input_ids'] for m in self.messages] + [question_tensors['input_ids']], dim=1) 
        attention_mask = torch.cat([torch.ones_like(m['attention_mask']) for m in self.messages] + [question_tensors['attention_mask']], dim=1) 
        # 你可以根据需要调整这些参数（如 max_length, top_p, temperature 等） 
        generated_ids = self.model.generate( input_ids=combined_input, attention_mask=attention_mask, max_length=4096, 
                                            # 你可以根据需要调整这个长度 
                                            temperature=1.0, top_p=0.9, top_k=0, repetition_penalty=1.0, do_sample=True, ) 
        # 解码生成的 ID 以获取文本 
        answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True) 
        #保存回答到消息列表中（这里只是简单记录，实际使用可能需要更复杂的数据结构
        self.messages.append({ 'input_ids': question_tensors['input_ids'], 'attention_mask': question_tensors['attention_mask'], 'generated_text': answer }) 
        return answer 


model_name = '/home/simon/disk1/Simon/Code/LLM/models/nvidia/Llama3-ChatQA-1.5-8B' 
init_prompt = "you are a userful helper" 
conv = MultiConversation(model_name, init_prompt) 
@app.route('/v1/chat/completions', methods=['POST'])
def predict():
    # 解析请求数据
    data = request.json
    # print("data here:",data)
    # print(data.keys())
    messages = data["messages"]
    # print(type(messages),len(messages))
    text = messages[0]['content']
    # print(text)
    ans = conv.qa(text)
    # print("Ans:",ans)
    return jsonify({'content': ans})        



if __name__ == '__main__':
    app.run(debug=True,port=5000)