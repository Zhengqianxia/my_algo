from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util as st_utils
import torch
import torch.nn as nn
import requests
import os
import json
import datetime
import numpy as np
from .lock import ReadWriteLock

class SbertEncoder(nn.Module):
    def __init__(self, device=None, model_path=""):
        super().__init__()
        self.device = device
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)

    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_sentence_embeddings(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

class LLMWrapper:
    def __init__(self, call_type, device=None, url="", username="", password=""):
        self.call_type = call_type  # "pr" or "rs"
        self.device = device
        self.url = url
        self.username = username
        self.password = password

        self.values = {}  # 用来存储 (state_caption, action_caption) 和对应的pr/rs值
        self.lock = ReadWriteLock()  # 使用 ReadWriteLock 类

        if self.call_type == "pr":
            self.sbert_encoder = SbertEncoder(device=self.device)  # 实例化SbertEncoder
            self.sbert_encoder.to(self.device)

    def state_captioner(self, cur_state):
        """
        cur_state: [是否绿灯，是否等待区域] 
        """
        if cur_state[0].item() == 1.0:  # 1.0是绿灯，0.0则不是
            state_caption = 'green'
        else:
            state_caption = 'red'
        return state_caption

    def action_captioner(self, action):  
        """
        action: [steer方向, motor速度]
        steer: -1往左, 0向前, 1往右
        motor: 1是往前, 0不动, -1后退
        """
        if action[1] > 0.5:
            action_caption = ["accelerate", "continue driving"]
        elif action[1] > 0.0:
            action_caption = ["decelerate", "continue driving"]
        elif action[1] == 0.0:
            action_caption = ["stop and wait"]
        else:
            action_caption = ["reversing"]

        return action_caption
    
    def get_value(self, state, action):
        if self.call_type == "pr":  # 计算正则项
            """
            state: [256,2]  [是否绿灯，是否等待区域]
            action: [256,2] [steer方向, motor速度]
            """
            pr_list = []

            # 将 state 移动到 CPU，以便可以转换为 NumPy 数组
            state_cpu = state.cpu() if state.is_cuda else state  # 检查是否在 GPU 上
            waiting_indices = np.where(state_cpu[:, 1] == 1.0)[0]  # 找到采样样本中所有到达等待区域的索引

            for i in waiting_indices:
                state_caption = self.state_captioner(state[i])
                action_caption = self.action_captioner(action[i])
                cur_key = (state_caption, " and ".join(action_caption))
                # 使用读锁来检查是否已有相应的值
                with self.lock.read():
                    if cur_key in self.values:
                        pr = self.values[cur_key]  # 已存在
                        pr_list.append(pr)
                        continue

                # 如果没有，使用写锁生成并保存新的回答
                with self.lock.write():
                    # 再次检查是否已经被其他线程写入
                    if cur_key in self.values:
                        pr = self.values[cur_key]  # 已存在
                        pr_list.append(pr)
                        continue
                    # 调用 LLM 生成新的回答
                    llm_goals_list = self.generate_goals(state_caption)
                    if llm_goals_list is None:
                        continue
                    llm_goals_embedding = self.sbert_encoder.get_sentence_embeddings(llm_goals_list)
                    action_embedding = self.sbert_encoder.get_sentence_embeddings(action_caption)
                    pr = self.calcu_pr(action_embedding, llm_goals_embedding)
                    pr_list.append(pr)
                    # 保存新的值
                    self.values[cur_key] = pr
                    print(f'Add {self.call_type} values pair: {cur_key} to {pr} \n')

            average_pr = np.mean(pr_list) if pr_list else 0
            if average_pr:
                pr_factor = 0.05  # 正则项的重要性因子
                regularization_term = pr_factor * average_pr
            else:
                regularization_term = 0
            return regularization_term

        elif self.call_type == "rs":   # 计算偏好奖励
            """
            输入: state: [是否绿灯，是否等待区域]
                action: [steer方向, motor速度]
            """
            preference_reward = 0
            human_preference = "safe"  # 设置的人类偏好 /urgent
            
            state_caption = self.state_captioner(state)
            action_caption = self.action_captioner(action)
            action_caption = " and ".join(action_caption)
            # action_caption = action_caption[-1]
            cur_key = (state_caption, action_caption)

            # 使用读锁来检查是否已有相应的值
            with self.lock.read():
                if cur_key in self.values:
                    return self.values[cur_key]  # 已存在，直接返回

            # 如果没有，使用写锁生成并保存新的回答
            with self.lock.write():
                # 再次检查是否已经被其他线程写入
                if cur_key in self.values:
                    return self.values[cur_key]  # 已存在，直接返回

                # 调用 LLM 生成新的回答
                try:
                    prompt = (
                        f"You are a {human_preference}-conscious driver. "
                        f"You see a zebra crossing, a {state_caption} traffic light. You are driving. You can go straight, turn right. "
                        f"You {action_caption} before the crosswalk. "
                        f"Are the actions you're taking {human_preference} ? "
                        f"Only allowed to answer 'Yes' or 'No'."
                    )
                    print("\nrs_prompt: ", prompt)

                    messages = [{
                            "role": "user",
                            "content": prompt
                    }]

                    response = requests.post(self.url,
                                json={"model": "llama3.3",  # "deepseek-r1:14b", "llama3.3"
                                    "messages": messages, 
                                    "stream": True
                                    },
                                auth=(self.username, self.password))
                    response.raise_for_status()
                    output = ""
                    for line in response.iter_lines():
                        body = json.loads(line)
                        if "error" in body:
                            raise Exception(body["error"])
                        if body.get("done") is False:
                            message = body.get("message", "")
                            content = message.get("content", "")
                            output += content
                            # the response streams one token at a time, print that as we receive it
                            # print(content, end="", flush=True)

                        if body.get("done", False):
                            response = output
                            break
                    print("rs_response:", response)

                    # 保存到文件
                    json_new_data = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "query": prompt,
                                "output": response}
                    file_path = os.path.join(os.getcwd(), 'LLMResults.json')
                    if not os.path.exists(file_path):
                        json_data = []
                        with open(file_path, 'w', encoding='utf-8') as file:
                            json.dump(json_data, file)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        json_data = json.load(file)
                    json_data.append(json_new_data)
                    with open(file_path, 'w', encoding="utf-8") as file:
                        json.dump(json_data, file, indent=2, ensure_ascii=False)

                    # 解析LLM的输出 response
                    if "Yes" in response or "yes" in response:
                        preference_reward = 1
                    elif "No" in response or "no" in response:
                        preference_reward = 0
                    else:
                        preference_reward = 0

                    preference_factor = 0.1
                    preference_reward = preference_factor * preference_reward

                    # 保存新的值
                    self.values[cur_key] = preference_reward
                    print(f'Add {self.call_type} values pair: {cur_key} to {preference_reward} \n')

                # 若调用大模型API获取响应失败
                except Exception as e:
                    print("\n\nLLM ERROR\n\n")
                    preference_reward = 0

                return preference_reward

        else:
            print("\n\ncall_type ERROR\n\n")
            return 0

    def generate_goals(self, state_caption):
        try:
            prompt = (
                f"You are a car driver. "
                f"You see a zebra crossing, a {state_caption} traffic light. You are driving. You can go straight, turn right.\n "
                f"Suggest the best actions the driver can take based on what you see. Only suggest valid actions: \n"
                f"- continue driving\n"
                f"- stop and wait\n"
                f"- accelerate\n"
                f"- decelerate\n"
            )
            print("\npr_prompt: ", prompt)

            messages = [{
                    "role": "user",
                    "content": prompt
            }]

            response = requests.post(self.url,
                        json={"model": "llama3.3",  # "deepseek-r1:14b", "llama3.3"
                            "messages": messages, 
                            "stream": True
                            },
                        auth=(self.username, self.password))
            response.raise_for_status()
            output = ""
            for line in response.iter_lines():
                body = json.loads(line)
                if "error" in body:
                    raise Exception(body["error"])
                if body.get("done") is False:
                    message = body.get("message", "")
                    content = message.get("content", "")
                    output += content
                    # the response streams one token at a time, print that as we receive it
                    # print(content, end="", flush=True)

                if body.get("done", False):
                    response = output
                    break
            print("pr_response:", response)

            # 保存到文件
            json_new_data = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": prompt,
                        "output": response}
            file_path = os.path.join(os.getcwd(), 'LLMResults.json')
            if not os.path.exists(file_path):
                json_data = []
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(json_data, file)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            json_data.append(json_new_data)
            with open(file_path, 'w', encoding="utf-8") as file:
                json.dump(json_data, file, indent=2, ensure_ascii=False)

            # 解析LLM的输出 response
            llm_goals_list = []
            lines = response.splitlines()  # 按行分割字符串
            # 遍历每一行，提取以 '-' 开头并且非空的
            for line in lines:
                if line.startswith('-') and line.strip():
                    llm_goals_list.append(line.strip())
            if llm_goals_list == []:
                raise ValueError(f"Response {response} is not a valid response")
            llm_goals_list = [r.strip(' .-\n') for r in llm_goals_list if len(r.strip(' .-\n')) > 0]  # 去除每条建议开头和末尾的空格句点换行符破折号
            llm_goals_list = [goal.lower() for goal in llm_goals_list]  # 转换为小写字母

        # 若调用大模型API获取响应失败
        except Exception as e:
            llm_goals_list = None

        return llm_goals_list

    def calcu_pr(self, action_embedding, llm_goals_embedding):
        # cos_sims = st_utils.pytorch_cos_sim(action_embedding, llm_goals_embedding)[0].detach().cpu().numpy()
        cos_sims = st_utils.pytorch_cos_sim(action_embedding, llm_goals_embedding).detach().cpu().numpy()
        max_cos_sim = np.max(cos_sims)
        test = np.max(max_cos_sim)
        threshold = 0.9
        if max_cos_sim > threshold:
            pr = max_cos_sim  # 设置 pr 为最大值
        else:
            pr = 0  # 否则设置 pr 为 0
            
        return pr



if __name__ == "__main__":
    sentences = ['This is an example sentence', 'Each sentence is converted']

    my_SbertEncoder = SbertEncoder()
    embeddings = my_SbertEncoder.get_sentence_embeddings(sentences)
    # print(embeddings)
    print(embeddings.shape)

    response = """
But it's crucial to do so at a safe speed and be prepared to stop if a pedestrian steps onto the crossing. If pedestrians are on or about to use the zebra crossing, then the appropriate action would be to:\n
- decelerates \n
and prepare to stop if necessary, giving priority to the pedestrians as required by traffic laws in many places. However, without explicit mention of pedestrians being 
present, continuing at a safe speed is the most straightforward choice given the green light and assuming no immediate hazard from pedestrians.\n
            """
    llm_goals_list = []
    lines = response.splitlines()  # 按行分割字符串
    # 遍历每一行，提取以 '-' 开头并且非空的
    for line in lines:
        if line.startswith('-') and line.strip():
            llm_goals_list.append(line.strip())
    llm_goals_list = [r.strip(' .-\n') for r in llm_goals_list if len(r.strip(' .-\n')) > 0]  # 去除每条建议开头和末尾的空格句点换行符破折号
    llm_goals_list = [goal.lower() for goal in llm_goals_list]  # 转换为小写字母
    print(llm_goals_list)





