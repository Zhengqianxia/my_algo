import torch
import torch.nn as nn
import requests
import os
import json
import datetime
import numpy as np
from enum import Enum, auto
from typing import Literal, Union
import re

class CallType(Enum):
    SEARCH = auto()
    TAIL = auto()

class LLMWrapper_dgfoll:
    def __init__(
        self,
        call_type: Union[Literal["search", "tail"], CallType],
        device=None,
        url="",
        username="",
        password=""
        ):
        """
        Args:
            call_type: Either string ("search"/"tail") or CallType enum
            device: Target device (e.g., "cuda", "cpu", torch.device("cuda:0"))
        """
        # 类型安全处理
        if isinstance(call_type, str):
            call_type = call_type.lower()
            if call_type not in ("search", "tail"):
                raise ValueError(f"Invalid call_type: {call_type}. Must be 'search' or 'tail'")
            self.call_type = CallType[call_type.upper()]
        elif isinstance(call_type, CallType):
            self.call_type = call_type
        else:
            raise TypeError("call_type must be str or CallType")

        # 设备处理
        self.device = device
        self.url = url
        self.username = username
        self.password = password
    
    def load_feedback_data(self):
        """
        参数:
            file_path (str): JSON文件路径
            
        返回:
            历史反馈信息文本
            
        异常:
            会抛出文件不存在异常和JSON解析异常
        """
        try:
            file_path = os.path.join(os.getcwd(), 'feedback.json')
            with open(file_path, 'r', encoding='utf-8') as f:
                feedback_info = json.load(f)
                
                # 验证字段是否存在
                required_fields = [
                    'average_episode_steps',  # 平均回合步长
                    'task_success_rate',  # 任务成功率
                    'average_episode_reward_final_convergence',  # 平均回合奖励最终收敛值
                    'average_episode_reward_max',  # 平均回合奖励最大值
                    'average_episode_reward_variance',  # 平均回合奖励方差
                    'lipschitz_constant'  # Lipschitz 常数
                ]
                
                for field in required_fields:
                    if field not in feedback_info:
                        raise KeyError(f"缺失必要字段: {field}")
                        
                # return feedback_info
                parts = []
                # 按照顺序处理字段
                for key in required_fields:
                    if key in feedback_info:
                        value = feedback_info[key]
                        # 特殊处理浮点型数据（保留两位小数）
                        if isinstance(value, float):
                            parts.append(f"{key}: {value:.2f}")
                        # 处理列表类型数据
                        elif isinstance(value, list):
                            list_str = "[%s]" % ", ".join(map(str, value))
                            parts.append(f"{key}: {list_str}")
                        else:
                            parts.append(f"{key}: {value}")
                
                # 使用逗号连接
                feedback_str = ", ".join(parts)
                return feedback_str
              
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {file_path} 未找到")
        except json.JSONDecodeError:
            raise ValueError("JSON 解析失败，请检查文件格式")


    def extract_function(self, response: str) -> str:
        """
        从大语言模型响应中提取C#函数主体
        
        参数:
            response: 包含C#代码的响应字符串
            
        返回:
            提取到的函数主体代码(包含花括号), 如果找不到返回None
        """
        # 定义更健壮的函数头识别模式
        pattern = r"public\s+override\s+void\s+CollectObservations\s*\([^)]*\)\s*([^{]*\{[^}]*\})"
        
        # 使用re.DOTALL标志使.匹配包括换行符在内的所有字符
        match = re.search(pattern, response, re.DOTALL)
        
        if not match:
            # 尝试更宽松的匹配（可能函数前有注释或修饰符）
            alt_pattern = r"(?://[^\n]*\n)*\s*public\s+override\s+void\s+CollectObservations\s*\(.*?\)\s*(\{.*?\})"
            match = re.search(alt_pattern, response, re.DOTALL)
            if not match:
                return None
        
        # 提取函数主体
        function_body = match.group(1).strip()
        
        # 验证花括号是否平衡
        if function_body.count('{') != function_body.count('}'):
            # 尝试修复不平衡的花括号（简单实现）
            if function_body.count('{') > function_body.count('}'):
                function_body += '}' * (function_body.count('{') - function_body.count('}'))
            else:
                function_body = function_body[:function_body.rfind('}')+1]
        
        # 加上函数头
        function_body = "public override void CollectObservations(VectorSensor sensor)\n" + function_body

        return function_body


    def optimize_state(self):
        if self.call_type == CallType.SEARCH:  # 目标探索定位场景
            task_description = "Optimize the CollectObservations function of this Unity RL-agent car for the target search task. Based on the given 'current function' and 'feedback', provide an improved version in C#. Keep the same function header and output the full function code.\n"
            
        elif self.call_type == CallType.TAIL:   # 动态目标控距跟踪场景
            task_description = "Optimize the CollectObservations function of this Unity RL-agent car for the distance-controlled tracking task. Based on the given 'current function' and 'feedback', provide an improved version in C#. Keep the same function header and output the full function code.\n"

        else:
            print("\n\ncall_type ERROR\n\n")
            return 0
            
        # 当前状态表示函数
        current_func = """
public override void CollectObservations(VectorSensor sensor)
{
    // 智能体的方向
    sensor.AddObservation(this.transform.forward.x);
    sensor.AddObservation(this.transform.forward.z);

    // 智能体的速度
    sensor.AddObservation(m_AgentRb.velocity.x);
    sensor.AddObservation(m_AgentRb.velocity.z);

    // 智能体的坐标
    sensor.AddObservation(this.transform.position.x);
    sensor.AddObservation(this.transform.position.z);

    // 目标的坐标
    sensor.AddObservation(Target.transform.position.x);
    sensor.AddObservation(Target.transform.position.z);
}
"""

        # 历史反馈信息文本
        feedback_str = self.load_feedback_data()
        # print("\nfeedback_str: ", feedback_str)

        # 调用 LLMs 优化状态表示函数
        try:
            prompt = (
                f"{task_description}"
                f"current function:\n"
                f"{current_func}"
                f"feedback:\n"
                f"{feedback_str}"
            )
            print("\nprompt: ", prompt)

            messages = [{
                    "role": "user",
                    "content": prompt
            }]

            response = requests.post(self.url,
                        json={"model": "deepseek-r1:14b",  # llama3.3","deepseek-r1:14b"
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
            print("response:", response)

            # 保存到文件
            json_new_data = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": prompt,
                        "output": response}
            file_path = os.path.join(os.getcwd(), 'LLMsResults.json')
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
            optimized_state = self.extract_function(response)
            if optimized_state:
                print("成功提取函数主体：")
                print("optimized_state:\n", optimized_state)
            else:
                print("未找到目标函数")

        # 若调用大模型API获取响应失败
        except Exception as e:
            print("\n\nLLMs ERROR\n\n")
            optimized_state = ""

        return optimized_state
    

if __name__ == "__main__":
    dgfoll_llm = LLMWrapper_dgfoll(call_type="search")
    dgfoll_llm.optimize_state()

