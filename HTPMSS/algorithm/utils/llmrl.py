import torch
import torch.nn as nn
import requests
import os
import json
import datetime
import numpy as np
from enum import Enum, auto
from typing import Literal, Union
from my_utils import execute_skill
import re


class LLMWrapper_htpmss:
    def __init__(
        self,
        device=None,
        url="",
        username="",
        password=""
        ):

        # 设备处理
        self.device = device
        self.url = url
        self.username = username
        self.password = password

        # 定义智能体的基础技能集
        self.skills = [
            "patrol",  # 街道巡逻"
            "search",  # 探索定位"
            "track",  #  "控距跟踪"
            "destroy",  #  "快速打击"
        ]
    

    def get_skills_description(self):
        """
        Generates a descriptive string of the agent's basic skill set.

        Returns:
            str: A string describing the agent's skills
        """

        skills_str = "、".join(self.skills)
        # skills_str = "智能体目前已学会的基础技能有：" + skills_str
        # skills_str = 
        # "The agent has currently mastered the following basic skills: " + skills_str
        return skills_str


    def parse_response(self, response, call_type):
        """
        Parses the LLM response to extract skill relevance scores and execution order scores.

        Args:
            response (str): The response text from the large language model
            call_type (str): The calling type, either "planning" or "schedule"
                - "planning": Extracts both relevance and order scores
                - "schedule": Extracts only order scores

        Returns:
            tuple: (skill_relevance, skill_order) two dictionaries
                When call_type is "schedule", skill_relevance returns an empty dict
        """

        skill_relevance = {}
        skill_order = {}
        
        # 获取基础技能列表
        target_skills = self.skills

        # 统一转换为小写方便搜索基础技能名称
        response_lower = response.lower()

        # 使用regex定义评级模式，方便提取
        order_pattern = re.compile(
            r'order:\s*(first|second|third|fourth|irrelevant)',
            re.IGNORECASE
            )
        relevance_pattern = re.compile(
            r'relevance:\s*(very low|low|medium|high|very high)',
            re.IGNORECASE
            )
        
        for skill in target_skills:
            # 搜索技能相关的段落，直到遇到空行或文本结束
            skill_matches = re.finditer(
                rf'\b{skill}\b.*?(?=\n\n|\Z)',  # 正则表达式模式
                response,                       # 要搜索的文本
                re.IGNORECASE | re.DOTALL       # 匹配标志
            )
            
            for match in skill_matches:
                skill_section = match.group()

                # 提取执行优先级评分order
                order_match = order_pattern.search(skill_section)
                if order_match:
                    skill_order[skill] = order_match.group(1).title()
                
                # 提取技能相关性评分Relevance（仅在任务规划模式下）
                if call_type == "planning":
                    relevance_match = relevance_pattern.search(skill_section)
                    if relevance_match:
                        skill_relevance[skill] = relevance_match.group(1).title()
        
        return skill_relevance, skill_order


    def skill_filter(self, skill_relevance):
        """
        Filter out skills with relevance higher than 'Low'
        
        Args:
            skill_relevance (dict): Skill relevance dictionary, 
            
        Returns:
            list: List of skill names with relevance higher than 'Low'
        """
        # Define rating levels and their weights
        rating_weights = {
            'Very Low': 0,
            'Low': 0.25,
            'Medium': 0.5,
            'High': 0.75,
            'Very High': 1
        }
        
        # Filter skills with relevance higher than 'Low'
        high_relevance_skills = [
            skill for skill, relevance in skill_relevance.items()
            if rating_weights.get(relevance.title(), 0) > rating_weights['Low']
        ]
        
        return high_relevance_skills


    def sort_skills(self, candidate_skills, skill_order):
        """
        Sorts candidate skills based on their priority order in skill_order dictionary.
        
        Args:
            candidate_skills (list): List of skill names to be sorted
            skill_order (dict): Dictionary mapping skills to their priority 
                            ('First', 'Second', 'Third', 'Fourth', or 'Irrelevant')
        
        Returns:
            list: Sorted list of skills (execution_order)
        """
        # Define the priority order mapping
        priority_order = {
            'First': 1,
            'Second': 2,
            'Third': 3,
            'Fourth': 4,
            'Irrelevant': 5
        }
        
        # Sort skills based on their priority
        execution_order = sorted(
            candidate_skills,
            key=lambda skill: priority_order.get(
                skill_order.get(skill, 'Irrelevant').title(),  # Default to 'Irrelevant' if skill not found
                5  # Default value if priority not found
            )
        )
        
        return execution_order


    def task_planning(self, command):
        
        task_description = """
Analyze the commander's 'command' and agent's 'basic skills' to:
Score each skill's relevance (Very Low/Low/Medium/High/Very High)
Determine the execution order (first/second/third/fourth/irrelevant) (since this is a complex task requiring at least two skills)

Please format your response as:
[Skill]:
Relevance: [Level]
Order: [Level]
(Repeat for all skills)
"""

        # 检查command是否包含自然语言高级指令
        if not isinstance(command, str):
            raise ValueError("The command must be of string type.")
        if not command.strip():
            raise ValueError("The command cannot be an empty string.")

        # 获取智能体基础技能集的描述
        skills_description = self.get_skills_description()

        # 调用 LLMs 进行高层任务规划
        try:
            prompt = (
                f"{task_description}\n"
                f"command:\n"
                f"{command}\n\n"
                f"basic skills:\n"
                f"{skills_description}"
            )
            print("\nprompt: ", prompt)

            messages = [{
                    "role": "user",
                    "content": prompt
            }]

            response = requests.post(self.url,
                        json={"model": "deepseek-r1:14b",
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

            # 解析LLM的输出 response，得到两个技能评分字典
            skill_relevance, skill_order = self.parse_response(response, call_type="planning")
            print("\nskill_relevance:\n", skill_relevance)
            # Output: {'patrol': 'Medium', 'search': 'High', 'track': 'Very high', 'destroy': 'Low'}
            print("\nskill_order:\n", skill_order)
            # Output: {'patrol': 'First', 'search': 'Second', 'track': 'Third', 'destroy': 'Irrelevant'}

            # 使用技能过滤器，得到候选技能池
            candidate_skills = self.skill_filter(skill_relevance)
            print("\ncandidate_skills:\n", candidate_skills)
            # Output: ['patrol', 'search', 'track']

            # 再根据优先级评分，得到初始技能执行序列
            sorted_skills = self.sort_skills(candidate_skills, skill_order)
            print("\nExecution order:\n", sorted_skills) 
            # Output: ['patrol', 'search', 'track']


        # 调用大模型失败
        except Exception as e:
            error_msg = f"""
            LLMs Error Type: {type(e).__name__}
            Error Details: {str(e)}
            """
            print(error_msg)
            sorted_skills = ""

        return sorted_skills


    def skill_schedule(self, command, sorted_skills):
        # 获取未执行完的基础技能序列
        uncompleted_skills = sorted_skills

        # 初始化技能完成度矩阵
        skill_status_matrix = {skill: "0" for skill in uncompleted_skills}

        while uncompleted_skills:
            # 执行 基础技能序列 中的第一个技能
            current_skill = uncompleted_skills[0]
            execute_info = execute_skill(current_skill)
            current_skill_state = execute_info["complete_state"]

            # 检查技能完成状态
            if current_skill_state == "1":
                # 更新 技能完成度矩阵 中 对应技能的状态
                if current_skill in skill_status_matrix:
                    skill_status_matrix[current_skill] = "1"

                # 从 未执行完的基础技能序列 移除 当前技能
                uncompleted_skills.pop(0)
                print(f"Completed: {current_skill} | Remaining: {len(uncompleted_skills)}")

                # 重评估执行优先级
                # 获取技能完成度矩阵的文本描述
                skill_status_description = []
                for skill, status in skill_status_matrix.items():
                    description = "completed" if status == "1" else "uncompleted"
                    skill_status_description.append(f"{skill}: {description}")
                skill_status_description = ", ".join(skill_status_description)

                # 场景任务介绍
                task_description = """
Analyze the commander's 'command', agent's 'basic skills' and 'skill complete status' to:
Score each skill's relevance (Very Low/Low/Medium/High/Very High)
Determine the execution order (first/second/third/fourth/irrelevant) (since this is a complex task requiring at least two skills)

Please format your response as:
[Skill]:
Relevance: [Level]
Order: [Level]
(Repeat for all skills)
"""

                # 获取智能体基础技能集的描述
                skills_description = self.get_skills_description()

                # 调用 LLMs 重评估执行优先级
                try:
                    prompt = (
                        f"{task_description}\n"
                        f"skill complete status:\n"
                        f"{skill_status_description}\n\n"
                        f"command:\n"
                        f"{command}\n\n"
                        f"basic skills:\n"
                        f"{skills_description}"
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

                    # 解析LLM的输出 response，得到一个优先级评分字典
                    skill_relevance, skill_order = self.parse_response(response, call_type="schedule")
                    print("\nskill_order:\n", skill_order)
                    # Output: {'patrol': 'Third', 'search': 'First', 'track': 'Second', 'destroy': 'Irrelevant'}

                    # 调整技能执行序列
                    candidate_skills = uncompleted_skills
                    sorted_skills = self.sort_skills(candidate_skills, skill_order)
                    uncompleted_skills = sorted_skills
                    print("\nuncompleted_skills:\n", uncompleted_skills) 
                    # Output: ['search', 'track']


                # 调用大模型失败
                except Exception as e:
                    error_msg = f"""
                    LLMs Error Type: {type(e).__name__}
                    Error Details: {str(e)}
                    """
                    print(error_msg)
                    sorted_skills = ""

        print("All skills completed successfully!")


if __name__ == "__main__":
    # "执行区域内重点目标监视警戒任务","清剿区域内威胁目标"
    command = "Conduct surveillance on high-priority targets within the operational area."  
    htpmss_llm = LLMWrapper_htpmss()
    sorted_skills = htpmss_llm.task_planning(command)
    htpmss_llm.skill_schedule(command, sorted_skills)




