
from typing import Dict, Any, List
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    # LangChain 0.1+ 兼容性处理
    # 尝试直接从 langchain.agents 导入 AgentExecutor
    from langchain.agents import AgentExecutor
try:
    from langchain.agents import create_tool_calling_agent
except ImportError:
    try:
        from langchain.agents import create_openai_tools_agent as create_tool_calling_agent
    except ImportError:
        pass

try:
    from langchain.prompts import MessagesPlaceholder
except ImportError:
    from langchain_core.prompts import MessagesPlaceholder

from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.tools import BaseTool

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Agent.Tool.predict_toxicity import PredictToxicityTool
from Agent.LLM.llm_interface import LLMInterface

class ToxicityAgent:
    """
    ToxicityAgent: 负责调用预测工具，预测未来水质毒性情况，并进行分析。
    """
    
    def __init__(self, llm_interface: LLMInterface = None):
        self.llm_interface = llm_interface or LLMInterface()
        self.tools = [PredictToxicityTool()]
        self.agent_executor = self._create_agent()
        
    def _create_agent(self) -> AgentExecutor:
        """创建LangChain Agent"""
        # 定义Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的水质毒性预测专家 (ToxicityAgent)。
你的主要任务是根据用户提供的当前水质参数，调用工具预测未来的毒性水平。

你需要：
1. 分析用户输入，提取水质参数（如温度、湿度、氨氮、硝氮、pH等）。
2. 调用 `predict_toxicity` 工具进行预测。
3. 根据工具返回的预测结果，提供专业的分析，包括毒性等级、可能的风险因素。

请确保你的分析基于工具返回的数据。如果工具返回数据中包含风险评估，请务必在回答中强调。
"""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 使用Qwen或OpenAI模型
        # 注意：这里我们需要将自定义的LLMInterface适配为LangChain的ChatModel
        # 为了简化，我们假设LLMInterface.client是OpenAI兼容的客户端，或者我们直接使用ChatOpenAI
        # 由于我们无法直接修改LLMInterface使其完全符合LangChain接口，这里我们构建一个简单的适配
        # 或者直接使用ChatOpenAI连接到相同的API
        
        from langchain_openai import ChatOpenAI
        
        api_key = self.llm_interface.qwen_api_key or self.llm_interface.openai_api_key or "sk-placeholder"
        base_url = self.llm_interface.qwen_api_base or self.llm_interface.openai_api_base
        model_name = self.llm_interface.qwen_model_name or "qwen-max"
        
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0.3
        )
        
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def run(self, input_text: str) -> Dict[str, Any]:
        """
        运行Agent
        
        Args:
            input_text: 包含水质数据的自然语言描述
            
        Returns:
            Dict: 包含 'output' (分析文本) 和可能的中间步骤
        """
        try:
            result = self.agent_executor.invoke({"input": input_text})
            return {
                "analysis": result["output"],
                "status": "success"
            }
        except Exception as e:
            return {
                "analysis": f"毒性预测过程中发生错误: {str(e)}",
                "status": "error"
            }
