
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Agent.LLM.llm_interface import LLMInterface

class ControlAgent:
    """
    ControlAgent: 污水处理厂工艺工程师。
    负责接收ToxicityAgent的预测结果和分析文本，针对特定工艺，给出权威的调整建议。
    """
    
    def __init__(self, llm_interface: LLMInterface = None):
        self.llm_interface = llm_interface or LLMInterface()
        self.chain = self._create_chain()
        
    def _create_chain(self):
        """创建处理建议生成链"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一名经验丰富的污水处理厂工艺工程师 (ControlAgent)。
你熟悉各种污水处理工艺（如AAO, SBR, 氧化沟, MBR等）及其运行参数。

你的任务是：
1. 接收 `ToxicityAgent` 提供的毒性预测结果和分析。
2. 结合用户提供的当前运行工艺 (`treatment_process`)。
3. 给出具体的、权威的工艺调整建议，以应对预测到的毒性风险或维持良好运行。

如果预测毒性较高：
- 提出具体的应急措施（如调整回流比、增加曝气、投加药剂、调整进水量等）。
- 针对特定工艺（如AAO）给出针对性建议。

如果预测毒性较低：
- 建议如何优化运行以节能降耗。

请保持输出专业、条理清晰。
"""),
            ("human", """
运行工艺: {treatment_process}
预测时间范围: {time_frame}

毒性预测与分析报告:
{toxicity_analysis}

请给出调整建议：
""")
        ])
        
        api_key = self.llm_interface.qwen_api_key or self.llm_interface.openai_api_key or "sk-placeholder"
        base_url = self.llm_interface.qwen_api_base or self.llm_interface.openai_api_base
        model_name = self.llm_interface.qwen_model_name or "qwen-max"
        
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0.5
        )
        
        return prompt | llm | StrOutputParser()

    def run(self, toxicity_analysis: str, treatment_process: str, time_frame: str = "24小时") -> Dict[str, Any]:
        """
        生成控制建议
        
        Args:
            toxicity_analysis: 毒性预测分析文本
            treatment_process: 运行工艺名称
            time_frame: 预测时间范围
            
        Returns:
            Dict: 包含 'suggestion' (建议文本)
        """
        try:
            suggestion = self.chain.invoke({
                "toxicity_analysis": toxicity_analysis,
                "treatment_process": treatment_process,
                "time_frame": time_frame
            })
            
            return {
                "suggestion": suggestion,
                "status": "success"
            }
        except Exception as e:
            return {
                "suggestion": f"生成控制建议时发生错误: {str(e)}",
                "status": "error"
            }
