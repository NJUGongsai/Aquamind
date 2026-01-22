
import sys
import os
import re
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Agent.Agent.ToxicityAgent import ToxicityAgent
from Agent.Agent.ControlAgent import ControlAgent
from Agent.LLM.llm_interface import LLMInterface

class AquamindOrchestrator:
    """
    AquamindOrchestrator: 顶层架构，污水处理厂运营工程师。
    负责协调 ToxicityAgent 和 ControlAgent，生成最终报告。
    """
    
    def __init__(self):
        self.llm_interface = LLMInterface()
        self.toxicity_agent = ToxicityAgent(self.llm_interface)
        self.control_agent = ControlAgent(self.llm_interface)
        
    def _parse_input(self, user_input: str) -> Dict[str, str]:
        """
        简单的输入解析器，提取关键信息。
        在更复杂的系统中，可以使用专门的Agent或提取链来完成。
        """
        # 默认值
        params = {
            "treatment_process": "未知工艺",
            "time_frame": "24小时"
        }
        
        # 尝试提取工艺
        # 假设格式如 "运行工艺是AAO" 或 "$treatment"
        treatment_match = re.search(r"工艺是\s*([a-zA-Z0-9\u4e00-\u9fa5]+)", user_input)
        if treatment_match:
            params["treatment_process"] = treatment_match.group(1)
        else:
            # 简单的关键词匹配
            known_processes = ["AAO", "A2O", "SBR", "MBR", "氧化沟", "活性污泥法"]
            for process in known_processes:
                if process in user_input.upper():
                    params["treatment_process"] = process
                    break
                    
        # 尝试提取时间
        time_match = re.search(r"未来\s*(\d+\s*(小时|天|h|day))", user_input)
        if time_match:
            params["time_frame"] = time_match.group(1)
            
        return params

    def run(self, user_input: str) -> str:
        """
        执行主流程
        
        Args:
            user_input: 用户输入的自然语言请求
            
        Returns:
            str: 最终生成的报告路径或内容摘要
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] AquamindOrchestrator: 收到请求，正在分析...")
        
        # 1. 分析输入
        parsed_params = self._parse_input(user_input)
        treatment_process = parsed_params["treatment_process"]
        time_frame = parsed_params["time_frame"]
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 识别关键信息: 工艺={treatment_process}, 预测时间={time_frame}")
        
        # 2. 调用 ToxicityAgent
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在调度 ToxicityAgent 进行毒性预测...")
        toxicity_result = self.toxicity_agent.run(user_input)
        
        if toxicity_result["status"] != "success":
            return f"流程中断：毒性预测失败 - {toxicity_result['analysis']}"
            
        toxicity_analysis = toxicity_result["analysis"]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ToxicityAgent 完成预测。")
        
        # 3. 调用 ControlAgent
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在调度 ControlAgent 生成工艺建议...")
        control_result = self.control_agent.run(
            toxicity_analysis=toxicity_analysis,
            treatment_process=treatment_process,
            time_frame=time_frame
        )
        
        if control_result["status"] != "success":
            return f"流程中断：控制建议生成失败 - {control_result['suggestion']}"
            
        control_suggestion = control_result["suggestion"]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ControlAgent 完成建议生成。")
        
        # 4. 生成报告
        report = self._generate_report(user_input, parsed_params, toxicity_analysis, control_suggestion)
        
        return report

    def _generate_report(self, user_input: str, params: Dict[str, str], toxicity_analysis: str, control_suggestion: str) -> str:
        """生成并保存报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_content = f"""# Aquamind Systems 智能预测与控制报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 用户请求摘要
- **原始输入**: {user_input}
- **识别工艺**: {params['treatment_process']}
- **预测时效**: {params['time_frame']}

## 2. 水质毒性预测分析 (by ToxicityAgent)
{toxicity_analysis}

## 3. 工艺调整建议 (by ControlAgent)
{control_suggestion}

---
*Aquamind Systems - 您的智慧水务专家*
"""
        
        # 保存报告
        report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Report")
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        report_path = os.path.join(report_dir, f"Report_{timestamp}.md")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 报告已生成: {report_path}")
        return f"执行完成。报告已保存至: {report_path}\n\n报告摘要:\n{report_content[:500]}..."

if __name__ == "__main__":
    # 简单测试
    orchestrator = AquamindOrchestrator()
    sample_input = "你好Aquamind，我目前的运行工艺是AAO，目前水质毒性数据是氨氮25mg/L，温度20度，毒性是10，请你帮我预测下未来24小时后的毒性数据并给出调整方案"
    print(orchestrator.run(sample_input))
