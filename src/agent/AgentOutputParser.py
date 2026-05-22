from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseOutputParser


class AgentOutputParser(BaseOutputParser[str]):
    def parse(self, output):
        if isinstance(output, dict):
            if "messages" in output:
                return output["messages"][-1].content
            if "output" in output:
                return output["output"]
            if "model" in output and "messages" in output.get("model", {}):
                return output["model"]["messages"][-1].content
        if hasattr(output, "content"):
            return output.content
        print("原始输出")
        return str(output)



if __name__ == '__main__':
    mode_str = {'model': {'messages': [AIMessage(content='我找到了一条与积分计算相关的笔记，内容如下：\n\n### 积分计算：cos\xb3x的积分\n- **标签**: 积分, 三角函数, 数学, 计算, 微积分\n- **内容**:\n  \\[\n  \\int \\cos^3 x \\, dx = \\int \\cos^2 x \\cdot \\cos x \\, dx = \\int (1 - \\sin^2 x) \\, d(\\sin x) \n  = \\cos x \\cdot \\sin x - \\int \\sin x \\, d(\\cos^2 x) \n  = \\cos^2 x \\cdot \\sin x + \\int \\sin x \\cdot 2\\cos x \\cdot \\sin x \\, dx \n  = \\cos^2 x \\cdot \\sin x + \\int 2\\sin^2 x \\, d(\\sin x) \n  = \\cos^2 x \\cdot \\sin x + \\frac{2}{3}\\sin^3 x + C\n  \\]\n\n如果你想查看这条笔记的详细内容或需要更多信息，请告诉我！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 252, 'prompt_tokens': 1596, 'total_tokens': 1848, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 1280}, 'latency_checkpoint': {'engine_tbt_ms': 13, 'engine_ttft_ms': 40, 'engine_ttlt_ms': 3198, 'pre_inference_ms': 231, 'service_tbt_ms': 12, 'service_ttft_ms': 599, 'service_ttlt_ms': 3673, 'total_duration_ms': 3451, 'user_visible_ttft_ms': 368}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_eb37e061ec', 'id': 'chatcmpl-Df4BxAdnM9wGmTZbKc8ObNq6L58Bw', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019e2195-ea65-77d3-8983-361c4c3d0e29-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 1596, 'output_tokens': 252, 'total_tokens': 1848, 'input_token_details': {'audio': 0, 'cache_read': 1280}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}

    print(AgentOutputParser().parse(mode_str))
