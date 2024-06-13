from nlfs.llm_agents.gpt import GPT_3PT5_TURBO
from nlfs.llm_agents.gpt import GPT_4
from nlfs.llm_agents.gpt import GPT_4_TURBO
from nlfs.llm_agents.gpt import GPT_4O
from nlfs.llm_agents.claude import Claude
from nlfs.llm_agents.vllm_agent import VLLMAgent

@staticmethod
def get_llm_agent(model_name, temperature=0.1):

    if model_name == "gpt-3.5-turbo":
        return GPT_3PT5_TURBO(temperature=temperature)
    elif model_name == "gpt-4":
        return GPT_4(temperature=temperature)
    elif model_name == "gpt-4-turbo":
        return GPT_4_TURBO(temperature=temperature)
    elif model_name == "gpt-4o":
        return GPT_4O(temperature=temperature)
    elif model_name == "claude":
        return Claude(temperature=temperature)    
    elif model_name == "mistral":
        return VLLMAgent("mistralai/Mistral-7B-Instruct-v0.2")
    elif model_name == "llama-3-8b":
        return VLLMAgent("meta-llama/Meta-Llama-3-8B-Instruct")
    elif model_name == "phi-3":
        return VLLMAgent("microsoft/Phi-3-medium-4k-instruct")
    else:
        raise Exception("Invalid model name")