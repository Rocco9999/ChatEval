import os
#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

# always remember to put these lines at the top of your code if you are using clash
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["all_proxy"] = "socks5://127.0.0.1:7890"


import json
import random
from eval_helper.get_evaluation import get_evaluation

from agentverse.agentverse import AgentVerse
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--config", type=str, default="config.yaml")  #cambiare il config in base a quello che si vuole utilizzare
parser.add_argument("--reverse_input", default=False, action="store_true")


args = parser.parse_args()

agentverse, args_data_path, args_output_dir = AgentVerse.from_task("agentverse/tasks/llm_eval/config.yaml") #cambiare il config in base a quello che si vuole utilizzare

print(args)

os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

# uncomment this line if you don't want to overwrite your output_dir
# if os.path.exists(args_output_dir) and len(os.listdir(args_output_dir)) > 1 :
#
#     raise ValueError("the output_dir is not empty, check if is expected.")


with open(args_data_path) as f:
    data = json.load(f)

if "faireval" in args_data_path:
    pair_comparison_output = []

    for conv in data[:1]:
        conversation_id = conv["conversation_id"]
        conversation_results = []
        
        for num, ins in enumerate(conv["exchanges"]):
            print(f"Processing conversation {conversation_id}, exchange {num}: {ins['question']}")


            print(f"================================ conversazione {conversation_id}, istanza {num}====================================")

            # reassign the text to agents, and set final_prompt to null for debate at first round
            for agent_id in range(len(agentverse.agents)):
                agentverse.agents[agent_id].source_text = ins["question"]

                if args.reverse_input:
                    agentverse.agents[agent_id].compared_text_two = ins["response"]["gpt4"]
                else:
                    agentverse.agents[agent_id].compared_text_one = ins["response"]["gpt4"]

                agentverse.agents[agent_id].final_prompt = ""

            agentverse.run()

            evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages, agent_nums=len(agentverse.agents))

            conversation_results.append({
                "exchange_id": num,
                "question": ins["question"],
                "response": {"gpt4": ins["response"]["gpt4"]},
                "evaluation": evaluation
            })

        pair_comparison_output.append({
            "conversation_id": conversation_id,
            "results": conversation_results
        })

        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "pair_comparison_results_conv_beta.json"), "w") as f:
            json.dump(pair_comparison_output, f, indent=4)
    # with open(os.path.join(args_output_dir, "gt_origin_results.json"), "w") as f:
    #     json.dump(gt_origin_output, f, indent=4)

elif "adversarial" in args_data_path:

    pair_comparison_output = []

    for num, ins in enumerate(data):

        print(f"================================instanza {num}====================================")

        # reassign the text to agents, and set final_prompt to null for debate at first round
        for agent_id in range(len(agentverse.agents)):
            agentverse.agents[agent_id].source_text = ins["question"]

            if args.reverse_input:
                agentverse.agents[agent_id].compared_text_one = ins["response"]["output_2"]
                agentverse.agents[agent_id].compared_text_two = ins["response"]["output_1"]
            else:
                agentverse.agents[agent_id].compared_text_one = ins["response"]["output_1"]
                agentverse.agents[agent_id].compared_text_two = ins["response"]["output_2"]

            agentverse.agents[agent_id].final_prompt = ""

        agentverse.run()

        evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages,
                                    agent_nums=len(agentverse.agents))

        pair_comparison_output.append({"question": ins["question"],
                                       "response": {"output_1": ins["response"]["output_1"],
                                                    "output_2": ins["response"]["output_2"]},
                                       "evaluation": evaluation})

        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "pair_comparison_results.json"), "w") as f:
            json.dump(pair_comparison_output, f, indent=4)