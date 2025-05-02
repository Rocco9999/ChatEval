import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr, kendalltau
import matplotlib.pyplot as plt

valutazioni_per_conversazione =np.load("valutazioni.npy", allow_pickle=True).item()

human_evaluation = np.load("human_evaluation.npy", allow_pickle=True).item()

print("Conversazioni caricate:", len(valutazioni_per_conversazione))
print("Valutazioni medie caricate:", len(human_evaluation))

def calc_and_save_mean_evaluation():

    mean_per_conversation = {}

    for conv_id, valutazioni in valutazioni_per_conversazione.items():
        agent_sums = [0.0] * 4
        agent_counts = [0] * 4

        for single_evaluation in valutazioni:
            for id in range(4):
                agent_sums[id] += single_evaluation[id]
                agent_counts[id] += 1

        conversation_agent_mean = [agent_sums[i] / agent_counts[i] for i in range(4)]
        total_mean = 0
        for agent_mean in conversation_agent_mean:
            total_mean += agent_mean
        mean_per_conversation[conv_id] = total_mean / 4.0
        total_mean = 0

    print("Medie per ogni agente per conversazione:")
    for k, v in mean_per_conversation.items():
        print(f"ID {k}: {v}")

    np.save("mean_agent_evaluation.npy", mean_per_conversation)

    return mean_per_conversation


if __name__ == "__main__":
    # mean_chateval= calc_and_save_mean_evaluation()
    mean_chateval = np.load("mean_agent_evaluation.npy", allow_pickle=True).item()

    common_keys = set(mean_chateval.keys()) & set(human_evaluation.keys())

    chat_eval_vals = [mean_chateval[k] for k in common_keys]
    human_eval_vals = [human_evaluation[k] for k in common_keys]
    # print("chat eval evaluation: ", chat_eval_vals)
    # print("human evaluation: ", human_eval_vals)

    # 1. Cohen's Kappa (serve arrotondare i valori perché è per classi discrete)
    chat_eval_rounded = [round(v) for v in chat_eval_vals]
    human_eval_rounded = [round(v) for v in human_eval_vals]
    kappa = cohen_kappa_score(chat_eval_rounded, human_eval_rounded)

    # 2. Spearman
    spearman_corr, spearman_p = spearmanr(chat_eval_vals, human_eval_vals)

    # 3. Pearson
    pearson_corr, pearson_p = pearsonr(chat_eval_vals, human_eval_vals)

    # 4. Kendall Tau
    kendall_corr, kendall_p = kendalltau(chat_eval_vals, human_eval_vals)

    # Stampa dei risultati
    print(f"Cohen's Kappa: {kappa:.3f}")
    print(f"Spearman's correlation: {spearman_corr:.3f}, P-value: {spearman_p:.4f}")
    print(f"Pearson's correlation: {pearson_corr:.3f}, P-value: {pearson_p:.4f}")
    print(f"Kendall-Tau correlation: {kendall_corr:.3f}, P-value: {kendall_p:.4f}")


    plt.figure(figsize=(8, 6))
    plt.scatter(human_eval_vals, chat_eval_vals, alpha=0.6)
    plt.xlabel("Valutazione Umana")
    plt.ylabel("Valutazione ChatEval")
    plt.title("Confronto tra Valutazioni Umane e ChatEval")
    plt.grid(True)
    plt.show()

