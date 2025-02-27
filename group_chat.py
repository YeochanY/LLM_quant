import os
import pandas as pd
from autogen import ConversableAgent, UserProxyAgent, GroupChatManager, GroupChat

# Define a Moderator Agent to enforce debate structure
moderator_agent = ConversableAgent(
    name="Moderator_Agent",
    system_message=(
        "You are a neutral moderator overseeing a debate among quant analysts. "
        "Ensure a structured discussion, challenge weak arguments, and guide the debate to a final sentiment consensus."
    ),
    human_input_mode="NEVER",
)

# Define sentiment summarization agent
summarize_agent = ConversableAgent(
    name="Summarize_Agent",
    system_message="Summarize the debate between agents and determine the final sentiment: positive, negative, or neutral. "
                   "Consider argument strengths and market conditions.",
    human_input_mode="NEVER",
)

# Define risk profile agents with specific trading styles
risk_profiles = {
    "Risk-Seeking": "You are a quant analyst trading aggressively with high risk.",
    "Risk-Neutral": "You are a quant analyst balancing risk and reward neutrally.",
    "Risk-Averse": "You are a quant analyst trading cautiously to minimize risk."
}

agents = []
for profile, description in risk_profiles.items():
    agent = ConversableAgent(
        name=f"{profile.replace('-', '_')}_Agent",
        system_message=description,
        llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]},
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "good bye" in msg["content"].lower(),
    )
    agent.description = description
    agents.append(agent)

# Define GroupChat with structured interactions
group_chat = GroupChat(
    agents=[moderator_agent] + agents + [summarize_agent],  # Moderator guides the discussion
    messages=[],
    max_round=8,  # Increased rounds for more structured debate
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]},
)

def init_debate(message):
    """
    Initiates a debate among agents and returns the summarized sentiment.
    """
    chat_result = summarize_agent.initiate_chat(
        group_chat_manager,
        message=message,
        summary_method="reflection_with_llm",  # Reflects on past arguments before summarizing
    )
    return chat_result.summary

def read_file(file_path: str, n_rows: int, usecols: list):
    """
    Reads CSV file and extracts required columns.
    """
    return pd.read_csv(file_path, nrows=n_rows, usecols=usecols)

if __name__ == '__main__':
    file_path = '../dataset/data_combined_label.csv'
    quant_df = read_file(file_path=file_path, n_rows=5, usecols=['작성일', '제목', 'Sentiment Label'])

    conversation_summary_ls = []

    for _, row in quant_df.iterrows():
        message_template = (
            'Analyze the sentiment of this news title "{title}" written on {date}. '
            'Each agent should debate based on their risk strategy and predict if sentiment is positive, negative, or neutral.'
        )
        message = message_template.format(title=row['제목'], date=row['작성일'])

        summary = init_debate(message=message)
        conversation_summary_ls.append(summary)

    # Store conversation summaries for further analysis
    pd.DataFrame(conversation_summary_ls, columns=["Sentiment Summary"]).to_csv("sentiment_results.csv", index=False)
