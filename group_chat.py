import os
import pandas as pd
from autogen import ConversableAgent, UserProxyAgent, GroupChatManager, GroupChat

summarize_agent = ConversableAgent(
    name="Summarize_agent",
    system_message="Summarize debate between agents and determine whether final sentiment is positive, negative, or neutral",
    human_input_mode="NEVER"
)


risk_seeking_agent = ConversableAgent(
    name="Risk_seeking_Agent",
    system_message="You are a quant analyst. You are trading with risk-seeking strategy",
    llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]},
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "good bye" in msg["content"].lower(),
)

risk_neutral_agent = ConversableAgent(
    name="Risk_neutral_Agent",
    system_message="You are a quant analyst. You are trading with risk-neutral strategy",
    llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]},
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "good bye" in msg["content"].lower(),
)

risk_averse_agent = ConversableAgent(
    name="Risk_averse_Agent",
    system_message="You are a quant analyst. You are trading with risk-averse strategy",
    llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]},
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "good bye" in msg["content"].lower(),
)

risk_seeking_agent.description = "Risk-seeking quant analyst"
risk_neutral_agent.description = "Risk-neutral quant analyst"
risk_averse_agent.description = "Risk-averse quant analyst"

group_chat = GroupChat(
    agents=[risk_seeking_agent, risk_neutral_agent, risk_averse_agent, summarize_agent],
    messages=[],
    max_round=6,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]},
)


def init_debate(message):
    chat_result = summarize_agent.initiate_chat(
        group_chat_manager,
        message=message,
        summary_method="reflection_with_llm",
    )
    return chat_result.summary

def read_file(file_path:str, n_rows:int, usecols:list):
    quant_df = pd.read_csv(file_path, nrows=5, usecols=usecols)
    return quant_df

if __name__ == '__main__':
    file_path = '../dataset/data_combined_label.csv'
    quant_df = read_file(file_path=file_path, n_rows=5, usecols=['작성일', '제목', 'Sentiment Label'])

    conversation_summary_ls = list()
    for _, row in quant_df.iterrows():
        message = 'Analyze the sentiment of this news title "{title}" which is written on {date} and determine whether it is positive, negative, or neutral'
        title = row['제목']
        date = row['작성일']
        message=message.format(title=title, date=date)

        summary = init_debate(message=message)
        conversation_summary_ls.append(summary)