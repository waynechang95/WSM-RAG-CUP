from ollama import Client
from pathlib import Path
import yaml


def load_ollama_config() -> dict:
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError("No configuration file found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


def generate_answer_en(query, context_chunks):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompt = f"""You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Here are some examples of the answer: \
    1. Question: "query_type": "Irrelevant Unsolvable Question", "Question": "What risk management measure was updated in March 2021?", "Context": "doc_ids": [50] \
    Answer: "Unable to answer." \
    \
    2. Question: "query_type": "Factual Question", "Question": "When was Green Fields Agriculture Ltd. established?", "Context": "doc_ids": [44] \
    Answer: "Green Fields Agriculture Ltd. was established on April 1, 2005." \
    \
    3. Question: "query_type": "Summary Question", "Question": "Based on the outline, summarize the corporate governance improvements made by Green Fields Agriculture Ltd. in 2021.", "Context": "doc_ids": [44] \
    Answer: "In 2021, Green Fields Agriculture Ltd. revised its corporate governance policy to enhance transparency and accountability, strengthened the Board and Supervisory Board through training, increased independent directors, optimized decision-making, and improved information disclosure to stakeholders." \
    \
    4. Question: "query_type": "Multi-hop Reasoning Question", "Question": "According to the judgment of Bayside, Roseville, Court, how many instances of embezzlement did V. Martin commit?", "Context": "doc_ids": [112] \
    Answer: "V. Martin committed four instances of embezzlement: transferring $150,000 from the disaster relief fund, withdrawing $200,000 from the emergency rescue fund, diverting $250,000 from the flood control fund, and failing to return $100,000 from the rural development fund." \
    \
    5. Question: "query_type": "Multi-document Information Integration Question", "Question": "According to the judgment of Quarryville, Yorktown, Court and Hamilton, Harrison, Court, what are the occupations of the defendants J. Thompson and M. Ward?", "Context": "doc_ids": [110, 127] \
    Answer: "J. Thompson is a Finance Manager at Yorktown Municipal Office, and M. Ward is a Nurse." \
    \
    6. Question: "query_type": "Multi-document Time Sequence Question", "Question": "According to the judgment of Quarryville, Yorktown, Court and Hamilton, Harrison, Court, whose judgment time is earlier, J. Thompson or M. Ward?", "Context": "doc_ids": [110, 127] \
    Answer: "M. Ward's judgment time is earlier, dated 30th April, 2023, while J. Thompson's judgment is dated 10th October, 2023." \
    \
    7. Question: "query_type": "Multi-document Comparison Question", "Question": "According to the judgment of Upton, Georgetown, Court and Trenton, Vandalia, Court, whose sentencing time is longer, F. Williams or G. Torres?", "Context": "doc_ids": [136, 122] \
    Answer: "F. Williams' sentencing time is longer, with five years of fixed-term imprisonment, compared to G. Torres' three years of fixed-term imprisonment with a five-year suspension." \
    \
    Anser the queries accordding to the examples and the different of query types(answering each query type with different answer formats).
    Use three sentences maximum and keep the answer concise.\n\nQuestion: {query} \nContext: {context} \nAnswer:\n"""
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], prompt=prompt)
    return response["response"]

def generate_answer_zh(query, context_chunks):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompt = f"""你是一個專門處理問答任務的助手。 \
    使用以下檢索到的上下文來回答問題。 \
    請根據不同的問題類型，使用不同的回答格式。 \
    如果你不知道答案，就直接說「無法回答」。 \
    以下是一些回答的範例： \
    1. Question: "query_type": "无关无解问", "Question": "绿源环保有限公司在2017年12月通过股权融资募集了多少资金？", "Context": "doc_ids": [5] \
    Answer: "無法回答。" \
    \
    2. Question: "query_type": "事实性问题", "Question": "绿源环保有限公司未来社会责任战略的年度预算是多少？", "Context": "doc_ids": [5] \
    Answer: "绿源环保有限公司未来社会责任战略的年度预算为300,000元。" \
    \
    3. Question: "query_type": "总结性问题", "Question": "根据荔枝市钢城区人民法院的判决书，总结被告人尹某的犯罪事实。", "Context": "doc_ids": [82] \
    Answer: "尹某于2022年5月20日闯红灯撞伤行人张某后逃逸；6月10日更换车牌并购买假车牌；6月12日再次发生交通肇事，撞伤刘某后再次逃逸，造成两人受伤。" \
    \
    4. Question: "query_type": "多跳推理问题", "Question": "根据荔枝市钢城区人民法院的判决书，尹某一共有几次犯罪行为？", "Context": "doc_ids": [82] \
    Answer: "尹某一共有三次犯罪行为：2022年5月20日交通肇事逃逸、2022年6月10日更换并购买假车牌、2022年6月12日再次交通肇事逃逸。" \
    \
    5. Question: "query_type": "多文档信息整合问题", "Question": "华夏娱乐有限公司和绿源环保有限公司在2017年分别在哪个月进行了股东大会决议？", "Context": "doc_ids": [0, 5] \
    Answer: "两家公司均在2017年11月进行了股东大会决议。" \
    \
    6. Question: "query_type": "多文档时间序列问题", "Question": "比较建业集团有限公司和滨江消费品有限公司分别进行的董事会变更时间，哪家公司变更时间更早？", "Context": "doc_ids": [11, 19] \
    Answer: "建业集团有限公司的董事会变更时间更早，发生在2020年5月；滨江消费品有限公司的董事会变更发生在2021年3月。" \
    \
    7. Question: "query_type": "多文档对比问题", "Question": "比较2017年华夏娱乐有限公司和2018年顶级购物中心的净利润，哪家公司的净利润更高？", "Context": "doc_ids": [0, 2] \
    Answer: "2017年华夏娱乐有限公司的净利润更高，为8500万元；而2018年顶级购物中心的净利润为800万元。" \
    \
    根據不同的 query_type，請按照範例回答問題。每種 query_type 使用不同的回答格式。
    回答不得超過三句，並保持簡潔。\n\nQuestion: {query} \nContext: {context} \nAnswer:\n"""
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], prompt=prompt)
    return response["response"]


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Generated Answer:", answer)