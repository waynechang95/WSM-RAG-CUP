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


def generate_answer(query, context_chunks):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompt = f"""You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Here are some examples of the answer: \
1. Question: "query_type": "无关无解问", "Question": "绿源环保有限公司在2017年12月通过股权融资募集了多少资金？", "Context": {"doc_ids": [5]} \
    Answer: "无法回答。" \
2. Question: "query_type": "事实性问题", "Question": "绿源环保有限公司未来社会责任战略的年度预算是多少？", "Context": {"doc_ids": [5]} \
    Answer: "绿源环保有限公司未来社会责任战略的年度预算为300,000元。" \
3. Question: "query_type": "总结性问题", "Question": "根据荔枝市钢城区人民法院的判决书，总结被告人尹某的犯罪事实。", "Context": {"doc_ids": [82]} \
    Answer: "尹某于2022年5月20日闯红灯撞伤行人张某后逃逸；6月10日更换车牌并购买假车牌；6月12日再次发生交通肇事，撞伤刘某后再次逃逸，造成两人受伤。" \
4. Question: "query_type": "总结性问题", "Question": "结合绿源环保有限公司2017年的财务报告，总结其在2017年发生的财务事件。", "Context": {"doc_ids": [5]} \
    Answer: "根据绿源环保有限公司2017年的财务报告，2017年公司发生了多起重要的财务事件。首先，公司完成了资产重组计划，包括旧工厂升级改造、新建绿色能源科技园项目等，增强了公司的经营能力和竞争力；其次，公司成功募集了5亿元资金，进行了债务重组，减少了1亿元负债，改善了财务状况；此外，公司还进行了重大投资和股权收购，分别投资了2亿元于D项目，并收购了绿能科技股份有限公司45%的股权，进一步拓宽了业务领域和市场份额；最后，公司完成了对清新环保公司的资产收购，总价值3.5亿元。这些事件综合提升了公司的财务状况和市场竞争力。" \
5. Question: "query_type": "多文档信息整合问题", "Question": "华夏娱乐有限公司和绿源环保有限公司在2017年分别在哪个月进行了股东大会决议？", "Context": {"doc_ids": [0, 5]} \
    Answer: "两家公司均在2017年11月进行了股东大会决议。" \
6. Question: "query_type": "多文档时间序列问题", "Question": "比较建业集团有限公司和滨江消费品有限公司分别进行的董事会变更时间，哪家公司变更时间更早？", "Context": {"doc_ids": [11, 19]} \
    Answer: "建业集团有限公司的董事会变更时间更早，发生在2020年5月；滨江消费品有限公司的董事会变更发生在2021年3月。" \
7. Question: "query_type": "多文档对比问题", "Question": "比较2017年华夏娱乐有限公司和2018年顶级购物中心的净利润，哪家公司的净利润更高？", "Context": {"doc_ids": [0, 2]} \
    Answer: "2017年华夏娱乐有限公司的净利润更高，为8500万元；而2018年顶级购物中心的净利润为800万元。" \
Anser the queries accordding to the examples and the different of query types(answering each query type with different answer formats).
Use three sentences maximum and keep the answer concise.\n\nQuestion: {query} \nContext: {context} \nAnswer:\n"""
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