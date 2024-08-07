import json
import os
from typing import List, Dict
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter, MarkdownElementNodeParser
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, SummaryIndex
from llama_index.readers.file import FlatReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever, SummaryIndexLLMRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import Document
from llama_index.core import PromptTemplate
from api_clients import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
logger.add("llama.log")

checkpoint_lock = threading.Lock()

# Define template for prompt
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)

# Configure embedding and LLM
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://10.26.1.146:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
model = "mistral-large:latest"
Settings.llm = Ollama(model=model, request_timeout=360.0,
                      base_url="http://10.26.1.146:11434")


def load_documents(file_path: Path) -> List[Document]:
    return FlatReader().load_data(file=file_path)


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = MarkdownElementNodeParser()
    Settings.text_splitter = text_splitter
    return text_splitter.get_nodes_from_documents(documents)


def generate_parse_node_prompt(ctx: str) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": """You are a skilled expert in reading academic papers, familiar with their structure and writing style.
Your job is to summarize the content provided by the user regards to the algorithm efficiency, for example, the inference time,
cost effectiveness, scalability, incremental training and human in the loop. if the content is related to these topics(or some of the
design can help these topics), you can summarize it. Please output json format. Here are some examples:

Example1:
----- context -----
2) Reliability Anomaly:  Reliability anomaly is detected based on the anomalous increase of error counts (EC). As shown in Figure 3(b) and Figure 3(e), the value of EC is 0 most of the time. In some cases, even when some errors are raised there is no inﬂuence on the business and the service may get back to normal in a short time. For example, service calls fail and errors are raised when the circuit break is open, but the service will get back to normal soon after the load decreases and the circuit break is closed. Therefore, we cannot use the performance anomaly detection model to detect reliability anomalies. Binary classiﬁcation models like Logic Regression (LR) or Random Forest (RF) can be used. However, as only a small number of EC changes are reliability anomalies, LR models are likely to cause overﬁtting.

Based on the characteristics of EC, we choose to use Ran- dom Forest (RF) to train a prediction model for anomalous EC increases. RF uses multiple decision trees for classiﬁcation. It can effectively combine multiple features and avoid overﬁtting. We deﬁne and use the following ﬁve features for the model. Note that some of the features combine other metrics (e.g., RT, QPS) together with EC, as anomalous EC increases often correlate with RT and QPS. Similar to performance anomaly detection, we consider the last 10 minutes as the current detection window.

Previous Day Delta Outlier Value : calculate the deltas • of the EC values in the last one hour and the EC values in the same hour of the previous day; use the 3-sigma rule to identify possible outliers of the deltas in the current detection window; if exist return the average of the outliers as the feature value, otherwise return 0.

Previous Minute Delta Outlier Value : calculate the • deltas of the EC values and the values of the previous minute in the last one hour; use the 3-sigma rule to identify possible outliers of the deltas in the current detection window; if exist return the average of the outliers as the feature value, otherwise return 0.

Response Time Over Threshold : whether the average • RT in the current detection window exceeds a predeﬁned threshold (e.g., 50ms).

Maximum Error Rate : the maximum error rate (i.e., EC • divided by number of requests) in the current detection window.

Correlation with Response Time : the Pearson correla- • tion coefﬁcient (see Equation 1) between EC and RT in the current detection window.
----- end context -----
----- response -----
{
  "algorithm_efficiency": {
    "inference_time": {
      "description": "The model considers the last 10 minutes as the current detection window, ensuring timely detection of reliability anomalies.",
      "relevance": "high"
    },
    "cost_effectiveness": {
      "description": "Random Forest (RF) is used to avoid overfitting and effectively combines multiple features, potentially reducing the need for frequent retraining.",
      "relevance": "high"
    },
    "scalability": {
      "description": "Random Forest's ability to handle multiple features and its robustness against overfitting suggests good scalability for large datasets and varied input features.",
      "relevance": "high"
    },
    "incremental_training": {
      "description": "Incremental training is not explicitly mentioned, but the use of recent data (last 10 minutes, last one hour) for feature calculation implies the model can adapt to new data without complete retraining.",
      "relevance": "medium"
    },
    "human_in_the_loop": {
      "description": "Human intervention is implied in setting predefined thresholds (e.g., average RT threshold) and potentially in validating the model's performance.",
      "relevance": "medium"
    }
  }
}
----- end response -----

Example2:

----- context -----
# C. Localization Efﬁciency (RQ2)

The 75 availability issues are collected from 28 subsystems and these subsystems have different numbers of services. To evaluate the efﬁciency and scalability of MicroHECL, we an- alyze the execution time of MicroHECL and the two baseline approaches for the 75 availability issues and investigate how the time changes with the size (service number) of the target system. The results are shown in Figure 4. Note that there may be multiple availability issues collected from the same subsystem and each of them is indicated by a point.

In general, the execution time of MicroHECL is  $22.3\%$ less than that of Microscope and  $31.7\%$  less than that of MonitorRank. For the subsystems of different sizes Micro- HECL uses the least time among the three approaches for most availability issues. The two curves in Figure 4 show the changes of the time differences of MicroHECL with the two baseline approaches respectively. It can be seen that the advantage of MicroHECL is not signiﬁcant when the number of services is less than 250; when the number of services exceeds 250, the advantage of MicroHECL is getting more and more signiﬁcant with the increase of service number. Moreover, the execution time of MicroHECL (also the two baseline approaches) increases linearly with the increase of service number, showing a good scalability.

The advantages of MicroHECL can be explained by it- s specially designed mechanisms for anomaly propagation analysis. MonitorRank traverses the service call graph using random walk algorithm and needs to calculate the correlation scores for a large number of the nodes in the graph. This process is time-consuming when the graph contains many nodes. Microscope traverses all the neighboring nodes of an

![](images/f6bef2f2363092153a7adab9e99e93b2c8f8e6561524e3f677fb7f46146b018e.jpg)
Fig. 4. Detection Time Changes with Num of Nodes

anomalous node, both upstream and downstream. Moreover, it has no pruning strategy. In contrast, MicroHECL considers only a single direction (upstream or downstream) for each anomaly type in anomaly propagation chain extension and uses a pruning strategy to eliminate branches that are likely irrelevant to the anomaly propagation.

# D. Effect of Pruning (RQ3)

The pruning strategy is a key for the accuracy and efﬁciency of root cause localization. We evaluate the effect of the pruning strategy by analyzing how the accuracy and time of root cause localization change with the threshold of correlation coefﬁcient. To this end, we run MicroHECL to analyze the 75 availability issues with different threshold settings and measure the accuracy and time. We choose the top-3 hit ratio (i.e.,  $\operatorname{HR}@3]$ ) as the indicator of accuracy.

The results of the evaluation are shown in Figure 5. It can be seen that both the accuracy and time decrease with the increase of the threshold, as more services and service call edges are pruned in the analysis process and less services are reached and considered. It can also be seen that the accuracy remains 0.67 when the threshold increases from 0 to 0.7, while the time decreases from 75 seconds to 46 seconds. This analysis conﬁrms the effectiveness of the pruning strategy, which can signiﬁcantly improve the efﬁciency of root cause localization while keeping the accuracy. And the best threshold is 0.7 for these availability issues.
----- end context -----
----- response -----
{
  "algorithm_efficiency": {
    "inference_time": {
      "description": "MicroHECL's execution time is 22.3% less than Microscope and 31.7% less than MonitorRank. It shows good scalability as execution time increases linearly with the number of services.",
      "relevance": "high"
    },
    "cost_effectiveness": {
      "description": "MicroHECL's pruning strategy reduces the number of nodes analyzed, thus reducing computational cost and improving efficiency.",
      "relevance": "high"
    },
    "scalability": {
      "description": "MicroHECL performs better than baseline approaches for larger systems (more than 250 services) and scales linearly with the number of services.",
      "relevance": "high"
    },
    "incremental_training": {
      "description": "Incremental training is not mentioned, but the model's design allows it to handle different sizes and numbers of services without retraining.",
      "relevance": "medium"
    },
    "human_in_the_loop": {
      "description": "Human intervention is implied in setting the threshold for the pruning strategy, which balances accuracy and efficiency.",
      "relevance": "medium"
    }
  }
}
----- end response -----
""",
        },
        {
            "role": "user",
            "content": f"""
            you should only output algorithm efficiency related information, like inference time, cost effectiveness, scalability,
            incremental
            training and human in
            the loop. You SHOULD put those do not have explicit evidence content to lower confidence. You have three types of choice:
            "high", "medium", "low".
            ----- context -----
            {ctx}
            ----- end context -----"""+"""
            ----- example output ----
            {
              "algorithm_efficiency": {
                "inference_time": {
                  "description": "your description",
                  "relevance": "your choice"
                },
                "cost_effectiveness": {
                  "description": "your description",
                  "relevance": "your choice"
                },
                "scalability": {
                  "description": "your description",
                  "relevance": "your choice"
                },
                "incremental_training": {
                  "description": "your description",
                  "relevance": "your choice"
                },
                "human_in_the_loop": {
                  "description": "your description",
                  "relevance": "your choice"
                }
              }
            }
            ----- end output ----
            """,
        },
    ]


def generate_summary_prompt(choice: str, context: str) -> List[Dict]:
    if choice == 'inference_time':
        return [
            {
                "role": "system",
                "content": """You are a skilled expert in reading academic papers, familiar with their structure and writing style.
    Your job is to summarize the content provided by the user regarding the algorithm efficiency. The provided content is assumed to be describing the inference time-related techniques. Your job is to summarize how it achieves shorter inference time, and what does it to achieve this. Consider the following questions:
    
    1. Did this paper use pruning techniques? If no, output "no", and stop.
    2. If yes, give a brief summary of how the paper use pruning techniques. In which stage, what component.
    3. If yes, give the original contents in the paper that can support your summary, for further reference.

    Some notes for you:
    1. You SHOULD output json format.
    {
        "use_pruning": "yes",
        "summary": "put your summary",
        "original_content": ["original content part1", "original content part2"]
    }
    2. You ONLY need to focus the technical part of the paper(i.e., implementation details and evaluations). For the introduction and backgroud part, you should omit.
    
    
    Here are some examples:
    
----- response -----
{
    "use_pruning": "yes",
    "summary": "The paper introduces MicroHECL, an efficient root cause localization approach for microservice systems. It utilizes a pruning strategy during the anomaly propagation chain analysis phase to enhance efficiency. This strategy eliminates irrelevant service calls by assessing the similarity of change trends in quality metrics between successive edges (service calls) in the propagation chain. Specifically, the Pearson correlation coefficient is employed to measure this similarity, and if the coefficient is below a certain threshold (e.g., 0.7), the edge is pruned, preventing the addition of potentially irrelevant anomalous nodes to the current anomaly propagation chain.",
    "original_content": [
    "To improve the efﬁciency of anomaly propagation chain analysis, MicroHECL adopts a pruning strategy to eliminate anomalous service call edges that are irrelevant to the current anomaly propagation chain.",
    "The pruning strategy is executed in the anomaly propagation chain extension process in the following way. Before adding an anomalous node to the current anomaly propagation chain, check the correlation coefﬁcient of the change trends of the corresponding quality metric of the current edge (service call) with the adjacent upstream/downstream edge. If the correlation coefﬁcient is lower than a threshold (e.g., 0.7), the current edge will be pruned and the current anomalous node will not be added to the current anomaly propagation chain."
    ]
}
----- response end-----
    """},
            {
                "role": "user",
                "content": f"""Consider the following questions:
    
    1. Did this paper use pruning techniques? If no, output "no", and stop.
    2. If yes, give a brief summary of how the paper use pruning techniques. In which stage, what component.
    3. If yes, give the original contents in the paper that can support your summary, for further reference.
    
    ----- context -----
    {context}
    ----- end context -----
    """
            }
        ]
    else:
        logger.info(f"No such choice: {choice}")
        return [
            {
                "role": "system",
                "content": """You are a skilled expert in reading academic papers, familiar with their structure and writing style."""
            }
        ]


def process_node_with_llm(node, llama_client, model) -> Dict:
    completion = llama_client.chat.completions.create(
        model=model,
        messages=generate_parse_node_prompt(ctx=node.get_content()),
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    logger.info(f"input context: {node.get_content()}")
    content = completion.choices[0].message.content
    logger.info(f"response: {content}")
    parse_result = json.loads(completion.choices[0].message.content)
    parse_result["original_content"] = node.get_content()
    return parse_result


def summary_with_llm(ctx, llama_client, model) -> Dict:
    completion = llama_client.chat.completions.create(
        model=model,
        messages=generate_summary_prompt("inference_time", ctx),
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    logger.info(f"input context: {ctx}")
    content = completion.choices[0].message.content
    logger.info(f"response: {content}")
    return content


def save_checkpoint(data, checkpoint_file="checkpoint.json"):
    with checkpoint_lock:
        with open(checkpoint_file, "w") as file:
            json.dump(data, file, indent=4)


def load_checkpoint(checkpoint_file="checkpoint.json"):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as file:
            return json.load(file)
    return {"document_info": [], "checkpoint_data": {}}


def process_node(node, llama_client, model, file_str, node_idx, document_info, checkpoint_data, checkpoint_file):
    try:
        result = process_node_with_llm(node, llama_client, model)
        result['file'] = file_str
        result['node_index'] = node_idx
        document_info.append(result)

        checkpoint_data[file_str] = {
            "last_processed_node": node_idx, "completed": False}
        save_checkpoint({"document_info": document_info,
                        "checkpoint_data": checkpoint_data}, checkpoint_file)
    except Exception as e:
        logger.error(
            f"Error processing node {node_idx} in file {file_str}: {e}")
        raise


def process_file(file_path, llama_client, model, document_info, checkpoint_data, checkpoint_file):
    file_str = str(file_path)
    if file_str in checkpoint_data and checkpoint_data[file_str]["completed"]:
        return

    try:
        documents = load_documents(file_path)
        nodes = split_documents(documents)

        node_idx = checkpoint_data[file_str].get(
            "last_processed_node", -1) + 1 if file_str in checkpoint_data else 0
        while node_idx < len(nodes):
            node = nodes[node_idx]
            process_node(node, llama_client, model, file_str, node_idx,
                         document_info, checkpoint_data, checkpoint_file)
            node_idx += 1

        checkpoint_data[file_str]["completed"] = True
        save_checkpoint({"document_info": document_info,
                        "checkpoint_data": checkpoint_data}, checkpoint_file)

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")


def process_documents(file_paths: List[Path], llama_client, model, output_file="outputs/paper.json", checkpoint_file="outputs/checkpoint.json"):
    checkpoint = load_checkpoint(checkpoint_file)
    document_info = checkpoint["document_info"]
    checkpoint_data = checkpoint["checkpoint_data"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, file_path, llama_client, model,
                                   document_info, checkpoint_data, checkpoint_file) for file_path in file_paths]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in processing file: {e}")

    with open(output_file, "w") as file:
        json.dump(document_info, file, indent=4)
