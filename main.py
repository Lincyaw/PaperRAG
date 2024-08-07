from llama import *
import os
from pathlib import Path
from api_clients import *


def parse_paper():
    files_to_process = []
    for root, dirs, files in os.walk("inputs"):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                print(f"Path: {file_path}, File Name: {file}")
                files_to_process.append(Path(file_path))
    process_documents(files_to_process, llama_client, model)


def merge_strings(strings):
    if not strings:
        return ""

    merged_string = strings[0]

    for i in range(1, len(strings)):
        overlap_len = min(len(merged_string), len(strings[i]))
        while overlap_len > 0:
            if merged_string[-overlap_len:] == strings[i][:overlap_len]:
                break
            overlap_len -= 1
        merged_string += strings[i][overlap_len:]

    return merged_string


def load_parse_result():
    with open("outputs/paper.json", "r") as file:
        data = json.load(file)
    grouped_items = {}
    for item in data:
        if item['file'] not in grouped_items:
            grouped_items[item['file']] = {"total": 0, "items": []}
        grouped_items[item['file']]['total'] += 1
        for key in item['algorithm_efficiency']:
            if item['algorithm_efficiency'][key]['relevance'] == 'high':
                grouped_items[item['file']]["items"].append(
                    (item['algorithm_efficiency'][key]['description'], item['original_content']))
    results = []
    for k, _ in grouped_items.items():
        results.append(
            (k, merge_strings([i[1]
                               for i in grouped_items[k]['items']])))
    with open("items_grouped.json", "w") as file:
        json.dump(grouped_items, file, indent=4)

    return results


if __name__ == "__main__":
    # parse_paper()
    # exit()
    for i in load_parse_result():
        summary_with_llm(i[1], llama_client, model)
