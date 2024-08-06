import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx
from loguru import logger
import requests
from api_clients import moon_client

def estimate_token_count(messages, model="moonshot-v1-32k", api_key=None):
    url = "https://api.moonshot.cn/v1/tokenizers/estimate-token-count"
    payload = {
        "model": model,
        "messages": messages
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        total_tokens = result.get("data", {}).get("total_tokens")
        return total_tokens
    else:
        logger.error(f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
        return None

def upload_or_get_files(files: List[str], cache_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    messages = []

    file_list = moon_client.files.list()

    for file in files:
        try:
            _id = ""
            for remote in file_list:
                if remote.filename == os.path.split(file)[1]:
                    _id = remote.id
                    logger.info(f"Found remote file: {file}! Using remote content")
                    break
            if _id == "":
                logger.info(f"Remote file: {file} not found! Sending file to the remote")
                file_object = moon_client.files.create(file=Path(file), purpose="file-extract")
                _id = file_object.id
            file_content = moon_client.files.content(file_id=_id).text
            messages.append({
                "role": "system",
                "content": file_content,
            })
        except Exception as e:
            logger.error(f"文件上传失败: {file}, 错误: {e}")
            continue

    return messages