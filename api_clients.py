from openai import OpenAI
from config import API_KEY, MOONSHOT_BASE_URL, LLAMA_BASE_URL

moon_client = OpenAI(
    api_key=API_KEY,
    base_url=MOONSHOT_BASE_URL,
)

llama_client = OpenAI(
    api_key=API_KEY,
    base_url=LLAMA_BASE_URL,
)

if __name__ == '__main__':
    files = moon_client.files.list()
    file_dict = {}
    queue = []

    for file in files:
        if file.filename in file_dict:
            if file.created_at < file_dict[file.filename]['created_at']:
                queue.append(file_dict[file.filename]['id'])
                file_dict[file.filename] = {'id': file.id, 'created_at': file.created_at}
            else:
                queue.append(file.id)
        else:
            file_dict[file.filename] = {'id': file.id, 'created_at': file.created_at}

    for file_id in queue:
        moon_client.files.delete(file_id=file_id)
        print(f"File with ID {file_id} is deleted")
