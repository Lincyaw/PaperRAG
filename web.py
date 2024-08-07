from flask import Flask, render_template_string, request, jsonify
import json
import urllib.parse
import markdown
import os


app = Flask(__name__)

# Load and parse the JSON data
with open('items_grouped.json') as f:
    data = json.load(f)
# Check if 'data.json' exists, if not create it with default values
if not os.path.exists('data.json'):
    default_status = {file: "未开始" for file in data.keys()}
    with open('data.json', 'w') as f:
        json.dump(default_status, f)

with open('data.json') as f:
    status_data = json.load(f)

# HTML templates with TailwindCSS integration
file_list_template = '''
<!doctype html>
<html>
<head>
    <title>File List</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        function updateStatus(file, status) {
            $.ajax({
                url: '/update_status',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ file: file, status: status }),
                success: function(response) {
                    console.log(response.message);
                }
            });
        }
    </script>
</head>
<body class="bg-gray-100 text-gray-900">
    <div class="container mx-auto p-4">
        <h1 class="text-3xl font-bold mb-4">File List</h1>
        <ul class="list-disc list-inside">
        {% for file in files %}
            <li class="mb-2 flex items-center">
                <a class="text-blue-500 hover:underline flex-grow" href="/file/{{ file | urlencode }}">{{ file }}</a>
                <select class="ml-4 p-1 border rounded" onchange="updateStatus('{{ file }}', this.value)">
                    <option value="未开始" {% if statuses[file] == '未开始' %}selected{% endif %}>未开始</option>
                    <option value="阅读中" {% if statuses[file] == '阅读中' %}selected{% endif %}>阅读中</option>
                    <option value="已结束" {% if statuses[file] == '已结束' %}selected{% endif %}>已结束</option>
                </select>
            </li>
        {% endfor %}
        </ul>
    </div>
</body>
</html>
'''

file_content_template = '''
<!doctype html>
<html>
<head>
    <title>{{ file }}</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script>
        function toggleContent(id) {
            var element = document.getElementById(id);
            var button = document.getElementById(id + '-button');
            if (element.style.display === "none") {
                element.style.display = "block";
                button.innerHTML = "Show Less";
            } else {
                element.style.display = "none";
                button.innerHTML = "Show More";
            }
        }
    </script>
</head>
<body class="bg-gray-100 text-gray-900">
    <div class="container mx-auto p-4">
        <h1 class="text-3xl font-bold mb-4">{{ file }}</h1>
        <ul class="divide-y divide-gray-300">
        {% for item in items %}
            <li class="py-2 flex flex-col">
                <div class="w-full mb-2 font-semibold">{{ loop.index }}. {{ item[0] }}</div>
                <div id="content-{{ loop.index }}" class="w-full overflow-hidden" style="display: none;">
                    {{ item[1] | safe }}
                </div>
                <button id="content-{{ loop.index }}-button" class="text-blue-500 hover:underline self-start" onclick="toggleContent('content-{{ loop.index }}')">Show More</button>
            </li>
        {% endfor %}
        </ul>
        <a class="text-blue-500 hover:underline mt-4 inline-block" href="/">Back to File List</a>
    </div>
</body>
</html>
'''


@app.route('/')
def file_list():
    files = data.keys()
    return render_template_string(file_list_template, files=files, statuses=status_data)


@app.route('/file/<path:encoded_filename>')
def file_content(encoded_filename):
    filename = urllib.parse.unquote(encoded_filename)
    items = data.get(filename, {}).get('items', [])

    # Convert Markdown content to HTML
    converted_items = [(item[0], markdown.markdown(item[1])) for item in items]

    return render_template_string(file_content_template, file=filename, items=converted_items)


@app.route('/update_status', methods=['POST'])
def update_status():
    content = request.json
    file = content['file']
    status = content['status']
    status_data[file] = status

    with open('data.json', 'w') as f:
        json.dump(status_data, f)

    return jsonify({'message': 'Status updated successfully'})


if __name__ == '__main__':
    app.run(debug=True)
