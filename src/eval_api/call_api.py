import os
import json
import requests
import time
import base64
import pandas as pd
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run API benchmark on TSV input")
    parser.add_argument("--benchmark_tsv", type=str, required=True, help="Path to the input TSV file. The image path in tsv is the local image path, such as:/path/to/img")
    parser.add_argument("--model_name", type=str, required=True, help="Model name used in the API request")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL of the Claude/Gemini API")
    parser.add_argument("--api_key", type=str, required=True, help="API key for authentication")
    return parser.parse_args()


def prepare_prompt(row):
    prompt = "Answer with the option's letter from the given choices directly.\n"
    prompt += f"Question: {row['question']}\n"
    for option in ['A', 'B', 'C', 'D', 'E']:
        if pd.notna(row.get(option)) and row[option] != "":
            prompt += f"{option}. {row[option]}\n"
    return prompt


def encode_image(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def create_payload(prompt, encoded_image, model_name):
    return json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a professional fundus imaging expert."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]
            }
        ],
        "temperature": 0,
        "top_p": 0,
        "seed": 42
    })


def send_request(payload, headers, base_url, retries=5, delay=5):
    for attempt in range(retries):
        try:
            response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, data=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"Request failed (attempt {attempt + 1}/{retries}): {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Final failure after {retries} attempts: {e}")
                return ""


def main():
    args = parse_args()

    input_tsv = args.benchmark_tsv
    model_name = args.model_name
    base_url = args.base_url
    api_key = args.api_key

    benchmark_name = os.path.splitext(os.path.basename(input_tsv))[0]
    output_csv = f"./result/{benchmark_name}/{model_name}/response.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(input_tsv, sep='\t')

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["index", "answer", "api_response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()
            print(f'###############{benchmark_name}#############\n')

        for _, row in df.iterrows():
            try:
                prompt = prepare_prompt(row)
                encoded_image = encode_image(row['image_path'])
                payload = create_payload(prompt, encoded_image, model_name)
                response_text = send_request(payload, headers, base_url)

                writer.writerow({
                    "index": row["index"],
                    "answer": row["answer"],
                    "api_response": response_text
                })
                csvfile.flush()
                os.fsync(csvfile.fileno())
                print(f"{row['index']} \t {row['answer']} \t {response_text}")

            except Exception as e:
                print(f"Error processing index {row['index']}: {e}")


if __name__ == "__main__":
    main()
