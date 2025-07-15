import os
import json
import requests
import time
import base64
import pandas as pd
import csv
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Retry failed Gemini API calls")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to original output CSV file")
    parser.add_argument("--benchmark_tsv", type=str, required=True, help="Path to input TSV with questions")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV after retrying")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., gemini-2.5-pro-preview-05-06)")
    parser.add_argument("--base_url", type=str, required=True, help="API base URL")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    return parser.parse_args()


def generate_indices_to_retry(input_csv_path):
    df = pd.read_csv(input_csv_path)
    df['api_response'] = df['api_response'].fillna('').astype(str)

    pattern = re.compile(r'^[A-E](?:\s*[.:]\s*.*)?$')
    keywords = ["nan", "sorry", "unable", "cannot", "can't", "don't"]

    contains_keywords = df['api_response'].str.lower().apply(
        lambda x: any(keyword in x for keyword in keywords)
    )
    not_match_pattern = ~df['api_response'].str.match(pattern)

    invalid_mask = contains_keywords | not_match_pattern
    invalid_indices = df[invalid_mask]

    invalid_indices[['index']].to_csv('indices_to_retry.csv', index=False)
    print(f"Saved {len(invalid_indices)} invalid indices to 'indices_to_retry.csv'")
    return set(invalid_indices['index'].astype(int))


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


def retry_failed_requests(args):
    indices_to_retry = generate_indices_to_retry(args.input_csv)
    df_input = pd.read_csv(args.benchmark_tsv, sep='\t')
    df_output_existing = pd.read_csv(args.input_csv)

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {args.api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "answer", "api_response"])
        print(f"Retrying on {len(indices_to_retry)} failed indices...")

        for _, row in df_input.iterrows():
            index = row["index"]
            if index not in indices_to_retry:
                existing = df_output_existing[df_output_existing['index'] == index]
                if not existing.empty:
                    writer.writerow(existing.iloc[0])
                continue

            prompt = "Answer with the option's letter from the given choices directly.\n"
            prompt += f"Question: {row['question']}\n"
            for opt in ['A', 'B', 'C', 'D', 'E']:
                if pd.notna(row.get(opt)) and row[opt] != "":
                    prompt += f"{opt}. {row[opt]}\n"

            try:
                encoded_image = encode_image(row['image_path'])
                payload = create_payload(prompt, encoded_image, args.model_name)

                for attempt in range(5):
                    try:
                        response = requests.post(f"{args.base_url}/v1/chat/completions", headers=headers, data=payload)
                        response.raise_for_status()
                        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        writer.writerow([index, row["answer"], content])
                        print(f"✅ Index {index}: {content}")
                        break
                    except Exception as e:
                        if attempt < 4:
                            print(f"⚠️  Retry {attempt + 1}/5 for index {index}: {e}")
                            time.sleep(5)
                        else:
                            print(f"❌ Failed index {index} after 5 retries: {e}")
                            writer.writerow([index, row["answer"], ""])
            except Exception as e:
                print(f"🚫 Skipping index {index} due to image error: {e}")
                writer.writerow([index, row["answer"], ""])


if __name__ == "__main__":
    args = parse_args()
    retry_failed_requests(args)
