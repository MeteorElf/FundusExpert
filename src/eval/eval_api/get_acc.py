import pandas as pd
import re
import argparse

def extract_api_answer(response):
    """Extracts the answer choice (A-E) from the API response."""
    if pd.isna(response):
        return ''
    # Match responses that start with A, B, C, D, or E
    match = re.search(r'^[A-E]', str(response))
    return match.group() if match else ''

def evaluate(gpt_response_csv, benchmark_tsv, output_csv):
    """Evaluates the accuracy of GPT responses."""
    # Read the GPT response data
    df = pd.read_csv(gpt_response_csv)

    # Extract the API answer and calculate if it's a hit
    df['api_answer'] = df['api_response'].apply(extract_api_answer)
    df['hit'] = 0
    df.loc[
        (df['api_answer'] == df['answer'].str[0]) &
        (df['api_answer'] != '') &
        (df['answer'].str[0] != ''),
        'hit'
    ] = 1

    # Overwrite the original file to include the new columns
    df.to_csv(gpt_response_csv, index=False)

    # Merge benchmark data and calculate accuracy
    df_hit = df
    df_benchmark = pd.read_csv(benchmark_tsv, sep='\t')

    # Ensure the 'index' column is of integer type for merging
    df_hit['index'] = df_hit['index'].astype(int)
    df_benchmark['index'] = df_benchmark['index'].astype(int)

    # Merge the dataframes
    merged_df = pd.merge(df_hit, df_benchmark, on='index', how='left')
    merged_df['category'] = merged_df['category'].fillna('Unknown')
    merged_df['hit'] = merged_df['hit'].astype(int)

    # Calculate hit rate by category
    category_hit_sum = merged_df.groupby('category')['hit'].sum()
    category_response_count = merged_df.groupby('category')['hit'].count()
    hit_rate = category_hit_sum / category_response_count
    hit_rate = hit_rate.fillna(0)

    all_categories = merged_df['category'].unique().tolist()
    
    # Create the output DataFrame
    output_df = pd.DataFrame(columns=['Overall'] + all_categories)

    # Calculate and populate the overall accuracy
    overall_accuracy = round(merged_df['hit'].sum() / merged_df['hit'].count(), 3)

    # Create a dictionary to hold the results row
    output_data = {'Overall': overall_accuracy}

    # Populate the accuracy for each category
    for category in all_categories:
        output_data[category] = hit_rate.get(category, 0.0)

    # Add the data to the DataFrame
    output_df = pd.DataFrame([output_data])

    # Save the results to a CSV file
    output_df.to_csv(output_csv, index=False)
    print(f"[Done] Results have been saved to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GPT response accuracy.")
    parser.add_argument('--gpt_response_csv', type=str, required=True, help='Path to the GPT response CSV file')
    parser.add_argument('--benchmark_tsv', type=str, required=True, help='Path to the benchmark TSV file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file')

    args = parser.parse_args()
    evaluate(args.gpt_response_csv, args.benchmark_tsv, args.output_csv)