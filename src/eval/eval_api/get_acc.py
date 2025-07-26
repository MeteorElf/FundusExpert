import pandas as pd
import re
import argparse
import os

def extract_api_answer(response):
    if pd.isna(response):
        return ''
    match = re.search(r'^[A-E]', str(response))
    return match.group() if match else ''

def evaluate(gpt_response_csv, benchmark_tsv, output_csv):
    df = pd.read_csv(gpt_response_csv)

    df['api_answer'] = df['api_response'].apply(extract_api_answer)
    df['hit'] = 0
    df.loc[
        (df['api_answer'] == df['answer'].str[0]) &
        (df['api_answer'] != '') &
        (df['answer'].str[0] != ''),
        'hit'
    ] = 1

    df.to_csv(gpt_response_csv, index=False)  # Overwrite the original file

    # Merge benchmark data and calculate accuracy
    df_hit = df
    df_benchmark = pd.read_csv(benchmark_tsv, sep='\t')

    df_hit['index'] = df_hit['index'].astype(int)
    df_benchmark['index'] = df_benchmark['index'].astype(int)

    merged_df = pd.merge(df_hit, df_benchmark, on='index', how='left')
    merged_df['category'] = merged_df['category'].fillna('Unknown')
    merged_df['hit'] = merged_df['hit'].astype(int)

    category_hit_sum = merged_df.groupby('category')['hit'].sum()
    category_response_count = merged_df.groupby('category')['hit'].count()
    hit_rate = category_hit_sum / category_response_count
    hit_rate = hit_rate.fillna(0)

    if "GMAI_mm" in os.path.basename(benchmark_tsv):
        required_categories = [
            "optic cup","microaneurysms","cotton wool spots", "retinal soft exudates", "retinal hard exudates","massive hard exudates",
            "dragged disc", "disc swelling and elevation","large optic cup",
            "non glaucoma","mid advanced glaucoma", "glaucoma",
            "cataract", "macular edema", "myopia", "pathological myopia",
            
            "no diabetic retinopathy","mild (or early) nonproliferative diabetic retinopathy","moderate nonproliferative diabetic retinopathy", "severe nonproliferative diabetic retinopathy", "very severe nonproliferative diabetic retinopathy", "proliferative diabetic retinopathy","advanced proliferative diabetic retinopathy",
            "level 0 diabetic retinopathy","level 1 diabetic retinopathy","level 2 diabetic retinopathy","level 3 diabetic retinopathy","level 4 diabetic retinopathy",
            "severe hypertensive retinopathy",
            "image with bad quality","image with good quality",
            
            "retinal hemorrhages","bietti crystalline dystrophy","branch retinal vein occlusion","central retinal vein occlusion","central serous chorioretinopathy","chorioretinal atrophy coloboma",
            "congenital disc abnormality","epiretinal membrane","fibrosis","fundus neoplasm","hypertension","laser spots",
            "left macula centered eye fundus images","maculopathy", "right macula centered eye fundus images",
            "myelinated nerve fiber","optic atrophy","retinitis pigmentosa","rhegmatogenous retinal detachment",
            "peripheral retinal degeneration and break","preretinal hemorrhage","retina disease","retinal artery occlusion",

            "silicon oil in eye","tessellated fundus","vessel tortuosity","vitreous particles","vkh disease","yellow white spots flecks"
    ]
    else:
        required_categories = [
            "optic cup", "optic disc", "microaneurysms", "retinal hard exudates", "cotton wool spots", 
            "no diabetic retinopathy",
            "proliferative diabetic retinopathy",
            "mild nonproliferative diabetic retinopathy",
            "moderate nonproliferative diabetic retinopathy",
            "severe nonproliferative diabetic retinopathy",
            "early age-related macular degeneration",
            "intermediate age-related macular degeneration",
            "advanced age-related macular degeneration",
            "no hypertensive retinopathy",
            "mild hypertensive retinopathy",
            "moderate hypertensive retinopathy",
            "severe hypertensive retinopathy",
            "non glaucoma",
            "glaucoma",
            "non cataract",
            "cataract",
            "non macular edema",
            "macular edema",
            "non drusens",
            "drusens",
            "non myopia",
            "myopia",
            "non increased cup-to-disc ratio",
            "increased cup-to-disc ratio",
            "non pathological myopia",
            "pathological myopia"
    ]

    output_df = pd.DataFrame(index=['none'], columns=['split', 'Overall'] + required_categories)
    output_df.loc['none', 'split'] = 'none'
    output_df.loc['none', 'Overall'] = merged_df['hit'].sum() / merged_df['hit'].count()

    for category in required_categories:
        output_df.loc['none', category] = hit_rate.get(category, 0.0)

    output_df.to_csv(output_csv, index=False)
    print(f"[Done] Results saved to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GPT response accuracy.")
    parser.add_argument('--gpt_response_csv', type=str, required=True, help='Path to GPT response CSV file')
    parser.add_argument('--benchmark_tsv', type=str, required=True, help='Path to benchmark TSV file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file')

    args = parser.parse_args()
    evaluate(args.gpt_response_csv, args.benchmark_tsv, args.output_csv)
