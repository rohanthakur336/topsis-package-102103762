import sys
import pandas as pd
import numpy as np

def check_input_params():
    if len(sys.argv) != 5:
        print("Usage: python topsis.py <input_file.csv> <weights> <impacts> <result_file.csv>")
        sys.exit(1)

def read_input_file(input_file):
    try:
        data = pd.read_csv(input_file)
        return data
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File {input_file} is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Unable to parse the content of {input_file}. Ensure it is a valid CSV file.")
        sys.exit(1)

def check_numeric_columns(data):
    try:
        data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    except ValueError as e:
        print(f"Error: Unable to convert columns to numeric. Reason: {str(e)}")
        sys.exit(1)

    if data.iloc[:, 1:].isna().any().any():
        print("Error: Columns from 2nd to last must contain numeric values only.")
        sys.exit(1)

    if not np.isfinite(data.iloc[:, 1:]).all().all():
        print("Error: Dataset contains non-finite (NaN or Inf) values after numeric conversion.")
        sys.exit(1)




def check_column_count(data):
    if len(data.columns) < 3:
        print("Error: Input file must contain three or more columns.")
        sys.exit(1)

def check_weights_impacts(weights, impacts, data):
    if len(weights) != len(impacts) or len(weights) != len(data.columns) - 1:
        print("Error: Number of weights, impacts, and columns (from 2nd to last) must be the same.")
        sys.exit(1)

    for impact in impacts:
        if impact not in ['+1', '-1']:
            print("Error: Impacts must be either +1 or -1.")
            sys.exit(1)

def save_results(result, topsis_scores, result_file, input_file):
    result_df = pd.DataFrame({"Rankings": result, "TOPSIS_Score": topsis_scores})

    # Read the original dataset
    input_df = pd.read_csv(input_file)

    # Add the results to the original dataset
    merged_df = pd.concat([input_df, result_df], axis=1)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(result_file, index=False)
    print(f"Results merged with the original dataset and saved to {result_file}")


def topsis(dataset, weights, impacts):
    # Convert dataset to float, handling NaN values
    dataset = dataset.astype(float)

    # Normalize the matrix
    normalized_matrix = dataset / np.linalg.norm(dataset, axis=0)

    # Multiply each column by its weight
    weighted_matrix = normalized_matrix * weights

    # Determine the ideal and negative-ideal solutions
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    # Calculate the distance from the ideal and negative-ideal solutions
    distance_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    distance_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

    # Calculate the relative closeness to the ideal solution
    closeness = distance_worst / (distance_best + distance_worst)

    # Calculate TOPSIS scores
    topsis_scores = 1 - closeness

    # Rank the alternatives based on closeness
    rankings = np.argsort(closeness)[::-1] + 1  # Add 1 to make the rankings start from 1

    return rankings, topsis_scores

def main():
    # Check the number of parameters
    check_input_params()

    # Read input file
    input_file = sys.argv[1]
    data = read_input_file(input_file)

    # Check the number of columns
    check_column_count(data)

    # Check if columns from 2nd to last are numeric
    check_numeric_columns(data)

    # Read weights and impacts
    weights = list(map(float, sys.argv[2].split(',')))
    impacts = sys.argv[3].split(',')

    # Check weights and impacts
    check_weights_impacts(weights, impacts, data)

    # Convert impacts to 1 for benefit and -1 for cost
    impacts = [1 if i == '+' else -1 for i in impacts]



    # Apply Topsis method
    result, topsis_scores = topsis(data.values[:, 1:], np.array(weights), np.array(impacts))

    # Save the results
    result_file = sys.argv[4]
    save_results(result, topsis_scores, result_file, input_file)

if __name__ == "__main__":
    main()
