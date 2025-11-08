import os
import sys
import json
import pytest
import glob
import pandas as pd
import io
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from utils.tools import get_price_history


def get_json_files():
    """Helper function to get all JSON files in the json folder."""
    json_folder_path = os.path.join(os.path.dirname(__file__), "json")
    return glob.glob(os.path.join(json_folder_path, "*.json"))


@pytest.mark.parametrize("json_file", get_json_files())
def test_agent_predicted_correct_recommendation(json_file):
    """
        Test that the agent has predicted the correct recommendation.
        Compares the average closing price before and after.
    """
    filename = os.path.basename(json_file)
    
    data = None
    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            pytest.fail(f"Failed to parse JSON file: {filename}")
    
    # Check if debate section exists and has recommendation
    assert data is not None, f"Failed to load data from {filename}"
    assert 'debate' in data, f"No 'debate' section found in {filename}"
    assert 'recommendation' in data['debate'], f"No 'recommendation' found in debate section of {filename}"
    
    current_date = datetime.strptime(data['end_date'], "%Y-%m-%d")

    recommendation = data['debate']['recommendation']
    price_csv_str = get_price_history.invoke({"ticker": data['ticker'], "period": data['period'], "interval": data['interval'], "end_date": current_date})
    past_df = pd.read_csv(io.StringIO(price_csv_str))

    future_end_date = current_date + timedelta(data['horizon_days'])
    price_csv_str = get_price_history.invoke({"ticker": data['ticker'], "period": str(data['horizon_days'])+"d", "interval": data['interval'], "end_date": future_end_date})
    future_df = pd.read_csv(io.StringIO(price_csv_str))
    
    past_avg_close_price = past_df['Close'].mean()
    future_avg_close_price = future_df['Close'].mean()
    if past_avg_close_price <= future_avg_close_price:
        print(f"Average Closing Price increased/did not change from {past_avg_close_price} to {future_avg_close_price} before & after {data['end_date']}")
        expected = ["BUY", "HOLD"]
    else:
        print(f"Average Closing Price decreased from {past_avg_close_price} to {future_avg_close_price} before & after {data['end_date']}")
        expected = ["SELL"]
    assert recommendation in expected, \
        f"Invalid test case. Actual recommendation {recommendation} doesn't match the expected value {expected}"
    
    print(f"✓ {filename}: {recommendation} = expected: {expected}")
