#!/usr/bin/env python3
"""
Table Extraction Pipeline: GPT-5.2 (Vision) Approach
====================================================
Uses OpenAI's GPT-5.2 to extract transaction data from a bank statement
image. Unlike the Tesseract approach, this works across different banks
and layouts without hardcoded markers or correction maps.

Dependencies:
    pip install openai

Usage:
    export OPENAI_API_KEY="your-key-here"
    python extraction_gpt.py
"""

import os
import sys
import csv
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def encode_image(image_path):
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_transactions(client, image_path):
    """
    Send the bank statement image to GPT-5.2 and ask it to extract
    all transactions as structured JSON.
    """
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL transactions from this bank statement image.\n"
                            "Return a JSON array where each transaction has these exact keys:\n"
                            '  "Date Posted", "Value Date", "Cheque Number", "Description", "Amount", "Balance"\n\n'
                            "Rules:\n"
                            "- Date format: DD MAY (e.g. '15 MAY', '27 MAY')\n"
                            "- The first row is B/F Balance — set Value Date to 'B/F Balance', "
                            "leave Cheque Number, Description, and Amount empty\n"
                            "- Cheque Number is empty for all rows in this statement\n"
                            "- For Description, combine all description lines into a single string separated by spaces\n"
                            "- Amount values that are debits should end with 'DR' (e.g. '7,010.00DR')\n"
                            "- Include commas in numbers where appropriate (e.g. '10,053.38')\n"
                            "- Return ONLY the JSON array, no markdown or explanation"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_completion_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw)


def export_csv(transactions, output_path):
    """
    Write transactions to CSV with minimal quoting
    (only values containing commas are double-quoted).
    """
    columns = [
        "Date Posted",
        "Value Date",
        "Cheque Number",
        "Description",
        "Amount",
        "Balance",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for tx in transactions:
            row = {col: tx.get(col, "") for col in columns}
            writer.writerow(row)

    print(f"Wrote {len(transactions)} transactions to {output_path}")


def validate(output_path, reference_path):
    """Compare the generated CSV against the reference file line by line."""
    with open(output_path) as f1, open(reference_path) as f2:
        out_lines = f1.readlines()
        ref_lines = f2.readlines()

    if len(out_lines) != len(ref_lines):
        print(f"MISMATCH: output has {len(out_lines)} lines, "
              f"reference has {len(ref_lines)} lines")
        return False

    all_match = True
    for i, (ol, rl) in enumerate(zip(out_lines, ref_lines), 1):
        if ol != rl:
            print(f"Line {i} differs:")
            print(f"  GOT:      {ol.rstrip()}")
            print(f"  EXPECTED: {rl.rstrip()}")
            all_match = False

    if all_match:
        print("VALIDATION PASSED — output matches reference exactly.")
    return all_match


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "img_sample.jpg")
    output_path = os.path.join(script_dir, "transactions_gpt.csv")
    reference_path = os.path.join(script_dir, "csv_sample.csv")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set your OPENAI_API_KEY environment variable first.")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    client = OpenAI()

    print(f"Image:  {image_path}")
    print(f"Output: {output_path}")
    print()

    print("Step 1: Sending image to GPT-5.2 ...")
    transactions = extract_transactions(client, image_path)
    print(f"         Extracted {len(transactions)} transactions")

    print("Step 2: Exporting to CSV ...")
    export_csv(transactions, output_path)

    if os.path.exists(reference_path):
        print("\nStep 3: Validating against reference ...")
        validate(output_path, reference_path)


if __name__ == "__main__":
    main()
