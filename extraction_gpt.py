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
    all transactions as structured JSON. Dynamically detects the
    column structure from the image.
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
                            "Extract ALL transactions from this bank statement image.\n\n"
                            "Step 1: Identify the column headers exactly as they appear in the table.\n"
                            "Step 2: Extract every transaction row using those exact column names as JSON keys.\n\n"
                            "Return a JSON object with two keys:\n"
                            '  "columns": an array of the column header names exactly as shown in the image\n'
                            '  "transactions": an array of objects, each using those column names as keys\n\n'
                            "Rules:\n"
                            "- Preserve dates exactly as shown (e.g. '15 MAY', '03/02', 'Mar 15')\n"
                            "- Preserve numbers exactly as shown, including commas and any suffixes like 'DR' or 'CR'\n"
                            "- If there is a B/F Balance or Previous Balance row, include it as a transaction\n"
                            "- Do NOT include summary rows like 'Ending balance' or 'Balance Carried Forward'\n"
                            "- For multi-line descriptions, combine them into a single string separated by spaces\n"
                            "- If a cell is empty, use an empty string\n"
                            "- Return ONLY the JSON object, no markdown or explanation"
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

    result = json.loads(raw)
    return result["columns"], result["transactions"]


def export_csv(columns, transactions, output_path):
    """
    Write transactions to CSV using the detected columns.
    Only values containing commas are double-quoted.
    """
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
            print(f"GOT: {ol.rstrip()}")
            print(f"EXPECTED: {rl.rstrip()}")
            all_match = False

    if all_match:
        print("VALIDATION PASSED — output matches reference exactly.")
    return all_match


def process_image(client, image_path, output_path, reference_path=None):
    """Process a single bank statement image end-to-end."""
    print(f"Image: {image_path}")
    print(f"Output:{output_path}")
    print()

    print("Step 1: Sending image to GPT-5.2 ...")
    columns, transactions = extract_transactions(client, image_path)
    print(f"Detected columns: {columns}")
    print(f"Extracted {len(transactions)} transactions")

    print("Step 2: Exporting to CSV ...")
    export_csv(columns, transactions, output_path)
    print(f"Wrote {len(transactions)} transactions to {output_path}")

    if reference_path and os.path.exists(reference_path):
        print("\nStep 3: Validating against reference ...")
        validate(output_path, reference_path)

    print()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "img_sample.jpg")
    output_path = os.path.join(script_dir, "transactions_gpt.csv")
    reference_path = os.path.join(script_dir, "csv_sample.csv")

    client = OpenAI()
    process_image(client, image_path, output_path, reference_path)


if __name__ == "__main__":
    main()
