import json
import jsonschema
import sys
import os

def validate(kb_path, schema_path):
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    try:
        jsonschema.validate(instance=kb, schema=schema)
        print("SUCCESS: JSON is valid against schema.")
    except jsonschema.exceptions.ValidationError as e:
        print(f"ERROR: Schema validation failed: {e.message}")
        print(f"Path: {e.path}")
        sys.exit(1)

if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
