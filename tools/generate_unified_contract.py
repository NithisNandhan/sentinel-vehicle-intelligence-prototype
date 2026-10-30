"""Derive browser types and JSON Schema from the shared Pydantic contract."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app import unified_contract as contract


def ts(schema):
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in schema:
        return " | ".join(ts(item) for item in schema["anyOf"])
    if "const" in schema:
        return json.dumps(schema["const"])
    if "enum" in schema:
        return " | ".join(json.dumps(item) for item in schema["enum"])
    kind = schema.get("type")
    if kind == "array":
        return "Array<" + ts(schema["items"]) + ">"
    if kind == "object":
        if "properties" not in schema:
            return "Record<string, " + ts(schema.get("additionalProperties", {})) + ">"
        required = schema.get("required", [])
        return "{\n" + "\n".join("  " + name + ("" if name in required else "?") + ": " + ts(value) + ";" for name, value in schema["properties"].items()) + "\n}"
    return {"string": "string", "integer": "number", "number": "number", "boolean": "boolean", "null": "null"}.get(kind, "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    definitions = {}
    for name, model in vars(contract).items():
        if isinstance(model, type) and issubclass(model, contract.ContractModel) and model is not contract.ContractModel:
            schema = model.model_json_schema()
            definitions.update(schema.pop("$defs", {}))
            definitions[name] = schema
    outputs = {
        ROOT / "contracts/unified-dashboard.schema.json": json.dumps({"$schema": "https://json-schema.org/draft/2020-12/schema", "$defs": definitions}, indent=2) + "\n",
        ROOT / "frontend/src/unified-contract.ts": "// Generated from backend/app/unified_contract.py; do not edit.\n" + "\n\n".join("export type " + name + " = " + ts(schema) + ";" for name, schema in sorted(definitions.items())) + "\n",
    }
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit("Contract is stale: " + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print("Unified contract " + ("verified" if args.check else "generated"))


if __name__ == "__main__":
    main()
