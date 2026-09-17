import re

files = [
    "src/flexparameterize/tests/regression/test_arima.py",
    "src/flexparameterize/tests/regression/test_arima_surrogate.py",
]

for f in files:
    with open(f) as fh:
        content = fh.read()
    matches = re.findall(r"to_surrogate_spec.*input_units", content)
    print(f"{f}: {len(matches)} remaining")
    for m in matches:
        print(f"  {m}")
