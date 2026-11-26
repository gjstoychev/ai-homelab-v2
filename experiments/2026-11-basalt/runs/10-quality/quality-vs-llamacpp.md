| prompt | llamacpp-1gpu-best | strata-det-nodraft | basalt-1gpu-nodraft | basalt-1gpu-nodraft-det | basalt-1gpu-nodraft-det-skip7 | basalt-peer-mtp | basalt-peer-mtp-det |
|---|---|---|---|---|---|---|---|
| arith | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| capital | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| primes | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| sort | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| time | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| author | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| german | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| json | = ✓ | ≠ ✓ | ≠ ✓ ~ | ≠ ✓ | ≠ ✓ | ≠ ✓ ~ | = ✓ |
| code | = ✓ | ≠ ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| explain | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |

Reference: `llamacpp-1gpu-best`. = same text as the reference, ≠ different, ✓/✗ the prompt's check, ~ repetitions differ.

- **json**, `basalt-1gpu-nodraft-det-skip7`, `strata-det-nodraft`: leaves the reference at character 49 of 90
  - reference: `…'{"name": "Elara Vance", "age": 34, "tags": ["archaeologist", "cryptographer", "explorer"]}'…`
  - this:      `…'{"name": "Elara Vance", "age": 34, "tags": ["archivist", "cryptographer", "explorer"]}'…`
- **json**, `basalt-1gpu-nodraft`, `basalt-1gpu-nodraft-det`, `basalt-peer-mtp`: leaves the reference at character 1 of 90
  - reference: `…'{"name": "Elara Vance", "age": 34, "tags": ["archaeologist", '…`
  - this:      `…'{\n  "name": "Elara Vance",\n  "age": 34,\n  "tags": ["astronome'…`
- **json**, `basalt-1gpu-nodraft`, `basalt-peer-mtp`: leaves the reference at character 46 of 90
  - reference: `…'{"name": "Elara Vance", "age": 34, "tags": ["archaeologist", "cryptographer", "explorer"]}'…`
  - this:      `…'{"name": "Elara Vance", "age": 34, "tags": ["astronomer", "novelist", "gardener"]}'…`
- **code**, `strata-det-nodraft`: leaves the reference at character 46 of 123
  - reference: `…'```python\ndef is_palindrome(s):\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == '…`
  - this:      `…"```python\ndef is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return clea"…`
