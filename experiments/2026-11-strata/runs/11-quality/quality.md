| prompt | llamacpp-1gpu-best | strata-nodraft | strata-det-nodraft | strata-det-mtp | strata-det-skip7 |
|---|---|---|---|---|---|
| arith | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| capital | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| primes | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| sort | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| time | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| author | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| german | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| json | = ✓ | ≠ ✓ ~ | ≠ ✓ | ≠ ✓ | ≠ ✓ |
| code | = ✓ | ≠ ✓ ~ | ≠ ✓ | = ✓ | = ✓ |
| explain | = ✓ | = ✓ | = ✓ | = ✓ | ≠ ✓ |

Reference: `llamacpp-1gpu-best`. = same text as the reference, ≠ different, ✓/✗ the prompt's check, ~ repetitions differ.

- **json**, `strata-det-nodraft`, `strata-det-skip7`, `strata-nodraft`: leaves the reference at character 49 of 90
  - reference: `…'{"name": "Elara Vance", "age": 34, "tags": ["archaeologist", "cryptographer", "explorer"]}'…`
  - this:      `…'{"name": "Elara Vance", "age": 34, "tags": ["archivist", "cryptographer", "explorer"]}'…`
- **json**, `strata-det-mtp`: leaves the reference at character 46 of 90
  - reference: `…'{"name": "Elara Vance", "age": 34, "tags": ["archaeologist", "cryptographer", "explorer"]}'…`
  - this:      `…'{"name": "Elara Vance", "age": 34, "tags": ["astronomer", "novelist", "gardener"]}'…`
- **code**, `strata-det-nodraft`, `strata-nodraft`: leaves the reference at character 46 of 123
  - reference: `…'```python\ndef is_palindrome(s):\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == '…`
  - this:      `…"```python\ndef is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return clea"…`
- **explain**, `strata-det-skip7`: leaves the reference at character 361 of 434
  - reference: `…' blue, our eyes are more sensitive to blue, and some violet is absorbed by the upper atmosphere, resulting in the sky ap'…`
  - this:      `…' blue, our eyes are more sensitive to blue, and some violet light is absorbed by the upper atmosphere, causing us to per'…`
