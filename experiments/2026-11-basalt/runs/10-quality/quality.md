| prompt | 1gpu-nodraft | 1gpu-nodraft-det | 1gpu-nodraft-det-skip7 | peer-mtp | peer-mtp-det |
|---|---|---|---|---|---|
| arith | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| capital | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| primes | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| sort | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| time | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| author | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| german | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| json | ≠ ✓ ~ | ≠ ✓ | ≠ ✓ | ≠ ✓ ~ | ≠ ✓ |
| code | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| explain | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |

Reference: `1gpu-nodraft`. = same text as the reference, ≠ different, ✓/✗ the prompt's check, ~ repetitions differ.

- **json**, `1gpu-nodraft`, `1gpu-nodraft-det`, `peer-mtp`: leaves the reference at character 1 of 82
  - reference: `…'{"name": "Elara Vance", "age": 34, "tags": ["astronomer", "no'…`
  - this:      `…'{\n  "name": "Elara Vance",\n  "age": 34,\n  "tags": ["astronome'…`
- **json**, `1gpu-nodraft-det-skip7`: leaves the reference at character 46 of 82
  - reference: `…'{"name": "Elara Vance", "age": 34, "tags": ["astronomer", "novelist", "gardener"]}'…`
  - this:      `…'{"name": "Elara Vance", "age": 34, "tags": ["archivist", "cryptographer", "explorer"]}'…`
- **json**, `peer-mtp-det`: leaves the reference at character 46 of 82
  - reference: `…'{"name": "Elara Vance", "age": 34, "tags": ["astronomer", "novelist", "gardener"]}'…`
  - this:      `…'{"name": "Elara Vance", "age": 34, "tags": ["archaeologist", "cryptographer", "explorer"]}'…`
