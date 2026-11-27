| prompt | matched-nomtp | matched-mtp4 | matched-split-nodraft | matched-split |
|---|---|---|---|---|
| arith | = ✓ | = ✓ | = ✓ | = ✓ |
| capital | = ✓ | = ✓ | = ✓ | = ✓ |
| primes | = ✓ | = ✓ | = ✓ | = ✓ |
| sort | = ✓ | = ✓ | = ✓ | = ✓ |
| time | = ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ |
| author | = ✓ | = ✓ | = ✓ | = ✓ |
| german | = ✓ | = ✓ | = ✓ | = ✓ |
| json | = ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ |
| code | = ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ |
| explain | = ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ |

Reference: `matched-nomtp`. = same text as the reference, ≠ different, ✓/✗ the prompt's check, ~ repetitions differ.

- **time**, `matched-mtp4`, `matched-split`, `matched-split-nodraft`: leaves the reference at character 12 of 41
  - reference: `…'The journey lasts 2 hours and 35 minutes.'…`
  - this:      `…'The journey takes 2 hours and 35 minutes.'…`
- **json**, `matched-mtp4`: leaves the reference at character 74 of 102
  - reference: `…'lena Voss",\n  "age": 34,\n  "tags": [\n    "astronomer",\n    "novelist",\n    "hiker"\n  ]\n}'…`
  - this:      `…'lena Voss",\n  "age": 34,\n  "tags": [\n    "astronomer",\n    "jazz pianist",\n    "urban gardener"\n  ]\n}'…`
- **json**, `matched-split`, `matched-split-nodraft`: leaves the reference at character 20 of 102
  - reference: `…'{\n  "name": "Elena Voss",\n  "age": 34,\n  "tags": [\n    "astronomer",\n    "noveli'…`
  - this:      `…'{\n  "name": "Elena Vance",\n  "age": 34,\n  "tags": [\n    "astronomer",\n    "novel'…`
- **code**, `matched-mtp4`, `matched-split`, `matched-split-nodraft`: leaves the reference at character 46 of 133
  - reference: `…"```python\ndef is_palindrome(s):\n    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return c"…`
  - this:      `…'```python\ndef is_palindrome(s):\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == '…`
- **explain**, `matched-mtp4`: leaves the reference at character 58 of 471
  - reference: `…"Sunlight contains all colors of the visible spectrum, but when it enters Earth's atmosphere, it collides with gas mole"…`
  - this:      `…'Sunlight contains all colors of the visible spectrum, but shorter wavelengths like blue and violet scatter more easily'…`
- **explain**, `matched-split`, `matched-split-nodraft`: leaves the reference at character 123 of 471
  - reference: `…"it enters Earth's atmosphere, it collides with gas molecules. These molecules scatter shorter wavelengths, such as blue "…`
  - this:      `…"it enters Earth's atmosphere, it collides with gas molecules and scatters in different directions. Shorter wavelengths, "…`
