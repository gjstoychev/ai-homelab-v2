| prompt | fp8-nomtp | fp8-mtp4 | nvfp4-mtp4 | basalt-split-nodraft | basalt-split |
|---|---|---|---|---|---|
| arith | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| capital | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| primes | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| sort | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| time | = ✓ | = ✓ | ≠ ✓ | = ✓ | = ✓ |
| author | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| german | = ✓ | = ✓ | = ✓ | = ✓ | = ✓ |
| json | = ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ |
| code | = ✓ | ≠ ✓ | ≠ ✓ | = ✓ | = ✓ |
| explain | = ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ | ≠ ✓ |

Reference: `fp8-nomtp`. = same text as the reference, ≠ different, ✓/✗ the prompt's check, ~ repetitions differ.

- **time**, `nvfp4-mtp4`: leaves the reference at character 12 of 41
  - reference: `…'The journey lasts 2 hours and 35 minutes.'…`
  - this:      `…'The journey is 2 hours and 35 minutes long.'…`
- **json**, `fp8-mtp4`: leaves the reference at character 14 of 107
  - reference: `…'{\n  "name": "Elias Thorne",\n  "age": 34,\n  "tags": [\n    "astronomer",\n   '…`
  - this:      `…'{\n  "name": "Evelyn Thorne",\n  "age": 34,\n  "tags": [\n    "astronomer",\n  '…`
- **json**, `nvfp4-mtp4`: leaves the reference at character 14 of 107
  - reference: `…'{\n  "name": "Elias Thorne",\n  "age": 34,\n  "tags": [\n    "astronomer",\n   '…`
  - this:      `…'{\n  "name": "Evelyn Thorne",\n  "age": 34,\n  "tags": [\n    "architect",\n   '…`
- **json**, `basalt-split`, `basalt-split-nodraft`: leaves the reference at character 15 of 107
  - reference: `…'{\n  "name": "Elias Thorne",\n  "age": 34,\n  "tags": [\n    "astronomer",\n    '…`
  - this:      `…'{\n  "name": "Elena Voss",\n  "age": 34,\n  "tags": [\n    "astronomer",\n    "p'…`
- **code**, `fp8-mtp4`, `nvfp4-mtp4`: leaves the reference at character 46 of 123
  - reference: `…'```python\ndef is_palindrome(s):\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == '…`
  - this:      `…"```python\ndef is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return clea"…`
- **explain**, `fp8-mtp4`: leaves the reference at character 9 of 454
  - reference: `…'Sunlight contains all colors of the visible spectrum, but when it ent'…`
  - this:      `…'Sunlight enters the atmosphere and scatters off gas molecules, a proc'…`
- **explain**, `nvfp4-mtp4`: leaves the reference at character 93 of 454
  - reference: `…"he visible spectrum, but when it enters Earth's atmosphere, it interacts with gas molecules that scatter shorter wavelen"…`
  - this:      `…"he visible spectrum, but when it enters Earth's atmosphere, shorter wavelengths like blue scatter much more than longer "…`
- **explain**, `basalt-split`, `basalt-split-nodraft`: leaves the reference at character 96 of 454
  - reference: `…"visible spectrum, but when it enters Earth's atmosphere, it interacts with gas molecules that scatter shorter wavelength"…`
  - this:      `…"visible spectrum, but when it enters Earth's atmosphere, it collides with gas molecules. These molecules scatter shorter"…`
