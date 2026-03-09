import json
from utils import score_review

res1 = score_review('Very nice product works good', 5)
res2 = score_review('The battery life on this phone is excellent and it charges very quickly.', 5)

with open('test_output.txt', 'w') as f:
    f.write('---- TEST 1 (Generic Bot) ----\n')
    f.write(json.dumps(res1, indent=2))
    f.write('\n\n---- TEST 2 (Genuine details) ----\n')
    f.write(json.dumps(res2, indent=2))
