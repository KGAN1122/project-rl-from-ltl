from ltlf2dfa.parser.ltlf import LTLfParser
import re
from itertools import chain, combinations

def extract_ap(ltl: str):
    ops = {'X', 'F', 'G', 'U', 'R', 'W', '!', '&', '|', '->', 'true', 'false'}
    tokens = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', ltl))
    return tokens - ops

def parse_dot_dfa(dot_str):
    states = set()
    accepting = set()
    transitions = {}
    initial_state = None

    lines = dot_str.splitlines()
    for line in lines:
        line = line.strip()
        if '->' in line and 'label' in line:
            src, rest = line.split('->')
            src = int(src.strip())
            tgt = int(re.search(r'(\d+)', rest).group(1))
            label = re.search(r'label\s*=\s*"?([^"]+)"?', line).group(1).strip()
            ap = tuple(sorted(label.replace('"', '').split(','))) if label else tuple()
            states.update([src, tgt])
            if src not in transitions:
                transitions[src] = {}
            transitions[src][ap] = tgt
        elif 'init ->' in line:
            initial_state = int(re.search(r'->\s*(\d+)', line).group(1))
        elif 'doublecircle' in line:
            accepting = set(int(float(s)) for s in re.findall(r'(\d+(?:\.\d+)?)', line))

    if not transitions:
        print("[Warning] DFA contains no transitions.")
        q0 = initial_state if initial_state is not None else 0
        n_qs = q0 + 2
        delta = [{(): n_qs-1} for _ in range(n_qs)]
        acc = [{(): [False]} for _ in range(n_qs)]
        eps = [[] for _ in range(n_qs)]
        return q0, delta, acc, eps, (1, n_qs)

    ap_all = set(x for t in transitions.values() for label in t.keys() for x in label)
    ap_list = [tuple(ap) for ap in chain.from_iterable(combinations(sorted(ap_all), k) for k in range(len(ap_all)+1))]
    n_qs = max(states) + 2
    delta = [{ap: n_qs-1 for ap in ap_list} for _ in range(n_qs)]
    acc = [{ap: [False] for ap in ap_list} for _ in range(n_qs)]
    eps = [[] for _ in range(n_qs)]

    for src in transitions:
        for ap, tgt in transitions[src].items():
            delta[src][ap] = tgt
            acc[src][ap] = [tgt in accepting]

    return initial_state, delta, acc, eps, (1, n_qs)

# ==== Run Test ====
ltl = "G(a -> X b)"
aps = extract_ap(ltl)
parser = LTLfParser()
formula = parser(ltl)
dot = formula.to_dfa(aps)

print("DOT string:")
print(dot)

q0, delta, acc, eps, shape = parse_dot_dfa(dot)
print("\nParsed DFA Structure:")
print("Initial state:", q0)
print("Shape:", shape)
for i, d in enumerate(delta):
    print(f"delta[{i}]: {d}")
for i, a in enumerate(acc):
    print(f"acc[{i}]: {a}")
