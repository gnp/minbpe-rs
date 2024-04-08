import random


def generate_test_case():
    """
    Generate a test case for checking Rust IndexMap preserves insertion order like Python dict.

    Note: The test case in question is in tokenizer.rs.
    """
    stats = {}
    for i in range(20):
        if random.random() < 0.25:
            value = 99
        else:
            value = random.randint(0, 20)
        stats[(i, i)] = value
        print(f"(({i},{i}), {value})")

    pair = max(stats, key=stats.get)
    print(pair)


generate_test_case()
