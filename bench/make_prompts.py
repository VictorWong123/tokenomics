"""Generate the versioned, original CC0 workload. No external/private text.

These are authored synthetic tasks, not benchmark scores for factual accuracy.
The manifest is committed, so generation is not part of an experiment.
"""

import hashlib
import json
from pathlib import Path

QA = [
    "Why does a hash table sometimes have linear lookup time?",
    "Explain what a database transaction guarantees.",
    "How does a CPU cache differ from main memory?",
    "Why is the sky usually blue?",
    "Explain the difference between TCP and UDP.",
    "How can two threads race even if each write is atomic?",
    "What does a compiler do between parsing and executable code?",
    "Why does indexing a database sometimes make writes slower?",
    "How does a solar panel produce electricity?",
    "What causes the seasons on Earth?",
    "How does public key encryption support secure communication?",
    "What is the distinction between precision and recall?",
    "How do DNS resolvers find an address?",
    "What makes a scientific experiment reproducible?",
    "Explain the role of a write-ahead log after a database crash.",
    "How does binary search work and which precondition does it need?",
    "Why can a distributed lock expire while its holder is still running?",
    "Explain why floating-point addition is not associative.",
    "How does a refrigerator move heat out of its interior?",
    "What does a p-value describe and what does it not establish?",
    "How does a garbage collector discover unreachable objects?",
    "What happens when a virtual-memory page is absent from RAM?",
    "Explain how a Bloom filter can have false positives.",
    "How does a load balancer use health checks?",
    "Why do tides occur?",
    "How can selection bias distort an observational study?",
    "What does idempotency mean for retrying an HTTP request?",
    "Explain the difference between authentication and authorization.",
    "How does a B-tree keep disk lookups efficient?",
    "Why is a confidence interval different from a prediction interval?",
    "What makes a deadlock possible?",
    "How does a transformer use a key-value cache during generation?",
    "What causes an eclipse?",
    "How do checksums help detect corrupted data?",
    "Explain the tradeoff between latency and throughput.",
    "What does backpressure accomplish in a streaming system?",
    "How can a control group help estimate an intervention's effect?",
    "Why do metals conduct electricity?",
    "How does a circuit breaker limit repeated service failures?",
    "Why does an algorithm's asymptotic complexity not determine its actual runtime?",
    "How does the water cycle redistribute fresh water?",
    "Explain why correlation does not identify a causal effect.",
    "What is the role of a memory barrier in concurrent programming?",
    "How does a relational join combine two tables?",
]
CODE = [
    "Return the first index of a target in a sorted integer vector, or -1, using binary search.",
    "Merge two sorted integer arrays while preserving duplicates.",
    "Implement a bounded FIFO queue with enqueue and dequeue returning failure on overflow or underflow.",
    "Parse a comma-separated list of signed integers; reject empty fields and trailing characters.",
    "Implement an LRU cache of integer keys and string values with O(1) get and put.",
    "Count distinct words in UTF-8 input using ASCII whitespace as the delimiter without changing case.",
    "Find the length of the longest substring without repeated ASCII bytes.",
    "Implement a thread-safe monotonic counter and explain its memory-order choice.",
    "Find connected components of an undirected graph given an adjacency list.",
    "Write a stable bottom-up mergesort for integers.",
    "Validate balanced parentheses, square brackets and braces while rejecting other characters.",
    "Compute a moving average over the last w floating-point samples.",
    "Implement a compact bitset with checked set, clear and test operations.",
    "Topologically sort a directed graph and report a cycle if no ordering exists.",
    "Return all overlapping occurrences of a pattern using the prefix-function algorithm.",
    "Implement a fixed-size ring buffer and distinguish full from empty.",
    "Compute shortest nonnegative-weight paths with Dijkstra's algorithm.",
    "Group records by a string key and sum 64-bit integer values with overflow detection.",
    "Implement a disjoint-set union structure with path compression and union by size.",
    "Parse a hexadecimal integer into uint64_t with explicit overflow handling.",
    "Implement a cancellable work queue using a mutex and a condition variable.",
    "Find the smallest window covering all characters of a target string, with multiplicity.",
    "Serialize and deserialize a binary tree using an unambiguous preorder format.",
    "Implement exponentiation by squaring for a uint64_t exponent modulo a positive 32-bit modulus.",
    "Detect a cycle in a singly linked list without allocating a visited set.",
    "Implement a token bucket rate limiter using an injected steady-clock timestamp.",
    "Merge overlapping half-open intervals and preserve gaps at touching endpoints.",
    "Return the median of a stream of integers using two heaps.",
    "Compute edit distance with O(min(m,n)) additional memory.",
    "Implement a scoped file descriptor wrapper with move semantics and no copying.",
    "Split an integer vector into at most k contiguous partitions minimizing the maximum sum.",
    "Implement lower_bound and upper_bound for a sorted vector of signed integers.",
    "Decode a length-prefixed byte stream incrementally when chunks split headers and payloads.",
    "Implement a reusable barrier for a fixed number of threads.",
    "Find bridges in an undirected graph, correctly handling parallel edges.",
    "Compute a rolling hash for byte windows and describe collision limits.",
    "Implement a tiny JSON string escaper including control characters.",
    "Implement a bounded retry loop with exponential backoff using an injected sleeper.",
    "Build a trie supporting insertion, exact lookup and prefix lookup over lowercase ASCII.",
    "Implement checked multiplication for two signed 64-bit integers.",
    "Produce a deterministic breadth-first traversal with neighbors visited in ascending order.",
    "Write a scope-exit guard that runs a callable once unless released.",
    "Deduplicate sorted records by key while keeping the newest timestamp.",
]


def document(i, long):
    towns = [
        "Alder",
        "Birch",
        "Cedar",
        "Dune",
        "Elm",
        "Fern",
        "Grove",
        "Harbor",
        "Iris",
        "Juniper",
        "Kestrel",
    ]
    topic = [
        "library lending",
        "water monitoring",
        "bus reliability",
        "community gardens",
        "building insulation",
        "school meals",
        "recycling collection",
        "clinic appointments",
    ][i % 8]
    name = f"{towns[i%len(towns)]} district {i+1}"
    before = 120 + i * 7
    after = before - 19 - (i % 9)
    budget = 28000 + i * 137
    paragraphs = [
        f"The {name} council reviewed a six-month {topic} pilot in an open meeting. This is an original hypothetical case, not a report about an actual town. The pilot began in March and ended in August. Staff recorded {before} delayed service cases during the first month and {after} in the last month. The approved operating budget was ${budget:,}. The report separates observations from explanations and does not claim the pilot caused every change.",
        f"Three neighborhoods participated. The central neighborhood received the new procedure first, the river neighborhood joined one month later, and the hill neighborhood retained the old procedure until the final month. Enrollment was voluntary. Staff said that participants who volunteered early had more time to complete forms, which may make their experiences unrepresentative. Records from two days were lost during a power outage and were marked missing rather than filled with estimated values.",
        f"An internal survey collected {83+i} responses. Respondents valued predictable opening hours and clear status updates. Several requested a telephone option because the online form was difficult to use on older devices. Survey respondents could submit only one response per household, but the council had no reliable way to identify households that moved during the pilot. The survey was used to identify practical problems, not to estimate the views of every resident.",
        f"The finance team reported that training and translation used most of the initial setup allowance. Recurring staff time was expected to matter more than printing costs in a larger rollout. The budget excludes volunteer labor, building depreciation and costs paid by partner organizations. A proposal to install extra sensors was deferred because procurement would extend beyond the pilot. Staff warned that comparing this partial accounting with a fully costed commercial service would be misleading.",
        f"The council approved a three-month extension rather than a permanent rollout. Staff must publish weekly counts, record missing data, and offer paper and telephone channels. The next review will compare the same calendar months to reduce seasonal confusion. One member requested an independent audit; another argued that the small sample could not support that expense yet. The decision preserves the existing service in neighborhoods that have not joined. No resident is required to participate.",
    ]
    if long:
        paragraphs += [
            f"Implementation notes describe a staffing disruption in week seven, when two trained employees were absent and temporary staff handled intake. Processing times rose during that week. The team kept those observations in the final table and added an annotation. Removing the week would make the trend look smoother but would hide a plausible operating risk. The report recommends cross-training at least two additional people before adding a fourth neighborhood. It does not quantify how much cross-training would improve reliability.",
            f"A local advisory group proposed publishing anonymized aggregate records, but individual case notes remain confidential. Aggregation will combine small cells to avoid revealing personal circumstances. The planned public file contains dates, district totals and missing-data flags. It will not include names, addresses or contact details. The council asked staff to publish a data dictionary with every release so that later analysts can distinguish a zero from a day with no usable records.",
            f"Several constraints remain unresolved. The hill neighborhood has fewer public transport connections, the river neighborhood has intermittent broadband coverage, and the central office has longer staffed hours. These differences may explain part of the observed variation. The next phase will record opening hours alongside service counts. The report explicitly says that no randomized allocation took place. Readers should avoid treating differences between neighborhoods as a controlled estimate of the procedure's effect.",
            f"The final recommendation is to continue carefully, monitor access, and revisit the decision with a more complete cost record. Staff will report whether participation changes after a telephone channel becomes available. They will also document any change in the definition of a delayed case. The chair asked that future summaries include both the improvement in raw counts and the limitations of the evidence. A successful extension would establish operational practicality; it would still not prove the best design for every district.",
        ]
    return "\n\n".join(paragraphs)


def build():
    rows = []
    for category, n_dev in [("qa", 14), ("code", 13), ("summary", 13)]:
        for i in range(n_dev + 30):
            split = "development" if i < n_dev else "heldout"
            long = i % 2 == 1
            if category == "qa":
                task = QA[i] + (
                    " Give a careful explanation with a concrete example, assumptions, failure cases, and a short worked comparison."
                    if long
                    else " Answer clearly in a short paragraph."
                )
            elif category == "code":
                task = f"Write C++20 code for this task: {CODE[i]}"
                if long:
                    task += " Include explicit invariants, edge-case handling, a short usage example, and time/space complexity. Inputs may be empty or contain duplicate values. State any preconditions and avoid undefined behavior. Use only the standard library. Explain how the implementation handles failed operations."
                else:
                    task += (
                        " Give the implementation and a brief complexity explanation."
                    )
            else:
                task = (
                    "Summarize the source below. Preserve the decision, quantitative findings, limitations, and next steps. Do not infer causation or invent missing facts.\n\nSOURCE:\n"
                    + document(i, long)
                )
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant. Answer the user directly.<|im_end|>\n<|im_start|>user\n"
                + task
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            rows.append(
                {
                    "id": f'{split}-{category}-{i if split=="development" else i-n_dev:02d}',
                    "split": split,
                    "category": category,
                    "length_bucket": "long" if long else "short",
                    "pilot": split == "development" and i < 6,
                    "source": "Tokenomics original synthetic workload v1; AI-assisted authorship",
                    "license": "CC0-1.0",
                    "prompt": prompt,
                    "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
    return rows


if __name__ == "__main__":
    path = Path(__file__).with_name("prompts-v1.jsonl")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build()))
    print(path)
