"""Textual Metal compiler-IR diagnostic, never a GPU correctness certificate."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re

KINDS = ("q4_K", "q6_K", "q5_0", "q8_0")


def functions(module):
    result = {}
    for match in re.finditer(
        r'^define\b[^\n]*@("[^"\n]+"|[-\w.$]+)\([^\n]*\{\n.*?^\}', module, re.M | re.S
    ):
        name = match[1].strip('"')
        if name in result:
            raise ValueError("Duplicate compiler-IR function")
        result[name] = match[0]
    return result


def normalize(body, attributes, *, swap=False):
    header = body.splitlines()[0]
    grids = re.findall(r"<[34] x i32>[^,%]*\s+(%[-\w.$]+)", header)
    if len(grids) != 1:
        raise ValueError("Expected one integer grid-coordinate argument")
    # Metadata is deliberately outside this textual diagnostic's claim.
    body = re.sub(r",?\s*![\w.]+\s+!\d+", "", body)
    if swap:
        grid = re.escape(grids[0])
        pattern = (
            rf"(extractelement\s+<[34] x i32>\s+{grid},\s+i(?:32|64)\s+)([0123])\b"
        )
        shuffle = re.compile(
            rf"^\s*(%[-\w.$]+) = shufflevector <3 x i32> {grid}, <3 x i32> (?:undef|poison), "
            r"<3 x i32> <i32 1, i32 0, i32 2>\s*$",
            re.M,
        )
        shuffles = list(shuffle.finditer(body))
        # A whole-vector use cannot be inverted by changing scalar extracts.
        # Recognize only the exact inverse shuffle observed in the compiler,
        # and reject any other use rather than claiming textual equivalence.
        remaining = "\n".join(body.splitlines()[1:])
        remaining = shuffle.sub("", remaining)
        remaining = re.sub(
            rf"^.*extractelement\s+<[34] x i32>\s+{grid},\s+i(?:32|64)\s+[0123]\b.*$",
            "",
            remaining,
            flags=re.M,
        )
        if re.search(grid + r"(?![-\w.$])", remaining):
            raise ValueError("Unmodeled direct use of the grid coordinate")
        body = re.sub(
            pattern,
            lambda m: m[1] + {"0": "1", "1": "0", "2": "2", "3": "3"}[m[2]],
            body,
        )
        body = shuffle.sub("", body)
        for match in shuffles:
            body = re.sub(re.escape(match[1]) + r"(?![-\w.$])", grids[0], body)
    body = re.sub(r'(@)("[^"\n]+"|[-\w.$]+)(\()', r"\1KERNEL\3", body, count=1)
    # Resolve attribute groups, preserving floating-point optimization flags.
    body = re.sub(
        r"#(\d+)\b", lambda m: attributes.get(m[1], "UNKNOWN_ATTRIBUTE_" + m[1]), body
    )
    names = {}

    def name(value):
        if value not in names:
            names[value] = f"%v{len(names)}"
        return names[value]

    # First-seen renaming keeps SSA uses/definitions and branch labels aligned.
    body = re.sub(
        r"^([-\w.$]+):|%[-\w.$]+",
        lambda m: name("%" + m[1]) + ":" if m[1] is not None else name(m[0]),
        body,
        flags=re.M,
    )
    return " ".join(body.split())


def compare_ir(module):
    definitions = functions(module)
    attributes = dict(re.findall(r"^attributes #(\d+) = (\{[^\n]*\})", module, re.M))
    pairs = []
    for kind in KINDS:
        base = f"kernel_mul_mv_{kind}_f32"
        names = []
        for desired in (base, base + "_interleaved"):
            matches = [name for name in definitions if name == desired]
            if not matches:
                # Some Metal versions retain C++ mangling for unannotated kernels.
                matches = [
                    name
                    for name in definitions
                    if desired in name
                    and ("_interleaved" in name) == desired.endswith("_interleaved")
                    and "_impl" not in name
                ]
            names.append(matches[0] if len(matches) == 1 else None)
        row = {"kind": kind, "symbols": names}
        if None in names:
            row["state"] = "missing_or_ambiguous_function"
        else:
            try:
                row["direct_callees"] = {
                    label: re.findall(
                        r'\bcall\b[^\n]*?@("[^"\n]+"|[-\w.$]+)\(', definitions[symbol]
                    )
                    for label, symbol in zip(("ordinary", "candidate"), names)
                }
                ordinary = normalize(definitions[names[0]], attributes)
                candidate = normalize(definitions[names[1]], attributes, swap=True)
                equal = ordinary == candidate
                row.update(
                    {
                        "state": "matched" if equal else "different",
                        "normalized_text_equal": equal,
                        "ordinary_sha256": hashlib.sha256(
                            ordinary.encode()
                        ).hexdigest(),
                        "candidate_sha256": hashlib.sha256(
                            candidate.encode()
                        ).hexdigest(),
                    }
                )
                if not equal:
                    row["first_token_difference"] = next(
                        (
                            {"index": i, "ordinary": a, "candidate": b}
                            for i, (a, b) in enumerate(
                                zip(ordinary.split(), candidate.split())
                            )
                            if a != b
                        ),
                        {
                            "ordinary_tokens": len(ordinary.split()),
                            "candidate_tokens": len(candidate.split()),
                        },
                    )
                    row["diff_excerpt"] = list(
                        difflib.unified_diff(ordinary.split(), candidate.split(), n=2)
                    )[:30]
            except ValueError as exc:
                row.update({"state": "unsupported_ir_layout", "error": str(exc)})
        pairs.append(row)
    return {
        "state": (
            "normalized_ir_matches"
            if all(p["state"] == "matched" for p in pairs)
            else "diagnostic_incomplete_or_different"
        ),
        "pairs": pairs,
        "parsed_functions": len(definitions),
        "gpu_execution": False,
        "performance_claim": None,
        "normalization": "Invert grid-coordinate extracts or cancel an exact (1,0,2) shuffle on candidate; reject other whole-grid uses; rename SSA/labels and kernel symbol; resolve attribute groups; ignore metadata attachments and whitespace.",
        "limitation": "Textual compiler-IR diagnostic only. Ignored metadata and later function-constant/GPU specialization can matter. It cannot replace actual M3 arithmetic/token gates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output must be fresh")
    result = compare_ir(args.ir.read_text())
    result["ir_sha256"] = hashlib.sha256(args.ir.read_bytes()).hexdigest()
    result["shader_source_sha256"] = hashlib.sha256(
        args.source.read_bytes()
    ).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("METAL_IR_SUMMARY=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
