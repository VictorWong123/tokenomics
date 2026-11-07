"""Tiny deterministic F32 transformers for state tests, never performance evidence."""
import argparse
import pathlib
import sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
import gguf

def create(path, winner, seed=31, random=False):
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name("Tokenomics synthetic correctness fixture; NOT a language model")
    w.add_context_length(512); w.add_embedding_length(32); w.add_block_count(2)
    w.add_feed_forward_length(64); w.add_head_count(4); w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5); w.add_rope_dimension_count(8)
    w.add_tokenizer_model("llama")
    tokens = ["<unk>","<s>","</s>"] + [f"<0x{i:02X}>" for i in range(256)] + ["▁a", "▁b", "▁"]
    w.add_token_list(tokens); w.add_token_scores([0.] * len(tokens))
    w.add_token_types([2,3,3] + [6]*256 + [1,1,1])
    w.add_unk_token_id(0); w.add_bos_token_id(1); w.add_eos_token_id(2)
    w.add_add_bos_token(True); w.add_add_eos_token(False)
    n = len(tokens)
    if random:
        emb = rng.normal(0,.2,(n,32)).astype(np.float32)
        out = rng.normal(0,.2,(n,32)).astype(np.float32)
    else:
        emb = np.ones((n,32),np.float32)
        out = np.zeros((n,32),np.float32); out[winner] = 1
    w.add_tensor("token_embd.weight",emb); w.add_tensor("output.weight",out)
    w.add_tensor("output_norm.weight",np.ones(32,np.float32))
    for i in range(2):
        for name in ("attn_norm","ffn_norm"):
            w.add_tensor(f"blk.{i}.{name}.weight",np.ones(32,np.float32))
        for name, shape in {"attn_q":(32,32),"attn_k":(32,32),"attn_v":(32,32),"attn_output":(32,32),"ffn_gate":(64,32),"ffn_up":(64,32),"ffn_down":(32,64)}.items():
            x = rng.normal(0,.08,shape).astype(np.float32) if random else np.zeros(shape,np.float32)
            w.add_tensor(f"blk.{i}.{name}.weight",x)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=pathlib.Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for name,win in (("target",259),("wrong",260),("eos",2)):
        create(a.out / f"{name}.gguf",win)
    create(a.out / "random.gguf",259,random=True)
