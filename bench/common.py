"""Shared artifact primitives. No network credentials are recorded."""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def atomic_json(path, obj):
    path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with open(tmp,"w") as f:
        json.dump(obj,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def read_jsonl(path):
    with open(path) as f: return [json.loads(x) for x in f if x.strip()]

def command(args):
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return {"returncode":p.returncode,"output":p.stdout.strip()}

def text_file(path):
    try: return pathlib.Path(path).read_text().strip()
    except OSError: return None

def hardware():
    result={"platform":platform.platform(),"machine":platform.machine(),"cpu_count":os.cpu_count(),
            "timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "execution_environment":"codespaces" if os.getenv("CODESPACES")=="true" else "macos" if platform.system()=="Darwin" else "linux-other",
            "compiler":command(["c++","--version"]),
            "git_revision":command(["git","rev-parse","HEAD"])["output"],
            "git_dirty":command(["git","status","--porcelain","--untracked-files=no"])["output"],
            "llama_revision":command(["git","-C",str(ROOT/"third_party/llama.cpp"),"rev-parse","HEAD"])["output"]}
    if platform.system()=="Darwin":
        result["sysctl"]=command(["sysctl","hw.memsize","hw.model","machdep.cpu.brand_string"])
        result["power"]=command(["pmset","-g"])
        result["memory"]=command(["vm_stat"])
        result["swap"]=command(["sysctl","vm.swapusage"])
        result["thermal"]=command(["pmset","-g","therm"])
    else:
        result["lscpu"]=command(["lscpu"])
        result["meminfo"]=text_file("/proc/meminfo")
        result["cpu_quota"]=text_file("/sys/fs/cgroup/cpu.max")
        result["memory_limit"]=text_file("/sys/fs/cgroup/memory.max")
        result["memory_events"]=text_file("/sys/fs/cgroup/memory.events")
        result["memory_pressure"]=text_file("/proc/pressure/memory")
        result["loadavg"]=text_file("/proc/loadavg")
        result["vmstat"]=text_file("/proc/vmstat")
    return result

def available_memory():
    """Conservative available RAM under the process cgroup, bytes."""
    if platform.system()=="Darwin":
        # Physical capacity is not free memory; vm_stat free+inactive is a
        # conservative screening estimate. Benchmark RSS/swap still gate runs.
        import re
        vm=command(["vm_stat"])["output"]
        page=int(re.search(r"page size of (\d+)",vm).group(1))
        counts={k:int(v) for k,v in re.findall(r"([^\n:]+):\s+(\d+)\.",vm)}
        return page*sum(counts.get(k,0) for k in ("Pages free","Pages inactive","Pages speculative"))
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines()}
    available=mem['MemAvailable']
    limit=text_file('/sys/fs/cgroup/memory.max');current=text_file('/sys/fs/cgroup/memory.current')
    if limit and limit.isdigit() and current and current.isdigit(): available=min(available,int(limit)-int(current))
    return available
