"""Independent PyTorch composition of standard Qwen3 + the declared W8A8 scheme.

Uses Torch integer GEMM, RMSNorm math, trigonometry, softmax and SiLU. No Rust
operators supply expected values. Run in the GPU workstation's Torch environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ap=argparse.ArgumentParser(__doc__)
ap.add_argument("--model",type=Path,required=True)
ap.add_argument("--request",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
manifest_bytes=(args.model/"manifest.json").read_bytes()
manifest=json.loads(manifest_bytes)
request=json.loads(args.request.read_text())
assert manifest["profile"]=="qwen3-8b-int8-channel-token-bf16-softmax-v1"
assert len(request["prompt_ids"])+request["max_tokens"]<=4096

def load(name):
    info=manifest["tensors"][name]
    path=args.model/f"{name}.bin"
    h=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(8<<20): h.update(b)
    assert h.hexdigest()==info["sha256"],name
    a=np.memmap(path,dtype=info["dtype"],mode="c",shape=tuple(info["shape"]))
    t=torch.from_numpy(a)
    if info["dtype"]=="<u2": t=t.view(torch.bfloat16)
    return t.cuda()

weights=[]
for layer in range(36):
    weights.append({name:load(f"layer.{layer}.{name}") for name in
                    ["norm1","norm2","qnorm","knorm"]+
                    [f"{p}.{s}" for p in ["q","k","v","o","gate","up","down"] for s in ["w","scale"]]})
emb=load("embedding");head=load("head");gain=load("final_gain")
freq=1.0/(manifest["config"]["rope_theta"]**(torch.arange(0,128,2,dtype=torch.float32,device="cuda")/128))
cache=[None]*36

def norm(x,w):
    f=x.float()
    return (f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-6)*w).bfloat16()

def linear(x,w,s):
    xf=x.float();scale=xf.abs().amax(-1,keepdim=True).clamp(min=1e-8)/127
    q=torch.round(xf/scale).clamp(-127,127).to(torch.int8)
    # Padding only supplies independent zero rows; it cannot change the dots.
    padded=F.pad(q,(0,0,0,max(32,((len(q)+31)//32)*32)-len(q)))
    dot=torch._int_mm(padded,w.t())[:len(q)]
    return ((dot.float()*scale)*s).bfloat16()

def rope(x,position):
    angles=torch.arange(position,position+x.shape[0],dtype=torch.float32,device="cuda")[:,None]*freq[None,:]
    c=angles.cos()[:,None,:];s=angles.sin()[:,None,:]
    a,b=x.float().chunk(2,dim=-1)
    return torch.cat([a*c-b*s,b*c+a*s],dim=-1).bfloat16()

traces={}
def save(step,name,value):
    a=value.float().cpu().numpy().astype("<f4")
    data=a.tobytes();filename=f"step.{step}.{name}.bin"
    (args.output/filename).write_bytes(data)
    traces[filename]=dict(shape=list(a.shape),sha256=hashlib.sha256(data).hexdigest())

ids=request["prompt_ids"];position=0;generated=[]
with torch.inference_mode():
    for step in range(request["max_tokens"]):
        x=emb[ids]
        for layer,w in enumerate(weights):
            z=norm(x,w["norm1"])
            if layer==4: save(step,"debug.norm",z)
            q=linear(z,w["q.w"],w["q.scale"]); k=linear(z,w["k.w"],w["k.scale"]); v=linear(z,w["v.w"],w["v.scale"])
            if layer==4:
                for name,value in [("q",q),("k",k),("v",v)]: save(step,"debug."+name,value)
            q=rope(norm(q.reshape(-1,32,128),w["qnorm"]),position)
            k=rope(norm(k.reshape(-1,8,128),w["knorm"]),position)
            v=v.reshape(-1,8,128)
            if layer==4:
                save(step,"debug.qpost",q);save(step,"debug.kpost",k)
            if cache[layer] is not None:
                k=torch.cat([cache[layer][0],k]);v=torch.cat([cache[layer][1],v])
            cache[layer]=(k,v)
            kh=k.repeat_interleave(4,dim=1).transpose(0,1).float()
            vh=v.repeat_interleave(4,dim=1).transpose(0,1).float()
            score=(q.transpose(0,1).float()@kh.transpose(1,2))*(128**-0.5)
            visible=torch.arange(len(k),device="cuda")[None,:] <= (position+torch.arange(len(x),device="cuda"))[:,None]
            probability=score.masked_fill(~visible[None,:,:],float('-inf')).softmax(-1)
            a=(probability@vh).transpose(0,1).contiguous().reshape(-1,4096).bfloat16()
            o=linear(a,w["o.w"],w["o.scale"])
            if layer==4:
                save(step,"debug.a",a);save(step,"debug.o",o)
            x=(x.float()+o.float()).bfloat16()
            z=norm(x,w["norm2"])
            if layer==4: save(step,"debug.norm2",z)
            gate=linear(z,w["gate.w"],w["gate.scale"])
            up=linear(z,w["up.w"],w["up.scale"])
            a=(F.silu(gate.float())*up.float()).bfloat16()
            if layer==4:
                save(step,"debug.gate",gate);save(step,"debug.up",up);save(step,"debug.act",a)
            down=linear(a,w["down.w"],w["down.scale"])
            if layer==4: save(step,"debug.down",down)
            x=(x.float()+down.float()).bfloat16()
            save(step,f"layer.{layer}",x)
        z=norm(x,gain);save(step,"norm",z)
        logits=(z[-1:].float()@head.float().t()).bfloat16().flatten()
        save(step,"logits",logits)
        next_id=int(logits.float().argmax());generated.append(next_id)
        print(step,next_id,flush=True)
        if next_id in request["stop_ids"]:break
        position+=len(ids);ids=[next_id]
(args.output/"request.json").write_text(json.dumps(request))
(args.output/"reference.json").write_text(json.dumps(dict(model_root=hashlib.sha256(manifest_bytes).hexdigest(),torch=torch.__version__,cuda=torch.version.cuda,token_ids=generated,traces=traces),indent=2)+"\n")
