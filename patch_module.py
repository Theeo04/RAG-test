# tools/patch_module.py
import os, re, pathlib, json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import yaml
from copy import deepcopy

# -------- utilitare micuțe --------
def parse_simple_value(v: str):
    s = v.strip()
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    if re.fullmatch(r"\d+", s):  # int
        return int(s)
    if re.fullmatch(r"\d+\.\d+", s):  # float
        try:
            return float(s)
        except:
            pass
    # păstrează stringul (unități k8s: 200m, 256Mi etc)
    return s

def dotted_to_nested(dotted: str, value):
    """
    Transformă 'a.b.c=1' într-un dict imbricat {'a':{'b':{'c':1}}}
    Suport minimal pentru liste: key[].
    """
    root: Dict[str, Any] = {}
    cur = root
    parts = dotted.split(".")
    for i, p in enumerate(parts):
        m = re.match(r"^([A-Za-z0-9_-]+)(\[\])?$", p)
        if not m:
            # cheie exotică; cădere simplă
            key = p
            if i == len(parts)-1:
                cur[key] = value
            else:
                cur = cur.setdefault(key, {})
            continue
        key, is_list = m.group(1), bool(m.group(2))
        if i == len(parts)-1:
            if is_list:
                cur.setdefault(key, []).append(value)
            else:
                cur[key] = value
        else:
            if is_list:
                cur = cur.setdefault(key, [])
                if not cur or not isinstance(cur[-1], dict):
                    cur.append({})
                cur = cur[-1]
            else:
                cur = cur.setdefault(key, {})
    return root

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a

def load_values_yaml(path: str) -> Dict[str, Any]:
    """
    Load a values YAML file (possibly multi-doc) and fold mapping docs together.
    Non-mapping docs are ignored. Returns {} if empty.
    """
    base: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f)) or []
    except Exception:
        docs = []
    for d in docs:
        if isinstance(d, dict):
            deep_merge(base, d)
    return base

# -------- selectare fișier values pentru o cheie --------
def best_values_file_for_key(values_files: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """
    Alege fișierul values „cel mai potrivit” pentru cheia dată.
    Heuristic: cel mai lung prefix de cheie din candidate_keys; +1 punct dacă profilul e 'common'.
    """
    best = None
    best_score = -1
    for vf in values_files:
        keys = vf.get("candidate_keys") or []
        score = 0
        for ck in keys:
            # compatibilitate simplă de prefix (exact sau cu '.')
            pref = os.path.commonprefix([ck, key])
            if pref and (pref.endswith(".") or pref == ck or pref == key):
                score = max(score, len(pref))
        if (vf.get("profile") or "").lower() == "common":
            score += 1
        if score > best_score:
            best_score = score
            best = vf
    return best or values_files[0]

# -------- motorul de patch --------
def build_patches_from_intents(values_files: List[Dict[str, Any]],
                               intents: List[Dict[str, Any]],
                               out_dir: str) -> Dict[str, Any]:
    """
    intents: listă cu elemente de forma:
      {"op":"enable","key":"istio.enabled"}
      {"op":"disable","key":"feature.x"}
      {"op":"set","key":"resources.requests.cpu","value":"200m"}
    Acceptă sinonime pentru op: add/update/change/patch -> set, on/turn_on/activate -> enable, off/turn_off/deactivate -> disable
    """
    patches_per_file: Dict[str, Dict[str, Any]] = defaultdict(dict)
    new_keys_per_file: Dict[str, List[str]] = defaultdict(list)
    routed: List[Dict[str, Any]] = []

    for it in intents:
        op_raw = it.get("op")
        key = it.get("key")
        if not op_raw or not key:
            raise ValueError(f"Intent invalid (lipsește op/key): {it}")

        op = _normalize_op(op_raw)

        if op == "enable":
            val = True
        elif op == "disable":
            val = False
        elif op == "set":
            if "value" not in it:
                # Conveniență: dacă e cheie *.enabled fără value, presupunem True
                if str(key).lower().endswith(".enabled"):
                    val = True
                else:
                    raise ValueError(f"Intent 'set' fără value: {it}")
            else:
                rawv = it["value"]
                if isinstance(rawv, (dict, list, int, float, bool)) or rawv is None:
                    val = rawv
                else:
                    val = parse_simple_value(str(rawv))
        else:
            raise ValueError(f"Op necunoscut: {op_raw} (normalizat: {op}). Folosește una din: set/enable/disable")

        vf = best_values_file_for_key(values_files, key)
        nested = dotted_to_nested(key, val)
        deep_merge(patches_per_file[vf["path"]], nested)

        ck = vf.get("candidate_keys") or []
        if key not in ck and not any(key.startswith(pfx + ".") for pfx in ck):
            new_keys_per_file[vf["path"]].append(key)

        routed.append({"key": key, "op": op, "value": val if op == "set" else None,
                       "values_path": vf["path"], "profile": vf.get("profile")})

    # scrie pe disc
    outp = pathlib.Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    patches_out = []
    apply_cmds = []
    final_merged_parts: List[str] = []
    for i, (vpath, pdata) in enumerate(patches_per_file.items(), start=1):
        profile_here = None
        for vf in values_files:
            if vf["path"] == vpath:
                profile_here = vf.get("profile")
                break
        patch_path = outp / f"patch_{i}_{profile_here or 'values'}.yaml"
        with open(patch_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(pdata, f, sort_keys=False)

        # Build merged YAML (original + patch) for copy-paste
        orig = load_values_yaml(vpath)
        merged = deepcopy(orig)
        deep_merge(merged, pdata)
        merged_yaml = yaml.safe_dump(merged, sort_keys=False)

        patches_out.append({
            "values_path": vpath,
            "profile": profile_here,
            "patch_yaml_path": str(patch_path),
            "new_keys": sorted(set(new_keys_per_file.get(vpath, []))),
            "merged_preview": merged_yaml
        })
        final_merged_parts.append(f"# values: {vpath} (profile: {profile_here or 'common'})\n{merged_yaml}".rstrip())

        apply_cmds.append(
            f"yq ea '. as $item ireduce ({{}}; . *+ $item)' {vpath} {patch_path} > {out_dir}/merged_{profile_here or 'values'}.yaml"
        )

    final_merged_yaml = "\n---\n".join(final_merged_parts)
    # Prefer a single-doc YAML when only one file was patched
    if len(patches_out) == 1:
        final_single_yaml = patches_out[0]["merged_preview"]
        is_multidoc = False
    else:
        final_single_yaml = final_merged_yaml  # fallback is multi-doc bundle
        is_multidoc = True

    return {
        "routed_changes": routed,
        "patches": patches_out,
        "apply_commands": apply_cmds,
        "final_merged_yaml": final_merged_yaml,
        "final_single_yaml": final_single_yaml,
        "final_yaml_is_multidoc": is_multidoc,
        "notes": [
            "Cheile din 'new_keys' nu există explicit și vor fi create de patch.",
            "Verifică merged_*.yaml înainte de commit/apply."
        ]
    }

# -------- integrare cu plan --------
def patch_from_plan(plan: Dict[str, Any],
                    intents: List[Dict[str, Any]],
                    out_dir: str) -> Dict[str, Any]:
    target = plan.get("target") or {}
    comp = target.get("component")
    if not comp:
        raise RuntimeError("Plan fără componentă țintă.")
    values_files = plan.get("values_files") or []
    if not values_files:
        raise RuntimeError(f"Componenta '{comp}' nu are values files în plan.")
    return build_patches_from_intents(values_files, intents, out_dir)