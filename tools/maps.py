"""Validate node metadata and rebuild recursive maps using only the stdlib.

NODE.yaml uses the JSON subset of YAML so no YAML dependency is necessary.
Edit node metadata, then run this tool with --write. CI/review uses --check.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

STATES = {
    "development": {"inactive", "active", "paused", "maintenance", "retired"},
    "implementation": {"unassessed", "planned", "skeleton", "partial", "complete"},
    "maturity": {None, "experimental", "candidate", "stable"},
    "verification": {"unknown", "not_run", "passed", "failed", "stale"},
    "integration": {None, "not_merged", "partial", "merged"},
}
MAPS = ("ROADMAP.md", "CODEMAP.md", "APIMAP.md")
TASK_STATES = {"planned", "active", "blocked", "complete", "cancelled"}
LOCAL_DIRS = {"__pycache__", "logs", "results", "artifacts", "cache", "downloads", "weights", "build", "dist", "venv", "node_modules"}


def directories(root: Path) -> list[str]:
    result = []
    for base, dirs, _ in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in LOCAL_DIRS)
        result.append(Path(base).relative_to(root).as_posix())
    return sorted(result)


def load_nodes(root: Path, dirs: list[str]) -> dict:
    nodes = {}
    ids = set()
    tasks = set()
    for directory in dirs:
        path = root / directory / "NODE.yaml"
        if not path.exists():
            continue
        node = json.loads(path.read_text(encoding="utf-8"))
        if node["schema_version"] != 1:
            raise ValueError(f"{directory}: unsupported schema")
        if node["id"] in ids:
            raise ValueError(f"{directory}: duplicate node ID")
        ids.add(node["id"])
        for key, allowed in STATES.items():
            if node[key] not in allowed:
                raise ValueError(f"{directory}: invalid {key}")
        if node["branch"]["status"] not in {"planned", "present", "archived"}:
            raise ValueError(f"{directory}: invalid branch status")
        if node['verification'] == 'passed' and not node.get('verification_scope'):
            raise ValueError(f"{directory}: verification scope is required")
        if "future" in node["flags"] and node["development"] == "active":
            raise ValueError(f"{directory}: future node must be explicitly activated")
        for task in node["tasks"]:
            if task["id"] in tasks or task["state"] not in TASK_STATES:
                raise ValueError(f"{directory}: duplicate task or invalid task state")
            tasks.add(task["id"])
        nodes[directory] = node
    if "." not in nodes:
        raise ValueError("Missing root node")
    for directory, node in nodes.items():
        if any(task_id not in tasks for task_id in node['blocked_by']):
            raise ValueError(f"{directory}: unknown blocking task")
    for directory, node in nodes.items():
        if directory == ".":
            if node["parent"] is not None or node["branch"]["parent"] is not None:
                raise ValueError("Root must not have a parent")
            continue
        ancestor = Path(directory).parent
        while ancestor.as_posix() not in nodes:
            ancestor = ancestor.parent
        parent = ancestor.as_posix()
        if node["parent"] != parent or node["branch"]["parent"] != nodes[parent]["branch"]["name"]:
            raise ValueError(f"{directory}: parent mismatch")
        for dependency in node["depends_on"]:
            if dependency not in nodes:
                raise ValueError(f"{directory}: unknown dependency {dependency}")
    branches = [n["branch"]["name"] for n in nodes.values()]
    if len(set(branches)) != len(branches):
        raise ValueError("Duplicate branch name")
    if any(a != b and b.startswith(a + "/") for a in branches for b in branches):
        raise ValueError("Git branch file/directory conflict")
    visiting, visited = set(), set()

    def visit(path):
        if path in visiting:
            raise ValueError(f"Dependency cycle at {path}")
        if path in visited:
            return
        visiting.add(path)
        for dependency in nodes[path]["depends_on"]:
            visit(dependency)
        visiting.remove(path)
        visited.add(path)
    for path in nodes:
        visit(path)
    return nodes


def link(directory: str, destination: str, label: str) -> str:
    return f"[{label}]({Path(os.path.relpath(destination, directory)).as_posix()})"


def render(root: Path) -> dict[str, str]:
    dirs = directories(root)
    nodes = load_nodes(root, dirs)
    result = {}
    for directory in dirs:
        ancestor = Path(directory)
        while ancestor.as_posix() not in nodes:
            ancestor = ancestor.parent
        owner = ancestor.as_posix()
        node = nodes[owner]
        children = sorted((p for p in nodes if nodes[p]["parent"] == directory), key=lambda p: nodes[p]["order"])
        direct_dirs = [p for p in dirs if p != "." and Path(p).parent.as_posix() == directory]
        heading = f"# {node['title']} / {directory}\n\n"
        header = "Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.\n\n"
        header += f"Nodo responsabile: {link(directory, owner + '/NODE.yaml', node['id'])}. "
        header += f"Ripresa: {link(directory, 'START_HERE.md', 'START_HERE')}.\n\n"
        if directory == owner:
            roadmap = f"{node['purpose']}\n\n"
            roadmap += f"Sviluppo: **{node['development']}**; codice: **{node['implementation']}**; "
            roadmap += f"maturita: {node['maturity'] or 'non assegnata'}; verifica: {node['verification']}; integrazione: {node['integration'] or 'sconosciuta'}.\n\n"
            roadmap += f"Flag: {', '.join(node['flags']) or 'nessuno'}. Blocchi: {', '.join(node['blocked_by']) or 'nessuno registrato'}.\n\n"
            if node.get('verification_scope'):
                roadmap += f"Ambito verifica: {node['verification_scope']}\n\n"
            roadmap += "## Attivita locali\n\n"
            roadmap += "\n".join(f"{i}. `{t['id']}` [{t['state']}] {t['description']}" for i, t in enumerate(node['tasks'], 1)) + "\n\n"
            if children:
                roadmap += "## Componenti (ordine locale, ID stabili)\n\n"
                roadmap += "\n".join(f"{i}. {link(directory, p+'/ROADMAP.md', nodes[p]['id'] + ': ' + nodes[p]['title'])} — {nodes[p]['development']}; {nodes[p]['implementation']}; {nodes[p]['verification']}." for i, p in enumerate(children, 1)) + "\n\n"
            if node["parent"] is not None:
                roadmap += "Propagare impatti e blocchi al " + link(directory, node['parent']+'/ROADMAP.md', 'padre') + " e proseguire fino alla radice.\n"
            codemap = f"Scopo: {node['purpose']}\n\n"
            b = node['branch']
            codemap += f"Branch: `{b['name']}` ({b['status']}); padre Git previsto: `{b['parent'] or 'nessuno'}`.\n\n"
            codemap += "La presenza dello scaffold non implica una implementazione runtime.\n\n"
            api = "## Contratto dichiarato\n\n" + "\n".join(f"- `{a}`" for a in node['api']) + "\n\n"
            api += "Dipendenze: " + (", ".join(link(directory, p+'/APIMAP.md', p) for p in node['depends_on']) or 'nessuna') + ".\n\n"
            if children:
                api += "## Contratti delegati\n\n" + "\n".join("- "+link(directory,p+'/APIMAP.md',nodes[p]['title']) for p in children) + "\n"
        else:
            lane = Path(directory).name
            meanings = {'runtime':'implementazioni eseguibili', 'tests':'test e fixture sintetiche', 'benchmarks':'codice di benchmark; risultati esclusi da Git', 'tools':'strumenti di sviluppo e ispezione'}
            purpose = meanings.get(lane, 'raggruppamento strutturale')
            roadmap = f"Corsia: {purpose}. Le priorita appartengono al " + link(directory,owner+'/ROADMAP.md','nodo responsabile') + ".\n\nNessuna capacita applicativa dedotta dalla presenza di questa cartella.\n"
            codemap = f"Contenuto riservato a: {purpose}.\n\n"
            api = "Nessun contratto autonomo. Fare riferimento a " + link(directory,owner+'/APIMAP.md','APIMAP del responsabile') + ".\n"
        if direct_dirs:
            codemap += "## Cartelle\n\n" + "\n".join("- " + link(directory,p+'/CODEMAP.md',Path(p).name) for p in direct_dirs) + "\n\n"
            if directory != owner:
                roadmap += "\n## Sottocartelle\n\n" + "\n".join("- "+link(directory,p+'/ROADMAP.md',Path(p).name) for p in direct_dirs)+"\n"
        files = sorted(p.name for p in (root/directory).iterdir() if p.is_file() and p.name not in MAPS and not p.name.startswith('.'))
        if files:
            codemap += "## File\n\n" + "\n".join(f"- [{name}]({name})" for name in files)+"\n"
        for filename, body in zip(MAPS, (roadmap, codemap, api)):
            result[(Path(directory)/filename).as_posix()] = heading+header+body
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.write and args.check:
        parser.error('choose --write or --check')
    root = Path(__file__).resolve().parents[1]
    try:
        generated = render(root)
    except (ValueError, KeyError, TypeError) as error:
        print(error)
        return 1
    stale = []
    for relative, content in generated.items():
        path = root/relative
        if args.write:
            path.write_text(content, encoding='utf-8')
        elif not path.exists() or path.read_text(encoding='utf-8') != content:
            stale.append(relative)
    if stale:
        print('Mappe da aggiornare: ' + ', '.join(stale))
        return 1
    print(f'Mappe coerenti: {len(generated)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
