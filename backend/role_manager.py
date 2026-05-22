import os
import re
from pathlib import Path

import yaml

ROLES_DIR = Path(__file__).resolve().parent.parent / "config" / "roles"
RVC_BASE = Path(os.environ.get("RVC_MODEL_DIR", str(ROLES_DIR.parent.parent / "models")))


def _parse_role_file(filepath: Path) -> dict | None:
    """解析 YAML frontmatter + 正文的 .md 角色文件。"""
    content = filepath.read_text(encoding="utf-8")

    # 匹配 --- ... --- 之间的 YAML
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not m:
        return None

    frontmatter = yaml.safe_load(m.group(1))
    system_prompt = m.group(2).strip()

    name = frontmatter.get("name", filepath.stem)
    model_cfg = frontmatter.get("model", {})
    rvc_model = frontmatter.get("rvc_model", f"{name}.pth")

    return {
        "name": name,
        "description": frontmatter.get("description", ""),
        "model_name": model_cfg.get("name", "deepseek-r1:7b"),
        "system_prompt": system_prompt,
        "rvc_model_path": str(RVC_BASE / rvc_model) if RVC_BASE else rvc_model,
    }


def get_role_list() -> list[dict]:
    """返回所有角色摘要列表。"""
    roles = []
    if not ROLES_DIR.exists():
        return roles

    for fp in sorted(ROLES_DIR.glob("*.md")):
        parsed = _parse_role_file(fp)
        if parsed:
            roles.append({
                "name": parsed["name"],
                "description": parsed["description"],
            })
    return roles


def load_role(role_name: str) -> dict | None:
    """加载指定角色完整配置。"""
    if not ROLES_DIR.exists():
        return None

    for fp in ROLES_DIR.glob("*.md"):
        parsed = _parse_role_file(fp)
        if parsed and parsed["name"] == role_name:
            return parsed

    return None