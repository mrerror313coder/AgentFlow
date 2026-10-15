"""Dynamic discovery of Agent Skill Open Standard implementations.

The loader imports every Python module in the skills directory, instantiates
concrete ``BaseSkill`` subclasses, and indexes them by ``name``. A broken
skill file is logged and skipped so one import error does not take down the
registry.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

from skills.base_skill import BaseSkill

logger = logging.getLogger("agentflow.skill_loader")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def discover_skills(skills_dir: str = "skills") -> dict[str, BaseSkill]:
    """Find, import, and instantiate every concrete skill.

    Args:
        skills_dir: Directory of skill modules, relative to the working
            directory or the project root.

    Returns:
        Mapping of skill name to a live skill instance.

    Raises:
        FileNotFoundError: If neither the given path nor the project-root
            fallback exists.
    """
    root = _resolve_skills_dir(skills_dir)
    project_root = root.parent
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    discovered: dict[str, BaseSkill] = {}
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_") or path.name == "base_skill.py":
            continue
        module_name = f"skills.{path.stem}"
        try:
            module = _load_module(module_name, path)
        except Exception as exc:
            logger.error("Failed to import skill module %s: %s", path.name, exc)
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is BaseSkill or not issubclass(obj, BaseSkill):
                continue
            if obj.__module__ != module.__name__:
                continue
            if inspect.isabstract(obj):
                logger.error("Skipping abstract skill class %s", obj.__name__)
                continue
            try:
                instance = obj()
            except Exception as exc:
                logger.error("Could not instantiate %s: %s", obj.__name__, exc)
                continue
            if instance.name in discovered:
                logger.warning(
                    "Duplicate skill name '%s' from %s; replacing the previous instance",
                    instance.name,
                    path.name,
                )
            discovered[instance.name] = instance
            logger.info("Discovered skill %s v%s from %s", instance.name, instance.version, path.name)
    logger.info("Skill discovery finished with %d skill(s)", len(discovered))
    return discovered


def get_registry(skills: dict[str, BaseSkill]) -> list[dict[str, Any]]:
    """Convert loaded skills into registry entries.

    Args:
        skills: Mapping returned by :func:`discover_skills`.

    Returns:
        One ``to_registry_entry`` dictionary per skill, sorted by name.
    """
    return [skills[name].to_registry_entry() for name in sorted(skills)]


def _resolve_skills_dir(skills_dir: str) -> Path:
    """Resolve the skills directory from the cwd, then the project root."""
    candidate = Path(skills_dir)
    if candidate.is_dir():
        return candidate.resolve()
    fallback = _PROJECT_ROOT / skills_dir
    if fallback.is_dir():
        return fallback.resolve()
    raise FileNotFoundError(f"Skills directory not found: {skills_dir}")


def _load_module(module_name: str, path: Path) -> Any:
    """Import a module from a file path and register it in ``sys.modules``."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
