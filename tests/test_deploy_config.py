"""The deployment files are code too: a typo here only shows up on Render."""

import re
import tomllib

import pytest
import yaml

from warmintro.profiles import ROOT


def load_yaml(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


@pytest.fixture
def service():
    [web] = load_yaml("render.yaml")["services"]
    return web


@pytest.fixture
def env(service):
    return {item["key"]: item for item in service["envVars"]}


def test_free_plan_and_deploys_only_from_the_workflow(service):
    assert service["plan"] == "free"
    assert service["autoDeployTrigger"] == "off"  # a string, not YAML's False


def test_python_version_is_full_and_matches_the_project(env):
    version = env["PYTHON_VERSION"]["value"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)  # Render needs x.y.z
    assert version.startswith((ROOT / ".python-version").read_text().strip() + ".")


def test_build_installs_the_locked_cloud_packages_only(service):
    assert service["buildCommand"] == \
        "pip install uv && uv sync --locked --no-dev --extra gemini"
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"]
    assert "gemini" in extras
    assert not any("huggingface" in dep for dep in extras["gemini"])  # no PyTorch


def test_start_command_runs_the_app(service):
    assert service["startCommand"] == ".venv/bin/python app.py"
    assert (ROOT / "app.py").exists()


def test_secret_is_never_committed(env):
    assert env["GOOGLE_API_KEY"] == {"key": "GOOGLE_API_KEY", "sync": False}


def test_cloud_uses_gemini_and_limits_questions(env):
    assert env["LLM_PROVIDER"]["value"] == env["EMBEDDINGS_PROVIDER"]["value"] == "gemini"
    assert int(env["MAX_QUESTIONS_PER_SESSION"]["value"]) > 0


def test_deploy_runs_tests_first_and_deploys_the_tested_commit():
    workflow = load_yaml(".github/workflows/deploy.yml")
    triggers = workflow[True]  # YAML reads the key "on" as True
    assert triggers["push"]["tags"] == ["v*"]
    assert workflow["jobs"]["test"]["uses"] == "./.github/workflows/tests.yml"
    deploy = workflow["jobs"]["deploy"]
    assert deploy["needs"] == "test"
    step = deploy["steps"][0]
    assert step["env"]["HOOK"] == "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
    assert "&ref=${GITHUB_SHA}" in step["run"] and "--fail" in step["run"]


def test_tests_workflow_can_be_called_by_the_deploy():
    assert "workflow_call" in load_yaml(".github/workflows/tests.yml")[True]


def test_actions_use_full_version_tags():
    # setup-uv publishes only full tags such as v10.3.2, so @v10 fails.
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        for ref in re.findall(r"uses:\s*([\w./-]+@[\w.]+)", path.read_text()):
            if not ref.startswith("./"):
                assert re.search(r"@v\d+\.\d+\.\d+$", ref), f"{path.name}: {ref}"
