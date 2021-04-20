import setuptools
import yaml

def get_dependencies(env_yaml_file):
    """Scan a YAML environment file to get a list of dependencies
    """
    with open(env_yaml_file, "r") as f:
        environment = yaml.safe_load(f)
    dependencies = []
    for dep in environment["dependencies"]:
        if not dep.startswith("python"):
            dependencies.append(dep)
    return dependencies

setuptools.setup(
    name="AutoDQM_ML",
    packages=[
        "dqm_data"
    ],
    install_requires=get_dependencies("environment.yml"),
    python_requires=">=3.8,!=3.9",
)

