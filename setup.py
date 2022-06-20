# Standard imports
import subprocess

# Third party imports
import setuptools


def git(*args):
    return subprocess.check_output(["git"] + list(args))


# get latest tag
latest = git("describe", "--tags").decode().strip()
latest = latest.split("-")[0]

with open("planner/README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Remove extra index urls
requirements = [
    requirement
    for requirement in requirements
    if "--extra-index-url" not in requirement
]

setuptools.setup(
    name="EthicalTrajectoryPlanning",
    version=latest,
    description="Trajectory planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.lrz.de/motionplanning1/EthicalTrajectoryPlanning",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

# EOF
