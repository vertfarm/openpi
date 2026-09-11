from setuptools import find_packages
from setuptools import setup

setup(
    name="keti_humanoid_inference",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/keti_humanoid_inference"]),
        ("share/keti_humanoid_inference", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="HV1 deployment team",
    maintainer_email="maintainer@example.invalid",
    description="Shadow-first HV1 OpenPI deployment adapter",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vla_client = keti_humanoid_inference.node:main",
            "vla_operator = keti_humanoid_inference.operator:main",
            "vla_guardian = keti_humanoid_inference.guardian:main",
        ]
    },
)
