from setuptools import setup, find_packages

setup(
    name="sycon_bench",
    # Use 'where' for the src-layout, or omit for the flat layout
    packages=(
        find_packages(where="src")
        if "src" in __import__("os").listdir()
        else find_packages()
    ),
    package_dir={"": "src"} if "src" in __import__("os").listdir() else {},
    version="0.1.0",
)
