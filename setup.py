from setuptools import setup

setup(
    name="ttracker",
    version="0.1.0",
    py_modules=["ttracker"],  # 指定单个文件作为模块
    install_requires=[
        "torch>=2.0.0",        # 需要GPU支持时添加版本约束
        "numpy>=1.21.0",
        "icecream>=2.1.3",
        # 其他隐式依赖（如系统自带库）无需声明
    ],
)
