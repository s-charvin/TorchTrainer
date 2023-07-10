# 验证是否可以正常加载 base 文件
_base_ = [
    "./base.yaml",
]


# 验证是否可以使用基本的变量
item8 = "fileBasename: {{fileBasename}}, fileDirname: {{fileDirname}}, fileBasenameNoExtension:{{fileBasenameNoExtension}}, fileExtname:{{fileExtname}}"
item11 = {{_base_.yaml_item2}}

base = "_base_.item8"

item15 = dict(
    a=dict(b={{_base_.item8}}),
    b=[{{_base_.item9}}],
    c=[{{_base_.item10}}],
    d=[[dict(e={{_base_.yaml_item2}})], {{_base_.item8}}],
)

# 验证环境变量是否可用

item1 = "APPDATA: {{$PATH: }}"
item2 = "PATH: {{ $FASFSAF:default_value }}"


# 验证是否可以使用预留的变量
# filename = "reserved.py"
