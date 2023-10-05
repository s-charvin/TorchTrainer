# 整体架构

Config: 默认配置文件管理工具

- `Config.fromfile` 支持解析 `python`、`json` 和 `yaml` 类型的配置文件或字符串. 
- `Config.fromstring` 支持解析 `python`、`json` 和 `yaml` 类型的配置字符串. 
- 支持类似于 `Config.key` 的属性访问操作. 
- 支持在配置文件中使用 `{{var}}` 的方式预定义变量, 例如 `fileDirname`、`fileBasename`、`fileBasenameNoExtension`、`fileExtname`. 
- 支持在配置文件中使用 `{{var:default}}` 的方式使用环境变量, 其中 "default" 是环境变量未找到时的默认值. 
- 支持将多个配置文件合并为一个配置文件. 

# 开发

```bash
git checkout -b fea dev
git add -A .
git commit -m "fix"
git checkout dev
git merge --no-ff fea
git branch -d fea
git push origin dev
```