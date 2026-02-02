# clawd_free_ai
基于腾讯元宝手搓自己的 Deepseek OpenAI 接口，适用于日常学习与测试场景

# Deepseek OpenAI API 接口服务

基于腾讯元宝实现的 Deepseek OpenAI 接口，适用于日常学习与测试场景。

## 项目简介

本项目通过把腾讯元宝的客户端，封装为一个 OpenAI 的 API 接口，对外供了一个兼容 OpenAI 他 Ollama 接口规范的服务，使您可以在本地环境中轻松使用 Deepseek 模型进行开发和测试，适用于学习和测试目的，请勿商用。

## 功能特点

- ✅ 兼容 OpenAI 聊天接口规范
- ✅ 支持流式响应（stream）
- ✅ 提供多种 API 端点
- ✅ 支持多个 Deepseek 模型
- ✅ 自动处理思考过程（think）和文本响应
- ✅ 完善的日志记录
- ✅ 健康检查接口

## 支持的模型

- deepseek_v3
- deepseek_r1

## 安装依赖

### 方法一：使用 pip 安装

```bash
pip install fastapi uvicorn requests pydantic
```

### 方法二：使用 requirements.txt

1. 创建 `requirements.txt` 文件
2. 添加以下内容：

```
fastapi
uvicorn
requests
pydantic
```
3. 执行安装命令：

```bash
pip install -r requirements.txt
```

## 配置方法

### 1. 配置模型会话

在项目根目录创建 `yuanbao_model_sessions.txt` 文件，并添加您的模型会话配置：

```
# 格式：
x_token:your_x_token
```

**获取 hy_token 的方法：**
1. 登录腾讯元宝网站
2. 打开浏览器开发者工具（F12）
3. 切换到 Network 标签页
4. 发送一条消息给 Deepseek 模型
5. 找到 `https://yuanbao.tencent.com/api/user/agent/conversation/v1/detail` 请求
6. 从该请求的 Cookie 中获取 `hy_token`
<img width="1920" height="810" alt="image" src="https://github.com/user-attachments/assets/b1bd11b7-6682-4bf5-9eb9-6f6fd7c4deb1" />

在Cookie 中找到 `hy_token`
<img width="1896" height="459" alt="image" src="https://github.com/user-attachments/assets/8683065f-fefe-4647-8108-5f612e383ac0" />
8. 复制hy_token值到配置文件中

### 2. 启动服务

#### 方法一：直接运行

```bash
python yuanbao_openai_api.py
```

#### 方法二：使用批处理文件

双击运行 `restart.bat` 文件。

## 使用方法

### OpenAI 兼容接口

**接口地址：** `http://localhost:9999/v1/chat/completions`

**请求示例：** `test.py`

### 其他 API 端点

- **健康检查：** `GET http://localhost:9999/health`
- **获取模型列表：** `GET http://localhost:9999/api/tags`
- **获取版本信息：** `GET http://localhost:9999/api/version`
- **简单生成：** `POST http://localhost:9999/api/generate`
- **聊天接口：** `POST http://localhost:9999/api/chat`

## 项目结构

```
deepseek_apiServer/
├── yuanbao_openai_api.py    # 主服务文件
├── yuanbao_model_sessions.txt  # 模型会话配置
├── yuanbao_api.log          # 日志文件
├── restart.bat              # 重启脚本
├── test.py                  # 测试脚本  
└── README.md                # 项目说明
```

## 注意事项

1. **仅供学习和测试使用**：本项目设计用于开发和测试环境，不建议在生产环境中使用。

2. **配置安全**：请妥善保管您的 `yuanbao_model_sessions.txt` 文件，不要将其提交到版本控制系统中。

3. **网络连接**：使用前请确保您的网络可以访问腾讯元宝服务。

4. **模型限制**：使用频率和请求量可能受到腾讯元宝服务的限制。

5. **错误处理**：如果遇到 API 调用失败，请检查您的网络连接和配置信息。

## 常见问题

### Q: 服务启动失败怎么办？
A: 请检查端口 9999 是否已被占用，或修改代码中的端口配置。

### Q: 如何获取 session_id 和 hy_token？
A: 请参考配置方法中的说明，通过浏览器开发者工具获取。

### Q: 支持哪些模型？
A: 目前支持 deepseek_v3、deepseek_r1。

## 许可证

本项目仅供学习和测试使用，基于 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

**祝您使用愉快！** 🚀
