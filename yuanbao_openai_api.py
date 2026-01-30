from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, Generator
import uvicorn
import json
import requests
import urllib3
import time
import socket
import logging
import re
import os

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 定义支持的模型列表
SUPPORTED_MODELS = [
    {
        "name": "deepseek_v3",
        "modified_at": "2024-04-25T00:00:00.000000Z",
        "size": 0,
        "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
        "name": "deepseek_r1",
        "modified_at": "2024-04-25T00:00:00.000000Z",
        "size": 0,
        "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
        "name": "deepseek_public_v3",
        "modified_at": "2024-04-25T00:00:00.000000Z",
        "size": 0,
        "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    {
        "name": "deepseek_public_r1",
        "modified_at": "2024-04-25T00:00:00.000000Z",
        "size": 0,
        "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
]

# 定义模型到 chatModelId 的映射
MODEL_TO_CHAT_ID = {
    "deepseek_v3": "deep_seek_v3",
    "deepseek_r1": "deep_seek",
    "deepseek_public_v3": "deep_seek_v3",
    "deepseek_public_r1": "deep_seek"
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yuanbao_api.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)

# 添加测试日志
logger.info("=== API服务启动 ===")
logger.info(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek_v3"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

# 读取模型会话ID配置
def load_model_sessions():
    sessions = {}
    try:
        with open('yuanbao_model_sessions.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    model_name, session_id, x_token = line.split(':', 2)
                    sessions[model_name] = {
                        'session_id': session_id,
                        'x_token': x_token
                    }
    except Exception as e:
        logger.error(f"读取模型会话配置失败: {str(e)}")
    return sessions

# 全局变量存储会话ID映射
MODEL_SESSIONS = load_model_sessions()

def clean_chinese_text(text: str) -> str:
    """清理中文文本，保持良好的格式和段落结构"""
    
    # 处理标题格式
    text = re.sub(r'(\d+\..*?架构)\s*', r'\1\n\n', text)
    
    # 处理步骤标记
    text = re.sub(r'([A-Z]-[^-]*?)-([^-]*?)-', r'\n\1-\2-\n', text)
    
    # 处理步骤内容，在逗号后添加换行
    text = re.sub(r'([^,]*?),((?:[^,]*?,)*[^,]*?)(?=\n|$)', lambda m: f"{m.group(1)}\n{m.group(2)}", text)
    
    # 处理连续的换行符
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 确保最后有一个换行符
    text = text.strip() + '\n'
    
    return text

def send_yuanbao_request(prompt: str, stream: bool = False, model: str = "deepseek_v3") -> Union[str, Generator[str, None, None]]:
    # 获取对应模型的配置
    model_config = MODEL_SESSIONS.get(model)
    if not model_config:
        raise ValueError(f"未找到模型 {model} 的配置")
        
    url = f"https://yuanbao.tencent.com/api/chat/{model_config['session_id']}"
    
    headers = {
        "sec-ch-ua-platform": "Windows",
        "sec-ch-ua": '"Microsoft Edge WebView2";v="135", "Chromium";v="135", "Not-A.Brand";v="8", "Microsoft Edge";v="135"',
        "x-token": model_config['x_token'],
        "x-instance-id": "1",
        "sec-ch-ua-mobile": "?0",
        "x-language": "zh-CN",
        "x-requested-with": "XMLHttpRequest",
        "content-type": "text/plain;charset=UTF-8",
        "x-operationsystem": "win",
        "x-channel": "7001",
        "chat_version": "v1",
        "x-id": "c03ce86d8f624ae1af7c1cbeea4161d7",
        "sidebarwidth": "204",
        "x-a10": "HP-2Q523AV",
        "x-product": "bot",
        "x-appversion": "1.10.0",
        "x-os_version": "Windows_11_Desktop",
        "x-source": "web",
        "x-hy92": "defb4cb89a98bc38c16530540200000c119419",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0 product_id/TM_Product_App app_theme/system app/tencent_yuanbao os_name/windows app_short_version/1.10.0 app_instance_id/2 c_district/0 system_lang/zh-CN os_version/10.0.22000 app_version/1.10.0 package_type/publish_release app_full_version/1.10.0.608 app_lang/zh-CN",
        "x-a3": "78755567e30633de2280724c300013617a17",
        "x-hy93": "1931b0f29271003d9a98bc38c165305447cafa1cb9",
        "x-a9": "HP",
        "accept": "*/*",
        "origin": "https://tencent.yuanbao",
        "sec-fetch-site": "cross-site",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": "https://tencent.yuanbao/",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "priority": "u=1, i"
    }
    
    payload = {
        "agentId": "naQivTmsDa",
        "displayPrompt": prompt,
        "supportFunctions": [""],
        "version": "v2",
        "docOpenid": "144115210554304601",
        "multimedia": [],
        "plugin": "Adaptive",
        "supportHint": 1,
        "displayPromptType": 1,
        "options": {
            "imageIntention": {
                "needIntentionModel": True,
                "backendUpdateFlag": 2,
                "intentionStatus": True
            }
        },
        "model": "gpt_175B_0404",
        "chatModelId": MODEL_TO_CHAT_ID.get(model, "deep_seek_v3"),
        "prompt": prompt
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False, stream=True)
        response.raise_for_status()

        if stream:
            def generate():
                full_response = []
                current_thought = []
                thinking_started = False
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        logger.info(f"原始响应行: {line}")
                        
                        # 跳过非JSON数据
                        if line in ['status', 'text']:
                            continue
                            
                        if line.startswith('data: '):
                            try:
                                data = line[6:]
                                if data:
                                    json_data = json.loads(data)
                                    # 处理思考过程
                                    if json_data.get('type') == 'think':
                                        msg = json_data.get('content', '')
                                        if msg:
                                            if not thinking_started:
                                                # 第一次遇到思考内容时，发送思考开始标记
                                                chunk = {
                                                    "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": "deepseek_v3",
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "delta": {
                                                                "role": "assistant" if len(full_response) == 0 else None,
                                                                "content": "<think>\n"
                                                            },
                                                            "finish_reason": None
                                                        }
                                                    ]
                                                }
                                                thinking_started = True
                                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                            
                                            current_thought.append(msg)
                                            # 当遇到句子结束标记时，发送完整的思考内容
                                            if msg.strip() in ['。', '？', '！', '.', '?', '!']:
                                                thought_text = ''.join(current_thought)
                                                chunk = {
                                                    "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": "deepseek_v3",
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "delta": {
                                                                "content": thought_text + "\n"
                                                            },
                                                            "finish_reason": None
                                                        }
                                                    ]
                                                }
                                                current_thought = []
                                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    
                                    # 处理普通文本消息
                                    elif json_data.get('type') == 'text':
                                        msg = json_data.get('msg', '')
                                        if msg:
                                            # 如果之前有未完成的思考内容，先发送出去
                                            if current_thought:
                                                thought_text = ''.join(current_thought)
                                                chunk = {
                                                    "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": "deepseek_v3",
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "delta": {
                                                                "content": thought_text + "\n"
                                                            },
                                                            "finish_reason": None
                                                        }
                                                    ]
                                                }
                                                current_thought = []
                                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                            
                                            # 如果是第一个文本消息且之前有思考过程，添加思考结束标记和换行
                                            if thinking_started and len(full_response) == 0:
                                                chunk = {
                                                    "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": "deepseek_v3",
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "delta": {
                                                                "content": "</think>\n\n"
                                                            },
                                                            "finish_reason": None
                                                        }
                                                    ]
                                                }
                                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                            
                                            # 发送实际的文本消息
                                            chunk = {
                                                "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": "deepseek_v3",
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": {
                                                            "role": "assistant" if len(full_response) == 0 and not thinking_started else None,
                                                            "content": msg
                                                        },
                                                        "finish_reason": None
                                                    }
                                                ]
                                            }
                                            full_response.append(msg)
                                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解析错误: {str(e)}")
                                continue
                
                # 发送结束标记
                end_chunk = {
                    "id": f"chatcmpl-{str(hash(''.join(full_response)))}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "deepseek_v3",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            
            return generate()
        else:
            full_response = []
            current_thought = []  # 用于收集当前思考过程的词组
            thinking_paragraphs = []  # 用于存储完整的思考段落
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    logger.info(f"原始响应行: {line}")
                    
                    # 跳过非JSON数据
                    if line in ['status', 'text']:
                        continue
                        
                    if line.startswith('data: '):
                        try:
                            data = line[6:]
                            if data:
                                json_data = json.loads(data)
                                # 处理思考过程
                                if json_data.get('type') == 'think':
                                    msg = json_data.get('content', '')
                                    if msg:
                                        # 如果消息以句号、问号或感叹号结尾，说明是一个完整的句子
                                        if msg.strip() in ['。', '？', '！', '.', '?', '!']:
                                            current_thought.append(msg.strip())
                                            # 将收集到的词组组合成完整的句子
                                            if current_thought:
                                                thinking_paragraphs.append(''.join(current_thought))
                                                current_thought = []
                                        else:
                                            current_thought.append(msg)
                                # 处理普通文本消息
                                elif json_data.get('type') == 'text':
                                    msg = json_data.get('msg', '')
                                    if msg:
                                        # 如果还有未处理的思考内容，先处理完
                                        if current_thought:
                                            thinking_paragraphs.append(''.join(current_thought))
                                            current_thought = []
                                        full_response.append(msg)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误: {str(e)}")
                            continue
            
            # 处理可能剩余的思考内容
            if current_thought:
                thinking_paragraphs.append(''.join(current_thought))
            
            # 合并思考过程和最终响应
            result = ""
            if thinking_paragraphs:
                # 将思考段落用换行符连接，每个段落占一行
                result = "<think>\n" + "\n".join(thinking_paragraphs) + "\n</think>\n\n"
            result += ''.join(full_response)
            
            logger.info(f"最终合并结果: {result}")
            
            if not result.strip():
                raise ValueError("Empty response from Yuanbao API")
            return result.strip()
            
    except Exception as e:
        logger.error(f"请求处理错误: {str(e)}")
        raise

async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # 记录完整的请求内容
        logger.info("\n=== 收到Dify请求 ===")
        ######################################## logger.info(f"完整请求内容: {request.model_dump_json(indent=2)}")
        
        # 获取系统提示词
        system_message = next((msg.content for msg in request.messages if msg.role == "system"), None)
        logger.info(f"提取到的系统提示词: {system_message}")
        
        # 获取最后一条用户消息
        user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
        logger.info(f"提取到的用户消息: {user_message}")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # 如果有系统提示词，将其添加到用户消息前
        if system_message:
            user_message = f"{system_message}\n\n用户问题：{user_message}"
            logger.info(f"合并后的完整提示词: {user_message}")

        # 如果是流式请求
        if request.stream:
            return StreamingResponse(
                send_yuanbao_request(user_message, stream=True),
                media_type="text/event-stream"
            )

        # 非流式请求
        response_text = send_yuanbao_request(user_message)
        logger.info(f"元宝API返回的响应: {response_text}")
        logger.info("=== 请求处理完成 ===\n")

        # 构建OpenAI格式的响应
        response = ChatCompletionResponse(
            id="chatcmpl-" + str(hash(response_text)),
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(user_message),
                "completion_tokens": len(response_text),
                "total_tokens": len(user_message) + len(response_text)
            }
        )

        return response
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    try:
        try:
            response_text = send_yuanbao_request(request.prompt, model=request.model)
        except Exception as e:
            response_text = str(e)
            
        return {
            "model": request.model,
            "response": response_text,
            "done": True
        }
    except Exception as e:
        return {
            "model": request.model,
            "response": str(e),
            "done": True
        }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # 获取系统提示词
        system_message = next((msg.content for msg in request.messages if msg.role == "system"), None)
        logger.info(f"提取到的系统提示词: {system_message}")
        
        # 获取最后一条用户消息
        user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
        logger.info(f"提取到的用户消息: {user_message}")
        
        if not user_message:
            return {
                "model": request.model,
                "message": {
                    "role": "assistant",
                    "content": "No user message found"
                },
                "done": True
            }

        # 如果有系统提示词，将其添加到用户消息前
        if system_message:
            user_message = f"{system_message}\n\n用户问题：{user_message}"
            logger.info(f"合并后的完整提示词: {user_message}")

        try:
            response_text = send_yuanbao_request(user_message, model=request.model)
        except Exception as e:
            response_text = str(e)
            
        logger.info(f"元宝API返回的响应: {response_text}")

        return {
            "model": request.model,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "done": True
        }
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return {
            "model": request.model,
            "message": {
                "role": "assistant",
                "content": str(e)
            },
            "done": True
        }

@app.get("/api/tags")
async def get_models():
    """返回支持的模型列表"""
    return {"models": SUPPORTED_MODELS}

@app.get("/api/version")
async def version():
    return {
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Yuanbao API is running"}

def get_ip():
    try:
        # 获取主机名
        hostname = socket.gethostname()
        # 获取IP地址
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except:
        return "获取IP失败"

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # 记录请求信息
    logger.info("\n=== 收到请求 ===")
    logger.info(f"请求方法: {request.method}")
    logger.info(f"请求路径: {request.url.path}")
    # logger.info(f"请求头: {dict(request.headers)}")
    
    # 如果是POST请求，记录请求体
    if request.method == "POST":
        try:
            body = await request.body()
            # 使用json.dumps处理Unicode字符，使中文显示更友好
            logger.info(f"请求体: {json.dumps(json.loads(body.decode()), ensure_ascii=False, indent=2)}")
        except Exception as e:
            logger.error(f"读取请求体时出错: {str(e)}")
    
    # 处理请求
    response = await call_next(request)
    
    # 记录响应信息
    process_time = time.time() - start_time
    logger.info(f"处理时间: {process_time:.2f}秒")
    logger.info("=== 请求处理完成 ===\n")
    
    return response

@app.post("/v1/chat/completions")
async def openai_chat_completion(request: ChatCompletionRequest):
    try:
        logger.info("\n=== 收到OpenAI兼容请求 ===")
        logger.info(f"完整请求内容: {request.model_dump_json(indent=2)}")
        
        # 获取用户消息
        user_message = request.messages[-1].content if request.messages else None
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # 如果是流式请求
        if request.stream:
            return StreamingResponse(
                send_yuanbao_request(user_message, stream=True, model=request.model),
                media_type="text/event-stream"
            )
        
        # 非流式请求
        try:
            response_text = send_yuanbao_request(user_message, model=request.model)
        except Exception as e:
            # 将错误信息作为正常响应返回
            response_text = str(e)
        
        response_data = {
            "id": f"chatcmpl-{str(hash(response_text))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message),
                "completion_tokens": len(response_text),
                "total_tokens": len(user_message) + len(response_text)
            }
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        logger.error(f"错误类型: {type(e)}")
        logger.error(f"错误详情: {str(e)}")
        # 将错误信息作为正常响应返回
        response_data = {
            "id": f"chatcmpl-{str(hash(str(e)))}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(e)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message) if user_message else 0,
                "completion_tokens": len(str(e)),
                "total_tokens": (len(user_message) if user_message else 0) + len(str(e))
            }
        }
        return JSONResponse(content=response_data)

if __name__ == "__main__":
    ip = get_ip()
    logger.info(f"服务器IP地址: {ip}")
    logger.info("服务即将启动...")
    
    logger.info("\n=== Yuanbao API Server ===")
    logger.info(f"Local IP address: {ip}")
    logger.info(f"Server will be available at:")
    logger.info(f"- Local: http://127.0.0.1:9999")
    logger.info(f"- Network: http://{ip}:9999")
    logger.info(f"- Docker: http://host.docker.internal:9999")
    logger.info("\nFor Dify in Docker, use either Network or Docker address")
    logger.info("You can test the server using:")
    logger.info(f"curl http://{ip}:9999/health")
    logger.info("===============================\n")
    
    logger.info("服务已启动，等待请求...")
    uvicorn.run(app, host="0.0.0.0", port=9999) 